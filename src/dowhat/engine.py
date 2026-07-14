"""Counterfactual engine over state-transition traces (thesis Experiment 3 core).

A solving episode is a :class:`Trace` — states connected by mechanism
applications. Because every transition is a pure function of its predecessor,
the trace IS a structural causal model: exogenous terms are the input grid and
the program (abduction is exact trace replay, so counterfactuals are
point-identified), and ``do()`` is twin-world forking — the counterfactual
trajectory shares the factual prefix by reference and recomputes only the
suffix (Balke & Pearl's twin network, materialized).

Two labeled counterfactual modes (von Kügelgen et al. 2023):
  - interventional: replace the mechanism at one step, keep the input;
  - backtracking:   replace the input, keep the program (fixed laws).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import networkx as nx

from .mechanisms import Mechanism
from .representation import Grid, StateGraph, as_grid, parse_grid

Program = tuple[Mechanism, ...]


@dataclass(frozen=True)
class Task:
    """An ARC-style task: demonstration pairs and held-out test pairs."""

    train: tuple[tuple[Grid, Grid], ...]
    test: tuple[tuple[Grid, Grid], ...]
    task_id: str = "synthetic"

    def colours(self) -> set[int]:
        return {
            c
            for pairs in (self.train, self.test)
            for pair in pairs
            for grid in pair
            for row in grid
            for c in row
        }


@dataclass(frozen=True)
class Trace:
    """One episode: states[0] --mechanisms[0]--> states[1] --...--> states[-1]."""

    states: tuple[StateGraph, ...]
    mechanisms: Program

    @property
    def outcome(self) -> StateGraph:
        return self.states[-1]

    def __len__(self) -> int:
        return len(self.mechanisms)


class ApplyCache:
    """Memoized mechanism application keyed by (state grid, mechanism)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Grid, Mechanism], StateGraph | None] = {}
        self.dag = nx.DiGraph()

    def apply(self, state: StateGraph, mech: Mechanism) -> StateGraph | None:
        key = (state.key, mech)
        if key not in self._cache:
            nxt = mech.apply(state)
            self._cache[key] = nxt
            self.dag.add_node(state.key, state=state)
            if nxt is not None:
                self.dag.add_node(nxt.key, state=nxt)
                if self.dag.has_edge(state.key, nxt.key):
                    self.dag.edges[state.key, nxt.key]["mechanisms"].add(mech)
                else:
                    self.dag.add_edge(state.key, nxt.key, mechanisms={mech})
        return self._cache[key]

    def run(self, state: StateGraph, program: Sequence[Mechanism]) -> Trace | None:
        states = [state]
        for mech in program:
            nxt = self.apply(states[-1], mech)
            if nxt is None:
                return None
            states.append(nxt)
        return Trace(tuple(states), tuple(program))


@dataclass
class Solution:
    """A fitted model of the task: the induced program plus everything observed."""

    task: Task
    program: Program
    train_traces: tuple[Trace, ...]
    test_traces: tuple[Trace, ...]
    cache: ApplyCache
    programs_tried: int = 0
    strategy: str = "enumerate"  # "enumerate" (blind) | "analogy" (induced rules)

    @property
    def dag(self) -> nx.DiGraph:
        """Trajectory DAG: every state expansion recorded during search."""
        return self.cache.dag


class UnsolvedTaskError(Exception):
    pass


def solve(
    task: Task,
    primitives: Sequence[Mechanism],
    max_depth: int = 2,
    abstraction: str = "cc4",
    strategy: str = "enumerate",
) -> Solution:
    """Breadth-first program induction: the shortest program mapping every train
    input to its train output, verified (not fitted) on the test pair(s).

    Every candidate expansion lands in the trajectory DAG, so counterfactual
    search later works over ground the solver has actually visited.
    """
    cache = ApplyCache()
    train_inputs = [parse_grid(i, abstraction) for i, _ in task.train]
    tried = 0
    for depth in range(1, max_depth + 1):
        for program in product(primitives, repeat=depth):
            tried += 1
            traces = []
            for state, (_, expected) in zip(train_inputs, task.train):
                trace = cache.run(state, program)
                if trace is None or trace.outcome.key != as_grid(expected):
                    break
                traces.append(trace)
            else:
                test_traces = tuple(
                    t
                    for i, _ in task.test
                    if (t := cache.run(parse_grid(i, abstraction), program)) is not None
                )
                return Solution(task, program, tuple(traces), test_traces, cache, tried, strategy)
    raise UnsolvedTaskError(
        f"no program of depth <= {max_depth} over {len(primitives)} primitives "
        f"solves task {task.task_id} ({tried} programs tried)"
    )


@dataclass(frozen=True)
class Counterfactual:
    """A factual/counterfactual trace pair with its divergence bookkeeping."""

    mode: str  # "interventional" | "backtracking"
    factual: Trace
    counterfactual: Trace | None  # None: the counterfactual world is inapplicable
    divergence_step: int
    program: Program  # program of the counterfactual world

    @property
    def applicable(self) -> bool:
        return self.counterfactual is not None


def intervene(
    solution: Solution, trace: Trace, step: int, alternative: Mechanism
) -> Counterfactual:
    """do(A_step = alternative): twin-world fork sharing the factual prefix.

    Abduction is trivial (exogenous input is stored), action replaces the
    mechanism at ``step``, prediction re-runs only the suffix.
    """
    if not 0 <= step < len(trace):
        raise IndexError(f"step {step} outside trace of length {len(trace)}")
    prefix = trace.states[: step + 1]  # shared by reference: the twin network
    cf_program = trace.mechanisms[:step] + (alternative,) + trace.mechanisms[step + 1 :]
    suffix = solution.cache.run(prefix[-1], cf_program[step:])
    if suffix is None:
        return Counterfactual("interventional", trace, None, step, cf_program)
    cf_trace = Trace(prefix[:-1] + suffix.states, cf_program)
    return Counterfactual("interventional", trace, cf_trace, step, cf_program)


def backtrack(solution: Solution, trace: Trace, edited_input: Grid) -> Counterfactual:
    """Backtracking counterfactual: same laws (program), different initial world."""
    start = parse_grid(edited_input, trace.states[0].abstraction)
    cf_trace = solution.cache.run(start, trace.mechanisms)
    return Counterfactual("backtracking", trace, cf_trace, 0, trace.mechanisms)
