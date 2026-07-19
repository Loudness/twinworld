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
from itertools import combinations, product
from typing import Sequence

import networkx as nx

from .backend import Raw, representation_of
from .mechanisms import Mechanism, PreimageBudget
from .representation import Grid, StateGraph

Program = tuple[Mechanism, ...]


@dataclass(frozen=True)
class Task:
    """An ARC-style task: demonstration pairs and held-out test pairs."""

    train: tuple[tuple[Grid, Grid], ...]
    test: tuple[tuple[Grid, Grid], ...]
    task_id: str = "synthetic"
    representation: str = "grid"

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
    # state keys visited DURING SEARCH — snapshot at solve() return. The live
    # cache.dag keeps growing as counterfactuals are computed through it, so
    # reachability certificates must use this frozen set, never the live DAG.
    searched: frozenset | None = None

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
    abstraction: str | None = None,
    strategy: str = "enumerate",
) -> Solution:
    """Breadth-first program induction: the shortest program mapping every train
    input to its train output, verified (not fitted) on the test pair(s).

    Every candidate expansion lands in the trajectory DAG, so counterfactual
    search later works over ground the solver has actually visited.
    """
    rep = representation_of(task)
    abstraction = abstraction or rep.default_abstractions[0]
    cache = ApplyCache()
    train_inputs = [rep.parse(i, abstraction) for i, _ in task.train]
    tried = 0
    for depth in range(1, max_depth + 1):
        for program in product(primitives, repeat=depth):
            tried += 1
            traces = []
            for state, (_, expected) in zip(train_inputs, task.train):
                trace = cache.run(state, program)
                if trace is None or trace.outcome.key != rep.canon(expected):
                    break
                traces.append(trace)
            else:
                test_traces = tuple(
                    t
                    for i, _ in task.test
                    if (t := cache.run(rep.parse(i, abstraction), program)) is not None
                )
                return Solution(
                    task, program, tuple(traces), test_traces, cache, tried, strategy,
                    searched=frozenset(cache.dag.nodes),
                )
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


def backtrack(solution: Solution, trace: Trace, edited_input: Raw) -> Counterfactual:
    """Backtracking counterfactual: same laws (program), different initial world."""
    start = representation_of(solution.task).parse(edited_input, trace.states[0].abstraction)
    cf_trace = solution.cache.run(start, trace.mechanisms)
    return Counterfactual("backtracking", trace, cf_trace, 0, trace.mechanisms)


def minimal_edits(
    solution: Solution,
    trace: Trace,
    target: Raw,
    pool: Sequence[Mechanism],
    k_max: int = 2,
    max_results: int = 16,
) -> tuple[int | None, list[Counterfactual]]:
    """All smallest program-edit sets that make ``trace`` end at ``target``.

    Edits are substitutions at program positions from ``pool`` (Tsirtsis et
    al.'s action-edit notion). The search is breadth-first over edit-set size
    k and exhaustive within each k, so the results at the first non-empty k
    are CERTIFIED minimal — and ``(None, [])`` is a certificate that the
    target is unreachable within ``k_max`` edits over this pool. Completeness
    of the returned *set* is capped at ``max_results`` per k; minimality of k
    itself is never affected by the cap.
    """
    target = representation_of(solution.task).canon(target)
    pool = list(dict.fromkeys(pool))
    n = len(trace.mechanisms)
    for k in range(1, min(k_max, n) + 1):
        found: list[Counterfactual] = []
        for positions in combinations(range(n), k):
            for replacements in product(pool, repeat=k):
                if any(replacements[i] == trace.mechanisms[p] for i, p in enumerate(positions)):
                    continue  # not a real edit at that position
                program = list(trace.mechanisms)
                for i, p in enumerate(positions):
                    program[p] = replacements[i]
                cf_trace = solution.cache.run(trace.states[0], program)
                if cf_trace is not None and cf_trace.outcome.key == target:
                    found.append(
                        Counterfactual("contrastive", trace, cf_trace, positions[0], tuple(program))
                    )
                    if len(found) >= max_results:
                        break
            if len(found) >= max_results:
                break
        if found:
            return k, found
    return None, []


def abduce_inputs(
    program: Sequence[Mechanism],
    final_state: StateGraph,
    limit: int = 16,
    budget: "PreimageBudget | None" = None,
) -> list[StateGraph]:
    """Time travel backwards: inputs the program maps to ``final_state``,
    enumerated by chaining mechanism preimages right-to-left. Completeness is
    bounded by each mechanism's ``exact_preimage`` declaration — with delete
    abduction this now works through non-invertible steps too."""
    frontier: list[StateGraph] = [final_state]
    for mech in reversed(tuple(program)):
        # round-robin across frontier states, so no single state's (possibly
        # long) preimage stream starves the others before the limit is hit
        generators = [mech.preimage(state, budget) for state in frontier]
        collected: list[StateGraph] = []
        while generators and len(collected) < limit:
            still_active = []
            for gen in generators:
                try:
                    pre = next(gen)
                except StopIteration:
                    continue
                still_active.append(gen)
                if pre not in collected:
                    collected.append(pre)
                    if len(collected) >= limit:
                        break
            generators = still_active
        frontier = collected
        if not frontier:
            return []
    return frontier[:limit]


def solve_all(
    task: Task,
    primitives: Sequence[Mechanism],
    max_depth: int = 2,
    abstraction: str | None = None,
    limit: int = 32,
) -> list[Program]:
    """Every program (up to ``limit``) that fits all train pairs, at the
    minimal depth where any fits — the raw material for underdetermination
    diagnosis: programs indistinguishable on the demonstrations can only be
    told apart counterfactually."""
    rep = representation_of(task)
    abstraction = abstraction or rep.default_abstractions[0]
    cache = ApplyCache()
    train_inputs = [rep.parse(i, abstraction) for i, _ in task.train]
    expected = [rep.canon(o) for _, o in task.train]
    for depth in range(1, max_depth + 1):
        fits: list[Program] = []
        for program in product(primitives, repeat=depth):
            for state, want in zip(train_inputs, expected):
                trace = cache.run(state, program)
                if trace is None or trace.outcome.key != want:
                    break
            else:
                fits.append(tuple(program))
                if len(fits) >= limit:
                    break
        if fits:
            return fits
    return []
