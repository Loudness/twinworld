"""Counterfactual-explanation metrics (thesis Experiment 3).

Every evaluator is generator-independent (CARLA's Benchmark discipline) and,
because the domain is deterministic and finite, **validity is a decidable
certificate, not an estimate** — the edited program either reproduces all
demonstration pairs or it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Counterfactual, Program, Solution
from .representation import StateGraph, as_grid, match_objects, parse_grid


def validity(solution: Solution, cf: Counterfactual) -> bool | None:
    """Certificate: does the counterfactual world still solve the task?

    Interventional mode: the edited program must reproduce every train pair.
    Backtracking mode: validity of the fixed program on an edited input is not
    defined by the task, so None is returned (the caller reports it as n/a).
    """
    if cf.mode != "interventional":
        return None
    if not cf.applicable:
        return False
    for grid_in, grid_out in solution.task.train:
        trace = solution.cache.run(
            _parse_like(solution, grid_in), cf.program
        )
        if trace is None or trace.outcome.key != as_grid(grid_out):
            return False
    return True


def sparsity(factual: Program, counterfactual: Program) -> int:
    """Number of action edits between the two programs (edit distance at the
    action granularity — Tsirtsis et al.'s ≤k-edit notion, not feature L0)."""
    edits = sum(1 for a, b in zip(factual, counterfactual) if a != b)
    return edits + abs(len(factual) - len(counterfactual))


def proximity(a: StateGraph, b: StateGraph) -> float:
    """Approximate object-graph edit distance between two states.

    Matched objects contribute the number of differing properties
    (colour, shape, location); unmatched objects cost 2 each.
    Greedy matching — an upper bound on the true edit distance.
    """
    cost = 0.0
    for x, y in match_objects(a, b):
        if x is None or y is None:
            cost += 2.0
            continue
        cost += (x.colour != y.colour) + (x.shape != y.shape) + (x.location != y.location)
    return cost


@dataclass(frozen=True)
class MetricVector:
    validity: bool | None
    sparsity: int
    proximity: float
    divergence_step: int
    applicable: bool

    def as_dict(self) -> dict:
        return {
            "validity": self.validity,
            "sparsity": self.sparsity,
            "proximity": self.proximity,
            "divergence_step": self.divergence_step,
            "applicable": self.applicable,
        }


def evaluate(solution: Solution, cf: Counterfactual) -> MetricVector:
    prox = (
        proximity(cf.factual.outcome, cf.counterfactual.outcome)
        if cf.applicable
        else float("inf")
    )
    if cf.mode == "representational":
        # a solving program exists under the alternative segmentation, or not
        valid = cf.applicable
    else:
        valid = validity(solution, cf)
    return MetricVector(
        validity=valid,
        sparsity=sparsity(cf.factual.mechanisms, cf.program),
        proximity=prox,
        divergence_step=cf.divergence_step,
        applicable=cf.applicable,
    )


def _parse_like(solution: Solution, grid) -> StateGraph:
    return parse_grid(grid, solution.train_traces[0].states[0].abstraction)
