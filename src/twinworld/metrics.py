"""Counterfactual-explanation metrics (thesis Experiment 3).

Every evaluator is generator-independent (CARLA's Benchmark discipline) and,
because the domain is deterministic and finite, **validity is a decidable
certificate, not an estimate** — the edited program either reproduces all
demonstration pairs or it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import representation_of
from .engine import Counterfactual, Program, Solution
from .representation import StateGraph


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
    rep = representation_of(solution.task)
    for grid_in, grid_out in solution.task.train:
        trace = solution.cache.run(
            _parse_like(solution, grid_in), cf.program
        )
        if trace is None or trace.outcome.key != rep.canon(grid_out):
            return False
    return True


def sparsity(factual: Program, counterfactual: Program) -> int:
    """Number of action edits between the two programs (edit distance at the
    action granularity — Tsirtsis et al.'s ≤k-edit notion, not feature L0)."""
    edits = sum(1 for a, b in zip(factual, counterfactual) if a != b)
    return edits + abs(len(factual) - len(counterfactual))


def proximity(a: StateGraph, b: StateGraph) -> float:
    """Approximate object-graph edit distance between two states.

    Delegates to the backend's ``distance`` when it defines one (the grid
    backend's reproduces the historical matched-object property diffs);
    otherwise a generic fallback matches entities by oid and charges 1 per
    differing attribute and 2 per unmatched entity.
    """
    backend = representation_of(a)
    dist = getattr(backend, "distance", None)
    if dist is not None:
        return dist(a, b)
    return _attribute_distance(a, b)


def _attribute_distance(a, b) -> float:
    by_oid = {o.oid: o for o in b.objects}
    cost = 0.0
    matched = 0
    for x in a.objects:
        y = by_oid.get(x.oid)
        if y is None:
            cost += 2.0
            continue
        matched += 1
        keys = set(x.attributes) | set(y.attributes)
        cost += sum(1.0 for k in keys if x.attributes.get(k) != y.attributes.get(k))
    cost += 2.0 * (len(b.objects) - matched)
    return cost


@dataclass(frozen=True)
class MetricVector:
    validity: bool | None
    sparsity: int
    proximity: float
    divergence_step: int
    applicable: bool
    # certificates, not manifold scores (None = the backend/solution cannot say):
    plausible: bool | None = None  # backend constraint-consistency of the CF outcome
    reachable: bool | None = None  # CF outcome was visited during the original search

    def as_dict(self) -> dict:
        return {
            "validity": self.validity,
            "sparsity": self.sparsity,
            "proximity": self.proximity,
            "divergence_step": self.divergence_step,
            "applicable": self.applicable,
            "plausible": self.plausible,
            "reachable": self.reachable,
        }


def edited_steps(cf: Counterfactual) -> frozenset[int]:
    """Program positions where the counterfactual world differs from the factual."""
    return frozenset(
        i for i, (a, b) in enumerate(zip(cf.factual.mechanisms, cf.program)) if a != b
    )


def responsibility_profile(cfs: list[Counterfactual]) -> dict[int, float]:
    """Chockler–Halpern degree of responsibility of each program step for the
    contrastive outcome: 1/k for a step appearing in a minimal edit set of
    size k, 0 for steps appearing in none (Chockler & Halpern, JAIR 2004;
    modified-HP causes per Halpern 2015 — in this deterministic single-path
    setting the minimal edit sets play the role of cause-plus-witness sets)."""
    if not cfs:
        return {}
    profile = {t: 0.0 for t in range(len(cfs[0].factual.mechanisms))}
    for cf in cfs:
        edits = edited_steps(cf)
        if not edits:
            continue
        share = 1.0 / len(edits)
        for t in edits:
            profile[t] = max(profile[t], share)
    return profile


def diversity(cfs: list[Counterfactual]) -> float:
    """Mean pairwise outcome proximity across a returned counterfactual set
    (DiCE's requirement that explainers offer genuinely different worlds)."""
    outcomes = [cf.counterfactual.outcome for cf in cfs if cf.applicable]
    if len(outcomes) < 2:
        return 0.0
    distances = [
        proximity(a, b) for i, a in enumerate(outcomes) for b in outcomes[i + 1 :]
    ]
    return sum(distances) / len(distances)


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
    plausible = None
    reachable = None
    if cf.applicable:
        outcome = cf.counterfactual.outcome
        plausible_fn = getattr(representation_of(solution.task), "plausible", None)
        if plausible_fn is not None:
            plausible = plausible_fn(outcome)
        if solution.searched is not None:
            reachable = outcome.key in solution.searched
    return MetricVector(
        validity=valid,
        sparsity=sparsity(cf.factual.mechanisms, cf.program),
        proximity=prox,
        divergence_step=cf.divergence_step,
        applicable=cf.applicable,
        plausible=plausible,
        reachable=reachable,
    )


def _ok(m: MetricVector) -> bool:
    return m.applicable and m.validity is not False


def dominates(a: MetricVector, b: MetricVector) -> bool:
    """Exact Pareto dominance over the counterfactual desiderata.

    Applicability + non-failed validity is a hard filter; among survivors,
    lower sparsity, lower proximity, and higher plausibility dominate. A None
    plausibility is neutral — never strictly better or worse — so backends
    without the capability neither win nor lose on it.
    """
    if _ok(a) != _ok(b):
        return _ok(a)
    if not _ok(a):
        return False
    plaus_ge, plaus_gt = True, False
    if a.plausible is not None and b.plausible is not None:
        plaus_ge = a.plausible >= b.plausible
        plaus_gt = a.plausible > b.plausible
    at_least = a.sparsity <= b.sparsity and a.proximity <= b.proximity and plaus_ge
    strictly = a.sparsity < b.sparsity or a.proximity < b.proximity or plaus_gt
    return at_least and strictly


def pareto_front(items: list) -> tuple:
    """The exact Pareto front of a set of alternatives, in original order.

    Items may be MetricVectors or anything carrying a ``.metrics`` vector
    (e.g. CounterfactualItem). Because twinworld's alternative sets are
    exhaustively enumerated within declared bounds, this is the TRUE front,
    not an approximation.
    """

    def vec(x) -> MetricVector:
        return getattr(x, "metrics", x)

    return tuple(
        x
        for i, x in enumerate(items)
        if not any(dominates(vec(y), vec(x)) for j, y in enumerate(items) if j != i)
    )


def _parse_like(solution: Solution, grid) -> StateGraph:
    rep = representation_of(solution.task)
    return rep.parse(grid, solution.train_traces[0].states[0].abstraction)
