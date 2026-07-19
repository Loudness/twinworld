"""Uniform evaluation of counterfactual ALTERNATIVE SETS (docs/beyond-grids.md §6).

Alternatives arise in four places — contrastive edit sets, preimage
candidates, underdetermined program classes, pertinent-negative witnesses —
and historically each carried only its generation order. This module gives
them one container with named score columns, an exact Pareto front (the sets
are exhaustively enumerated within declared bounds, so the front is true, not
an NSGA-II approximation), and a common ``Policy`` ranking interface that the
six selection policies of :mod:`twinworld.select` plug into unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Protocol

from .engine import abduce_inputs
from .metrics import pareto_front, proximity


class Policy(Protocol):
    """Rank alternatives: return the index of the chosen item."""

    def __call__(self, alts: "Alternatives", rng: random.Random) -> int: ...


@dataclass(frozen=True)
class Alternatives:
    """A set of competing alternatives, in generation order (the historical
    implicit rank), with named per-item score columns.

    ``kind`` ∈ {"program_classes", "edit_sets", "preimages", "pn_witnesses"}.
    ``context`` carries whatever richer object produced the set (a
    CounterfactualSet, a (rep, report) pair, ...) for policies that need it.
    """

    kind: str
    items: tuple
    scores: tuple[dict[str, object], ...] = ()
    context: Any = None

    def column(self, name: str) -> tuple:
        if not self.scores or any(name not in row for row in self.scores):
            raise ValueError(f"alternatives of kind {self.kind!r} carry no {name!r} score")
        return tuple(row[name] for row in self.scores)

    def pareto(self) -> "Alternatives":
        """The exact Pareto front (items must carry MetricVectors)."""
        front = pareto_front(list(self.items))
        keep = [i for i, item in enumerate(self.items) if any(item is f for f in front)]
        return Alternatives(
            self.kind,
            tuple(self.items[i] for i in keep),
            tuple(self.scores[i] for i in keep) if self.scores else (),
            self.context,
        )

    def ranked(self, policy: "Policy | str", rng: random.Random | None = None) -> tuple[int, ...]:
        """Index order under ``policy``: the chosen index first, the rest in
        generation order. ``policy`` may be a name from GENERIC_POLICIES."""
        fn = GENERIC_POLICIES[policy] if isinstance(policy, str) else policy
        chosen = fn(self, rng if rng is not None else random.Random(0))
        return (chosen, *(i for i in range(len(self.items)) if i != chosen))


def as_policy(select_policy: Callable) -> Policy:
    """Adapt a :mod:`twinworld.select`-style ``(rep, report, rng)`` policy to the
    Alternatives interface; the (rep, report) pair rides on ``context``."""

    def policy(alts: Alternatives, rng: random.Random) -> int:
        rep, report = alts.context
        return select_policy(rep, report, rng)

    return policy


def _min_proximity(alts: Alternatives, rng: random.Random) -> int:
    column = alts.column("proximity")
    return column.index(min(column))


def _max_plausibility(alts: Alternatives, rng: random.Random) -> int:
    order = {True: 2, None: 1, False: 0}
    column = [order[p] for p in alts.column("plausible")]
    return column.index(max(column))


def _pareto_then_first(alts: Alternatives, rng: random.Random) -> int:
    front = alts.pareto()
    if not front.items:
        return 0
    first = front.items[0]
    return next(i for i, item in enumerate(alts.items) if item is first)


GENERIC_POLICIES: dict[str, Policy] = {
    "min_proximity": _min_proximity,
    "max_plausibility": _max_plausibility,
    "pareto_then_first": _pareto_then_first,
}


def class_alternatives(rep, report) -> Alternatives:
    """The underdetermined program classes of a ConfidenceReport as an
    Alternatives set — the same rows the selection policies read."""
    discrimination = report.discrimination
    scores = tuple(
        {
            "programs": len(cls),
            "min_len": min(len(p) for p in cls),
            "stability": sum(1 for out in sig if out is not None),
        }
        for cls, sig in zip(discrimination.classes, discrimination.signatures)
    )
    return Alternatives("program_classes", discrimination.classes, scores, context=(rep, report))


def rank_preimages(mechanism, state, budget=None, limit: int = 64) -> Alternatives:
    """Preimage candidates as a first-class ranked set.

    ``mechanism`` may be a single Mechanism (its ``preimage`` stream is
    consumed up to ``limit``) or a program sequence (chained through
    :func:`~twinworld.engine.abduce_inputs`). Scores: the Occam generation
    order and each candidate's distance to the observed state — turning
    examples/abduction_scaling.py's external rank-of-true-origin measurement
    into library vocabulary.
    """
    if isinstance(mechanism, (list, tuple)):
        candidates = list(abduce_inputs(tuple(mechanism), state, limit=limit, budget=budget))
    else:
        candidates = list(islice(mechanism.preimage(state, budget), limit))
    scores = tuple(
        {"order": i, "proximity": proximity(pre, state)} for i, pre in enumerate(candidates)
    )
    return Alternatives("preimages", tuple(candidates), scores)
