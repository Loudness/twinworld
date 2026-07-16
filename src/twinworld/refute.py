"""Refutation battery (the DoWhy stance: no global validator exists, so attack
the explanation from multiple angles and report what survived).

Slice scope: the placebo intervention — perturbing a property the induced
program never references must pass through the pipeline inertly. If it does
not, the model secretly depends on something the explanation calls irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

from .backend import representation_of
from .engine import Solution, backtrack
from .mechanisms import Mechanism
from .representation import MAX_COLOURS


@dataclass(frozen=True)
class RefutationRow:
    name: str
    passed: bool | None  # None: not applicable to this task
    detail: str


@dataclass(frozen=True)
class RefutationReport:
    rows: tuple[RefutationRow, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.rows)

    def __str__(self) -> str:
        marks = {True: "PASS", False: "FAIL", None: "SKIP"}
        return "\n".join(f"[{marks[r.passed]}] {r.name}: {r.detail}" for r in self.rows)


def referenced_values(program: Sequence[Mechanism], attr: str = "colour") -> frozenset | None:
    """Attribute values the program actually mentions, via each mechanism's
    ``touched`` introspection; None when any mechanism may touch anything."""
    refs: set = set()
    for mech in program:
        introspect = getattr(mech, "touched", None)
        vals = introspect(attr) if introspect is not None else None
        if vals is None:
            return None
        refs |= vals
    return frozenset(refs)


def referenced_colours(solution: Solution) -> set[int]:
    """Colours the induced program actually mentions."""
    refs = referenced_values(solution.program)
    return set(range(MAX_COLOURS)) if refs is None else set(refs)


def placebo_intervention(solution: Solution) -> RefutationRow:
    """Perturb one program-irrelevant entity in a train input and rerun.

    The perturbed attribute is the backend's ``placebo_attr`` (grid: recolour
    a spectator object; relational: rename a spectator block); the expectation
    is that the perturbation passes through the program unchanged. Backends
    without a ``placebo_edit`` capability skip the check.
    """
    backend = representation_of(solution.task)
    edit_fn = getattr(backend, "placebo_edit", None)
    if edit_fn is None:
        return RefutationRow(
            "placebo_intervention",
            None,
            f"the {backend.name} representation defines no placebo edit",
        )
    attr = backend.placebo_attr
    refs = referenced_values(solution.program, attr=attr)
    for trace in solution.train_traces:
        start = trace.states[0]
        spectators = (
            []
            if refs is None  # the program may touch anything: nothing is irrelevant
            else [o for o in start.objects if o.attributes.get(attr) not in refs]
        )
        if not spectators:
            continue
        target = spectators[0]
        edit = edit_fn(start, target, frozenset(refs))
        if edit is None:
            continue
        edited, placebo_value, expect = edit
        old = target.attributes.get(attr)
        cf = backtrack(solution, trace, edited)
        if not cf.applicable:
            return RefutationRow(
                "placebo_intervention",
                False,
                f"program became inapplicable after perturbing an irrelevant "
                f"entity ({attr} {old!r}, entity {target.oid})",
            )
        passed = cf.counterfactual.outcome.key == expect(trace.outcome)
        detail = (
            f"perturbed {attr} {old!r} -> {placebo_value!r} on spectator entity "
            f"{target.oid}; outcome {'unchanged elsewhere' if passed else 'CHANGED'}"
        )
        return RefutationRow("placebo_intervention", passed, detail)
    return RefutationRow(
        "placebo_intervention",
        None,
        "no program-irrelevant object exists in any train input",
    )


def asp_crosscheck(solution: Solution) -> RefutationRow:
    """Independent logic-based validation: selector semantics re-derived by
    clingo under negation-as-failure must match the Python implementation on
    every reached state (thesis Experiment 4's 'extra path validation')."""
    from .asp import crosscheck_selectors

    passed, detail = crosscheck_selectors(solution)
    return RefutationRow("asp_selector_crosscheck", passed, detail)


def refutation_battery(solution: Solution) -> RefutationReport:
    return RefutationReport((placebo_intervention(solution), asp_crosscheck(solution)))
