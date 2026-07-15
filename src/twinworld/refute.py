"""Refutation battery (the DoWhy stance: no global validator exists, so attack
the explanation from multiple angles and report what survived).

Slice scope: the placebo intervention — perturbing a property the induced
program never references must pass through the pipeline inertly. If it does
not, the model secretly depends on something the explanation calls irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Solution, backtrack
from .mechanisms import ByColour, Not, ObjectRule, Recolor, RecolourTo, Translate
from .representation import MAX_COLOURS, Obj, StateGraph, as_grid


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


def referenced_colours(solution: Solution) -> set[int]:
    """Colours the induced program actually mentions."""
    refs: set[int] = set()
    for mech in solution.program:
        if isinstance(mech, Recolor):
            refs |= {mech.src, mech.dst}
        elif isinstance(mech, Translate):
            if mech.colour is None:
                return set(range(MAX_COLOURS))  # moves everything: no colour is irrelevant
            refs.add(mech.colour)
        elif isinstance(mech, ObjectRule):
            sel = mech.selector
            if isinstance(sel, ByColour):
                refs.add(sel.colour)
            elif isinstance(sel, Not) and isinstance(sel.inner, ByColour):
                refs |= set(range(MAX_COLOURS)) - {sel.inner.colour}  # touches all BUT c
            else:
                return set(range(MAX_COLOURS))  # All/Largest/Smallest/Not may touch anything
            if isinstance(mech.transform, RecolourTo):
                refs.add(mech.transform.colour)
        else:
            return set(range(MAX_COLOURS))  # unknown mechanism: assume nothing is irrelevant
    return refs


def placebo_intervention(solution: Solution) -> RefutationRow:
    """Recolour one program-irrelevant object in a train input and rerun.

    Expectation: the output equals the factual output with exactly that
    object's cells recoloured — the perturbation passes through unchanged.
    """
    refs = referenced_colours(solution)
    for trace, (grid_in, _) in zip(solution.train_traces, solution.task.train):
        start = trace.states[0]
        spectators = [o for o in start.objects if o.colour not in refs]
        if not spectators:
            continue
        target = spectators[0]
        free_colours = [
            c for c in range(MAX_COLOURS)
            if c not in refs and c != target.colour and c != start.background
        ]
        if not free_colours:
            continue
        placebo_colour = free_colours[0]
        edited = _recolour_cells(start, target, placebo_colour)
        cf = backtrack(solution, trace, edited)
        if not cf.applicable:
            return RefutationRow(
                "placebo_intervention",
                False,
                f"program became inapplicable after perturbing irrelevant "
                f"colour-{target.colour} object at {target.location}",
            )
        expected = _recolour_cells(trace.outcome, _find(trace.outcome, target), placebo_colour)
        actual = cf.counterfactual.outcome.key
        passed = actual == as_grid(expected)
        detail = (
            f"perturbed colour-{target.colour} object at {target.location} to "
            f"colour {placebo_colour}; outcome {'unchanged elsewhere' if passed else 'CHANGED'}"
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


def _recolour_cells(state: StateGraph, obj: Obj | None, colour: int):
    rows = [list(row) for row in state.grid]
    if obj is not None:
        for r, c in obj.cells:
            rows[r][c] = colour
    return as_grid(rows)


def _find(state: StateGraph, obj: Obj) -> Obj | None:
    """Locate the same (untouched) object in another state by cells+colour."""
    for candidate in state.objects:
        if candidate.cells == obj.cells and candidate.colour == obj.colour:
            return candidate
    return None
