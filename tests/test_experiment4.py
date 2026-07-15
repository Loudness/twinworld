"""Experiment 4: negation — in the rule language, as pertinent negatives,
and as an ASP negation-as-failure cross-check."""

import pytest
from conftest import T

import twinworld
from twinworld import (
    ByColour,
    IdentificationError,
    Largest,
    Not,
    ObjectRule,
    PertinentNegative,
    RecolourTo,
    Task,
    parse_grid,
)
from twinworld.mechanisms import All, Smallest


# ---------------------------------------------------------- the Not selector


def test_not_selector_semantics():
    state = parse_grid(T("22200", "00000", "30040"))
    complement = Not(Largest()).select(state.objects)
    assert {o.colour for o in complement} == {3, 4}
    assert all(o.colour != 3 for o in Not(ByColour(3)).select(state.objects))
    assert Not(All()).select(state.objects) == ()
    assert str(Not(Largest())) == "objects other than largest object(s)"


@pytest.fixture
def except_largest_task() -> Task:
    """Recolour everything EXCEPT the largest object to 5. The largest object's
    colour varies across pairs (2, then 6), so no positive ByColour selector
    expresses the rule; the non-largest objects have differing sizes in pair 1,
    so Smallest cannot express it either. Negation is necessary."""
    return Task(
        train=(
            (T("22200", "00000", "30044"), T("22200", "00000", "50055")),
            (T("00000", "06660", "20070"), T("00000", "06660", "50050")),
        ),
        test=((T("33300", "00000", "20004"), T("33300", "00000", "50005")),),
        task_id="synthetic-except-largest",
    )


def test_negation_extends_the_solvable_language(except_largest_task):
    rep = twinworld.model(except_largest_task)
    assert rep.solution.program == (ObjectRule(Not(Largest()), RecolourTo(5)),)
    assert rep.solution.strategy == "analogy"
    expected = twinworld.as_grid(except_largest_task.test[0][1])
    assert rep.solution.test_traces[0].outcome.key == expected
    # without negation (blind vocabulary), the task is not expressible
    with pytest.raises(twinworld.UnsolvedTaskError):
        twinworld.model(except_largest_task, induction="never")


# ------------------------------------------------------- pertinent negatives


def test_pertinent_negatives_discriminate_hypotheses(small_ambiguous_task):
    """Negation-as-discriminator: the Largest-hypothesis depends on the ABSENCE
    of any bigger object; the ByColour-hypothesis does not."""
    by_colour = twinworld.model(
        small_ambiguous_task,
        primitives=[ObjectRule(ByColour(2), RecolourTo(5))],
        induction="never",
        max_depth=1,
    )
    robust = twinworld.compute(twinworld.identify(by_colour, PertinentNegative(max_cells=3)))
    (item,) = robust.items
    assert "bounded certificate" in item.narrative
    assert not item.metrics.applicable

    largest = twinworld.model(
        small_ambiguous_task,
        primitives=[ObjectRule(Largest(), RecolourTo(5))],
        induction="never",
        max_depth=1,
    )
    fragile = twinworld.compute(twinworld.identify(largest, PertinentNegative(max_cells=3)))
    assert fragile.items
    for item in fragile.items:
        # a strictly larger added object steals the selection; 1- and 2-cell
        # additions tie at best and provably cannot, so 3 is certified minimal
        assert "3-cell addition" in item.narrative
        assert "load-bearing" in item.narrative


def test_pertinent_negative_finds_applicability_boundary(move_recolor_task):
    """A colour-2 cell added near the right edge would be swept out of the grid
    by the translate step — the program's applicability depends on its absence."""
    rep = twinworld.model(move_recolor_task)
    pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(max_cells=1)))
    assert pn.items
    assert any("no longer apply" in item.narrative for item in pn.items)


def test_pertinent_negative_identify_errors(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="max_cells"):
        twinworld.identify(rep, PertinentNegative(max_cells=0))
    with pytest.raises(IdentificationError, match="does not exist"):
        twinworld.identify(rep, PertinentNegative(on="test[7]"))


def test_negation_increases_underdetermination(recolor_task):
    """The cost of negation: 'recolour colour 3' and 'recolour everything that
    is not the colour-5 spectator' both fit the demonstrations — an ambiguity
    the positive-only language did not have, exposed by a recolour probe."""
    from twinworld.discriminate import diagnose
    from twinworld.engine import solve_all

    fits = solve_all(recolor_task, twinworld.induce_rules(recolor_task), max_depth=1)
    assert (ObjectRule(ByColour(3), RecolourTo(4)),) in fits
    assert (ObjectRule(Not(ByColour(5)), RecolourTo(4)),) in fits
    report = diagnose(recolor_task, fits)
    assert report.underdetermined


# ------------------------------------------- abduction through RecolourTo


def test_recolour_preimage_bycolour_pinned():
    rule = ObjectRule(ByColour(3), RecolourTo(4))
    s = parse_grid(T("300", "000", "005"))
    t = rule.apply(s)
    assert s in list(rule.preimage(t))
    assert not rule.exact_preimage


def test_recolour_preimage_enumerates_candidate_colours():
    rule = ObjectRule(All(), RecolourTo(4))
    s = parse_grid(T("330", "000", "000"))
    t = rule.apply(s)
    pres = list(rule.preimage(t))
    assert s in pres  # the true origin is among the verified candidates
    for pre in pres:
        assert rule.apply(pre) == t  # every candidate is a genuine preimage


def test_delete_preimage_is_catalogue_bounded():
    rule = ObjectRule(Smallest(), twinworld.Delete())
    s = parse_grid(T("330", "000", "005"))
    t = rule.apply(s)
    preimages = list(rule.preimage(t))
    assert preimages  # the documented wall is gone (M9: bounded hypothesis space)
    assert all(rule.apply(pre) == t for pre in preimages)
    # ... but bounded honestly: the true origin's colour (5) is outside the
    # surviving palette {3}, so it lies beyond this catalogue
    assert s not in preimages


# --------------------------------------------------- ASP cross-check (NAF)


def test_asp_selectors_agree_with_python():
    pytest.importorskip("clingo")
    from twinworld.asp import asp_select

    state = parse_grid(T("22200", "00000", "30040"))
    for selector in (
        All(),
        ByColour(3),
        Largest(),
        Smallest(),
        Not(Largest()),
        Not(ByColour(3)),
        Not(Smallest()),
    ):
        expected = frozenset(o.oid for o in selector.select(state.objects))
        assert asp_select(selector, state) == expected, str(selector)


def test_asp_refuter_row_passes_on_solved_task(recolor_task):
    pytest.importorskip("clingo")
    rep = twinworld.model(recolor_task, max_depth=1)
    report = twinworld.refute(rep)
    row = next(r for r in report.rows if r.name == "asp_selector_crosscheck")
    assert row.passed is True
    assert "agrees" in row.detail