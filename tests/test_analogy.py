"""Experiment 2: structure mapping, rule induction, analogy-first search, MAC/FAC."""

import pytest
from conftest import T

import dowhat
from dowhat import (
    ByColour,
    Delete,
    ObjectRule,
    RecolourTo,
    Smallest,
    TranslateBy,
    content_vector,
    induce_rules,
    parse_grid,
    retrieve,
    structure_map,
)
from dowhat.analogy import pair_deltas


# ------------------------------------------------------------ structure map


def test_structure_map_uses_relations_to_beat_overlap_bait():
    """Both dots moved down 2; the naive best match (cell overlap) crosses the
    pair over. Relational agreement (above, same_colour preserved) repairs it."""
    before = parse_grid(T("30", "00", "30", "00", "00"))
    after = parse_grid(T("00", "00", "30", "00", "30"))
    mapping = {x.location: y.location for x, y in structure_map(before, after) if x and y}
    assert mapping == {(0, 0): (2, 0), (2, 0): (4, 0)}


def test_structure_map_reports_deletion():
    before = parse_grid(T("300", "000", "005"))
    after = parse_grid(T("300", "000", "000"))
    pairs = structure_map(before, after)
    gone = [x for x, y in pairs if y is None]
    assert len(gone) == 1 and gone[0].colour == 5


def test_pair_deltas_capture_move_and_recolour():
    deltas = pair_deltas(
        parse_grid(T("0330", "0000", "0050")), parse_grid(T("0440", "0000", "0050"))
    )
    by_colour = {d.obj.colour: d for d in deltas}
    assert by_colour[3].recoloured_to == 4 and by_colour[3].moved == (0, 0)
    assert by_colour[5].recoloured_to is None and by_colour[5].moved == (0, 0)


# ------------------------------------------------------------ rule induction


def test_induce_rules_proposes_selective_recolour(recolor_task):
    rules = induce_rules(recolor_task)
    assert ObjectRule(ByColour(3), RecolourTo(4)) in rules
    # the spectator is unchanged, so no all-objects recolour is proposed
    assert all(
        not (isinstance(r.selector, dowhat.All) and isinstance(r.transform, RecolourTo))
        for r in rules
    )


def test_analogy_solves_three_way_move(three_way_move_task):
    rep = dowhat.model(three_way_move_task)
    sol = rep.solution
    assert sol.strategy == "analogy"
    assert len(sol.program) == 3
    assert sol.programs_tried < 400  # 3 candidates, depth 3 — nowhere near blind scale
    assert sol.test_traces[0].outcome.key == dowhat.as_grid(three_way_move_task.test[0][1])


def test_analogy_solves_same_colour_denoise(denoise_task):
    rep = dowhat.model(denoise_task)
    sol = rep.solution
    assert sol.strategy == "analogy"
    assert sol.program == (ObjectRule(Smallest(), Delete()),)
    assert sol.test_traces[0].outcome.key == dowhat.as_grid(denoise_task.test[0][1])


def test_blind_enumeration_cannot_solve_the_analogy_fixtures(
    three_way_move_task, denoise_task
):
    for task in (three_way_move_task, denoise_task):
        with pytest.raises(dowhat.UnsolvedTaskError):
            dowhat.model(task, induction="never")


# --------------------------------------------------------------- ObjectRule


def test_object_rule_apply_and_noop_guard():
    state = parse_grid(T("300", "000", "005"))
    rule = ObjectRule(ByColour(3), RecolourTo(7))
    assert rule.apply(state).grid == T("700", "000", "005")
    assert ObjectRule(ByColour(9), RecolourTo(7)).apply(state) is None  # empty selection
    assert ObjectRule(ByColour(5), RecolourTo(5)).apply(state) is None  # no-op


def test_object_rule_translate_preimage_round_trip():
    state = parse_grid(T("300", "000", "005"))
    rule = ObjectRule(ByColour(3), TranslateBy(1, 1))
    out = rule.apply(state)
    assert list(rule.preimage(out)) == [state]
    assert rule.exact_preimage
    assert not ObjectRule(ByColour(3), RecolourTo(7)).exact_preimage


# ------------------------------------------------------------------ MAC/FAC


def test_macfac_retrieval_prefers_matching_colour_structure():
    # identical geometry, different colour signature: only the colour keys of
    # the content vector can separate the two library entries
    library = [
        ("colours-3-5", content_vector(parse_grid(T("30005", "00000", "00000")))),
        ("colours-7", content_vector(parse_grid(T("70007", "00000", "00000")))),
    ]
    probe = parse_grid(T("00300", "00000", "50000"))  # colours {3, 5} again
    ranked = retrieve(probe, library, k=2)
    assert ranked[0][0] == "colours-3-5"
    assert ranked[0][1] > ranked[1][1]