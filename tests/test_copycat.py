"""Phase A: the Copycat correspondence backend — deterministic under a seed,
SME-equivalent objective at zero slippage, and a usable mapper inside model()."""

import random

from conftest import T

import dowhat
from dowhat import induce_rules, parse_grid, structure_map
from dowhat.copycat import copycat_map


def _pairs_as_locations(pairs):
    return {x.location: y.location for x, y in pairs if x and y}


def test_copycat_deterministic_given_seed():
    a = parse_grid(T("30", "00", "30", "00", "00"))
    b = parse_grid(T("00", "00", "30", "00", "30"))
    one = copycat_map(a, b, rng=random.Random(7))
    two = copycat_map(a, b, rng=random.Random(7))
    assert one == two


def test_copycat_matches_sme_on_overlap_bait():
    """At zero slippage the objective is SME's; annealing must find the same
    relation-repaired mapping on the crossover bait."""
    a = parse_grid(T("30", "00", "30", "00", "00"))
    b = parse_grid(T("00", "00", "30", "00", "30"))
    assert _pairs_as_locations(copycat_map(a, b)) == _pairs_as_locations(
        structure_map(a, b)
    )


def test_mapper_both_is_candidate_superset(recolor_task, three_way_move_task):
    for task in (recolor_task, three_way_move_task):
        sme = induce_rules(task)
        both = induce_rules(task, mapper="both")
        assert set(map(str, sme)) <= set(map(str, both))


def test_model_with_copycat_mapper_solves_deterministically(three_way_move_task):
    rep1 = dowhat.model(three_way_move_task, mapper="copycat")
    rep2 = dowhat.model(three_way_move_task, mapper="copycat")
    assert rep1.solution.program == rep2.solution.program
    assert rep1.solution.programs_tried == rep2.solution.programs_tried
    assert rep1.mapper == "copycat"
    out = rep1.solution.test_traces[0].outcome.key
    assert out == three_way_move_task.test[0][1]
