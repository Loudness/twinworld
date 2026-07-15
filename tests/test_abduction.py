"""Phase 9: abduction through deletion — the documented wall becomes a
bounded hypothesis search, verified exactly by re-application."""

import twinworld
from conftest import T

from twinworld import ByColour, Delete, ObjectRule, RecolourTo, Smallest, parse_grid
from twinworld.engine import abduce_inputs


def test_delete_preimage_recovers_denoise_input(denoise_task):
    rep = twinworld.model(denoise_task)
    (rule,) = rep.solution.program
    trace = rep.solution.train_traces[0]
    preimages = list(rule.preimage(trace.outcome))
    assert preimages  # the wall is gone
    # the true input had TWO specks: recovered via a two-object hypothesis
    assert trace.states[0] in preimages


def test_bycolour_delete_preimages_pin_the_colour():
    rule = ObjectRule(ByColour(5), Delete())
    t = parse_grid(T("330", "000", "000"))
    preimages = list(rule.preimage(t))
    assert preimages
    for pre in preimages:
        assert any(o.colour == 5 for o in pre.objects)  # something 5 was deleted
        assert rule.apply(pre) == t


def test_smallest_delete_respects_size_ordering():
    """Survivors are dominoes (size 2): a hypothesised deleted object of size
    >= 2 would tie or beat them and delete them too — verification must
    reject everything but 1-cell hypotheses."""
    rule = ObjectRule(Smallest(), Delete())
    t = parse_grid(T("22000", "00033", "00000", "00000"))
    preimages = list(rule.preimage(t))
    assert preimages
    surviving_cells = {o.cells for o in t.objects}
    for pre in preimages:
        extras = [o for o in pre.objects if o.cells not in surviving_cells]
        assert extras and all(o.size == 1 for o in extras)


def test_backward_chain_through_recolour_and_delete():
    program = (
        ObjectRule(ByColour(3), RecolourTo(4)),
        ObjectRule(Smallest(), Delete()),
    )
    s0 = parse_grid(T("33000", "00000", "00030"))  # domino + dot, both colour 3
    mid = program[0].apply(s0)
    final = program[1].apply(mid)
    assert final is not None
    # ~30 delete-hypotheses x ~3 recolour-preimages each: the budget must
    # cover the frontier's breadth for the true origin to surface
    inputs = abduce_inputs(program, final, limit=128)
    assert s0 in inputs  # time travel backwards, through a deletion


def test_abduce_inputs_is_bounded_and_deterministic(denoise_task):
    rep = twinworld.model(denoise_task)
    outcome = rep.solution.train_traces[0].outcome
    inputs = abduce_inputs(rep.solution.program, outcome, limit=5)
    assert 0 < len(inputs) <= 5
    assert inputs == abduce_inputs(rep.solution.program, outcome, limit=5)
