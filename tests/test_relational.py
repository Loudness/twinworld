"""The relational backend: identity laws, entities, conformance, capabilities."""

from twinworld import conformance_battery, get_representation
from twinworld.backends.relational import state_from_towers
from twinworld.domains.blocks import MoveBlock

REL = get_representation("relational")

SAMPLES = (
    ((1, 2), (), ()),
    ((1, 2), (3,), ()),
    ((), (5, 2), (1,)),
)


def test_parse_canon_round_trip():
    for raw in SAMPLES:
        assert REL.parse(raw).key == REL.canon(raw)
    assert REL.canon([[1, 2], [], []]) == REL.canon(((1, 2), (), ()))  # lists welcome


def test_key_is_abstraction_invariant():
    for raw in SAMPLES:
        keys = {REL.parse(raw, name).key for name in REL.abstractions}
        assert len(keys) == 1
        assert REL.parse(raw, "consts") == REL.parse(raw, "towers")


def test_entities_consts_and_towers():
    consts = REL.parse(((1, 2), (3,), ()), "consts").objects
    assert [o.oid for o in consts] == [1, 2, 3]
    two = next(o for o in consts if o.oid == 2)
    assert two.attributes == {"block": 2, "column": 0, "level": 1, "on": 1, "clear": True}
    one = next(o for o in consts if o.oid == 1)
    assert one.attributes["on"] is None and one.attributes["clear"] is False
    towers = REL.parse(((1, 2), (3,), ()), "towers").objects
    assert [(o.oid, o.size) for o in towers] == [(0, 2), (1, 1)]  # empty column: no entity
    assert towers[0].attributes["blocks"] == (1, 2)


def test_frame_is_constant_universe():
    a = REL.parse(((1, 2), (), ()))
    b = REL.parse(((2,), (1,), ()))
    assert REL.frame(a) == REL.frame(b)  # same blocks, same columns, both unbounded
    assert REL.frame(a) != REL.frame(REL.parse(((1, 2), ())))  # fewer columns
    assert REL.frame(a) != REL.frame(REL.parse(((1, 2, 9), (), ())))  # new block


def test_rebuild_closure_and_gravity():
    state = REL.parse(((1, 2), (3,), ()))
    rebuilt = REL.rebuild(state, state.objects)
    assert rebuilt is not None and rebuilt.key == state.key
    floating = [o for o in state.objects if o.attributes["block"] == 2]
    assert REL.rebuild(state, floating) is None  # level 1 with no level 0: floats


def test_relational_conformance_battery_passes():
    report = conformance_battery(
        REL,
        SAMPLES,
        mechanisms=(MoveBlock(2, 1), MoveBlock(1, 2), MoveBlock(9, 0)),
    )
    assert report.passed, str(report)
    by_name = {row.name: row for row in report.rows}
    assert by_name["L5_preimage_sound"].passed is True
    assert by_name["L6_exact_preimage_spot"].passed is True


def test_unbounded_height_move_succeeds_natively():
    tall = state_from_towers([[1, 2, 3, 5], [4], []])  # would be a FULL column at height 4
    moved = MoveBlock(4, 0).apply(tall)
    assert moved is not None
    assert moved.towers == ((1, 2, 3, 5, 4), (), ())


def test_asp_induction_gated_on_non_grid():
    import pytest

    import twinworld
    from twinworld.domains.blocks import task_from_towers

    task = task_from_towers(
        train=[([[1, 2], [], []], [[], [2], [1]])],
        test=[([[1, 2], [], []], [[], [2], [1]])],
    )
    with pytest.raises(twinworld.UnsolvedTaskError):  # graceful, not ImportError/KeyError
        twinworld.model(task, induction="asp")


def test_relational_placebo_passes_end_to_end():
    import twinworld
    from twinworld.domains.blocks import candidate_moves, task_from_towers

    task = task_from_towers(
        train=[
            ([[1, 2], [], []], [[], [2], [1]]),
            ([[1, 2], [3], []], [[], [3, 2], [1]]),
        ],
        test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
    )
    rep = twinworld.model(task, primitives=candidate_moves(task), induction="never", max_depth=2)
    report = twinworld.refute(rep)
    placebo = next(row for row in report.rows if row.name == "placebo_intervention")
    assert placebo.passed is True  # block 3 renamed; the plan passes it through
    assert "block" in placebo.detail


def test_relational_plausible_rejects_duplicate_ids():
    assert REL.plausible(state_from_towers([[1, 2], [3], []])) is True
    assert REL.plausible(state_from_towers([[1, 2], [1], []])) is False  # duplicate block id
    bounded = REL.parse(((1, 2, 3), ()), context={"height": 2})
    assert REL.plausible(bounded) is False  # exceeds the declared bound


def test_render_raw_contains_palette_columns():
    from twinworld.viz import PALETTE

    doc = REL.render_raw(((1, 2), (3,), ()), caption="towers <x>")
    assert "<table" in doc
    assert PALETTE[1] in doc and PALETTE[3] in doc
    assert "towers &lt;x&gt;" in doc  # captions are escaped


def test_probe_perturbations_cover_remove_move_add():
    state = REL.parse(((1, 2), (), ()))
    raws = list(REL.probe_perturbations(state, used=state.universe))
    assert ((1,), (), ()) in raws  # remove the clear block
    assert ((1,), (2,), ()) in raws  # move it to another column
    assert ((1, 2), (3,), ()) in raws  # add a fresh block (max id + 1 = 3)
