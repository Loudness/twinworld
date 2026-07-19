"""Representation-backend contract: registry, protocols, conformance battery."""

from dataclasses import dataclass
from typing import Iterator

import pytest
from conftest import T

from twinworld import (
    Task,
    conformance_battery,
    get_representation,
    model,
    register,
    representation_of,
)
from twinworld.engine import solve
from twinworld.backend import Entity, State
from twinworld.mechanisms import (
    ByColour,
    Delete,
    Identity,
    ObjectRule,
    Recolor,
    RecolourTo,
    Smallest,
    Translate,
)
from twinworld.representation import parse_grid

SAMPLES = (
    T("300", "000", "005"),
    T("40004", "00000", "00000"),
    T("030", "000", "060"),
)

MECHANISMS = (
    Identity(),
    Recolor(3, 4),
    Translate(0, 1, 3),
    ObjectRule(ByColour(3), RecolourTo(4)),
    ObjectRule(Smallest(), Delete()),
)


def test_registry_resolves_grid_lazily():
    rep = get_representation("grid")
    assert rep.name == "grid"
    assert get_representation("grid") is rep
    assert representation_of("grid") is rep


def test_unknown_representation_lists_known():
    with pytest.raises(KeyError, match="unknown representation 'voxels'"):
        get_representation("voxels")


def test_obj_and_stategraph_satisfy_protocols():
    state = parse_grid(SAMPLES[0])
    assert isinstance(state, State)
    assert all(isinstance(o, Entity) for o in state.objects)
    assert state.representation == "grid"


def test_obj_attributes_and_extent():
    obj = parse_grid(T("330", "000", "000")).objects[0]
    assert obj.attributes == {
        "colour": 3,
        "shape": frozenset({(0, 0), (0, 1)}),
        "location": (0, 0),
        "size": 2,
    }
    assert obj.extent == obj.pixels


def test_task_defaults_to_grid_representation(recolor_task: Task):
    assert recolor_task.representation == "grid"
    assert representation_of(recolor_task).name == "grid"


@dataclass(frozen=True)
class _Tick:
    value: int
    abstraction: str = "unit"

    representation = "counter"
    objects: tuple = ()

    @property
    def key(self):
        return ("counter", self.value)


class _CounterRep:
    name = "counter"
    default_abstractions = ("unit",)
    transform_families: tuple = ()
    abstractions = {"unit": object()}

    def parse(self, raw, abstraction=None, context=None):
        return _Tick(int(raw))

    def canon(self, raw):
        return ("counter", int(raw))


@dataclass(frozen=True)
class _Inc:
    exact_preimage = True

    def apply(self, s: _Tick) -> _Tick:
        return _Tick(s.value + 1)

    def preimage(self, s: _Tick, budget=None) -> Iterator[_Tick]:
        yield _Tick(s.value - 1)

    def __str__(self) -> str:
        return "inc"


def test_solve_dispatches_through_registered_backend():
    register(_CounterRep())
    task = Task(train=((0, 1),), test=((5, 6),), task_id="counter", representation="counter")
    sol = solve(task, [_Inc()], max_depth=1)
    assert sol.program == (_Inc(),)
    assert sol.test_traces[0].outcome.key == ("counter", 6)


def test_model_default_abstractions_come_from_backend(recolor_task: Task):
    rep = model(recolor_task)
    assert set(rep.solutions) == set(get_representation("grid").default_abstractions)


def _legacy_induce(task):
    """The pre-refactor emission logic, inlined verbatim — pins that the
    transform-family refactor reproduces candidate lists including ORDER."""
    import random as _random

    from twinworld.analogy import pair_deltas
    from twinworld.mechanisms import All, Largest, Not, TranslateBy
    from twinworld.representation import parse_grid

    inputs = [parse_grid(i) for i, _ in task.train]
    outputs = [parse_grid(o) for _, o in task.train]
    if any(not s.objects for s in inputs):
        return []
    all_deltas = [
        pair_deltas(a, b, None, "sme", rng=_random.Random(k))
        for k, (a, b) in enumerate(zip(inputs, outputs))
    ]
    by_obj = [{d.obj.oid: d for d in ds} for ds in all_deltas]
    shared_colours = set.intersection(*({o.colour for o in s.objects} for s in inputs))
    positive = [All(), *[ByColour(c) for c in sorted(shared_colours)], Largest(), Smallest()]
    negated = [Not(Largest()), Not(Smallest()), *[Not(ByColour(c)) for c in sorted(shared_colours)]]
    rules = []
    for sel in positive + negated:
        selected = [
            [by_obj[k].get(o.oid) for o in sel.select(s.objects)] for k, s in enumerate(inputs)
        ]
        if any(not group or None in group for group in selected):
            continue
        flat = [d for group in selected for d in group]
        moves = {d.moved for d in flat}
        if len(moves) == 1 and not any(d.deleted for d in flat):
            (move,) = moves
            if move != (0, 0) and all(d.shape_stable for d in flat):
                rules.append(ObjectRule(sel, TranslateBy(*move)))
        targets = {d.recoloured_to for d in flat}
        if len(targets) == 1 and None not in targets:
            (target,) = targets
            rules.append(ObjectRule(sel, RecolourTo(target)))
        if all(d.deleted for d in flat):
            rules.append(ObjectRule(sel, Delete()))
    unique = []
    for r in rules:
        if r not in unique:
            unique.append(r)
    return unique


def test_grid_transform_families_reproduce_induce_rules(
    recolor_task, three_way_move_task, denoise_task
):
    from twinworld import induce_rules

    for task in (recolor_task, three_way_move_task, denoise_task):
        assert induce_rules(task) == _legacy_induce(task)


def test_empty_transform_families_short_circuits_induction():
    from twinworld import induce_rules

    register(_CounterRep())
    task = Task(train=((0, 1),), test=((5, 6),), task_id="counter", representation="counter")
    assert induce_rules(task) == []


@pytest.mark.parametrize(
    ("mechanism", "expected"),
    [
        (Identity(), frozenset()),
        (Recolor(3, 4), frozenset({3, 4})),
        (Translate(0, 1), None),
        (Translate(0, 1, 5), frozenset({5})),
        (ObjectRule(ByColour(3), RecolourTo(4)), frozenset({3, 4})),
        (ObjectRule(ByColour(2), Delete()), frozenset({2})),
        (ObjectRule(Smallest(), Delete()), None),
    ],
)
def test_touched_matches_the_historical_reference_table(mechanism, expected):
    assert mechanism.touched("colour") == expected
    assert mechanism.touched("shape") == (frozenset() if isinstance(mechanism, Identity) else None)


def test_moveblock_touched_block_attr():
    from twinworld.domains.blocks import MoveBlock

    assert MoveBlock(1, 2).touched("block") == frozenset({1})
    assert MoveBlock(1, 2).touched("colour") is None  # grids know nothing of blocks


def test_referenced_values_accumulates_and_bails():
    from twinworld.domains.blocks import MoveBlock
    from twinworld.mechanisms import Not
    from twinworld.refute import referenced_values
    from twinworld.representation import MAX_COLOURS

    assert referenced_values([Recolor(3, 4), ObjectRule(ByColour(5), Delete())]) == frozenset(
        {3, 4, 5}
    )
    assert referenced_values([Translate(0, 1)]) is None
    assert referenced_values([MoveBlock(1, 2)]) is None
    assert referenced_values([ObjectRule(Not(ByColour(3)), RecolourTo(4))]) == frozenset(
        set(range(MAX_COLOURS)) - {3} | {4}
    )


def _legacy_probes(task):
    """The pre-extraction probe generation, inlined with its own helpers —
    pins that the capability move preserved content and ORDER exactly."""
    from twinworld.representation import as_grid, parse_grid

    def paint(grid, obj, colour):
        rows = [list(r) for r in grid]
        for r, c in obj.cells:
            rows[r][c] = colour
        return as_grid(rows)

    def nudge(grid, obj, dr, dc, background):
        h, w = len(grid), len(grid[0])
        targets = {(r + dr, c + dc) for r, c in obj.cells}
        if not all(0 <= r < h and 0 <= c < w for r, c in targets):
            return None
        if any((r, c) not in obj.cells and grid[r][c] != background for r, c in targets):
            return None
        rows = [list(r) for r in grid]
        for r, c in obj.cells:
            rows[r][c] = background
        for r, c, colour in obj.pixels:
            rows[r + dr][c + dc] = colour
        return as_grid(rows)

    out, seen = [], set()

    def add(grid):
        if grid is not None and grid not in seen:
            seen.add(grid)
            out.append(grid)

    used = task.colours()
    fresh = next((c for c in range(9, -1, -1) if c not in used), 9)
    for grid_in, _ in task.train:
        state = parse_grid(grid_in)
        base = state.grid
        add(base)
        for o in state.objects[:6]:
            add(paint(base, o, state.background))
            add(paint(base, o, fresh))
            for dr, dc in ((0, 1), (1, 0)):
                add(nudge(base, o, dr, dc, state.background))
    return out


def test_grid_probes_unchanged_after_extraction(recolor_task, denoise_task):
    from twinworld.discriminate import probes

    for task in (recolor_task, denoise_task):
        assert probes(task) == _legacy_probes(task)


def test_grid_addition_catalogue_order_and_phrase(recolor_task):
    from twinworld.representation import parse_grid

    grid_rep = get_representation("grid")
    state = parse_grid(recolor_task.train[0][0])
    additions = list(grid_rep.addition_catalogue(state, 2, True, [7, 9]))
    assert additions, "the fixture state must leave free separated cells"
    sizes = [a.size for a in additions]
    assert sizes == sorted(sizes), "catalogue must stream footprint sizes in order"
    first = additions[0]
    assert first.size == 1
    assert first.phrase == f"a colour-7 object occupied {[first.group]}"
    assert additions[1].group == first.group  # same anchor, next colour
    assert additions[1].phrase.startswith("a colour-9 object")
    changed = [
        (r, c)
        for r, row in enumerate(first.raw)
        for c, v in enumerate(row)
        if state.grid[r][c] != v
    ]
    assert changed == [first.group]


def test_capabilities_of_grid():
    from twinworld import capabilities

    assert capabilities(get_representation("grid")) == frozenset(
        {"probes", "pertinent_negative", "placebo", "distance", "plausibility"}
    )


def test_attribute_score_generalized_matches_inlined_legacy():
    from twinworld import DEFAULT_CONCEPTS, ConceptNet
    from twinworld.representation import attribute_score

    def legacy(x, y, w_shape, w_colour, w_location, w_iou):
        s = 0.0
        s += w_shape if x.shape == y.shape else 0
        s += w_colour if x.colour == y.colour else 0
        s += w_location if x.location == y.location else 0
        s += w_iou * len(x.cells & y.cells) / len(x.cells | y.cells)
        return s

    state_a = parse_grid(T("330", "000", "005"))
    state_b = parse_grid(T("033", "000", "500"))
    learned = ConceptNet(shape=0.27, colour=3.1, location=2.82, iou=1.5)
    for x in state_a.objects:
        for y in state_b.objects:
            assert attribute_score(x, y) == legacy(x, y, 4.0, 2.0, 1.0, 3.0)
            assert attribute_score(x, y, DEFAULT_CONCEPTS) == legacy(x, y, 4.0, 2.0, 1.0, 3.0)
            assert attribute_score(x, y, learned) == legacy(x, y, 0.27, 3.1, 2.82, 1.5)


def test_conceptnet_weight_slip_lookups():
    from twinworld import ConceptNet

    net = ConceptNet(
        shape=4.0, slip_move=0.25,
        attributes=(("letter", 2.0),), slips=(("letter", 0.1),),
    )
    assert net.weight("shape") == 4.0
    assert net.weight("letter") == 2.0
    assert net.weight("size") == 0.0
    assert net.slip("location") == 0.25  # grid location slip = "move"
    assert net.slip("letter") == 0.1
    assert net.slip("unknown") == 0.0


def test_learned_concepts_json_still_loads():
    from pathlib import Path

    from twinworld.concepts import load_concepts

    path = Path(__file__).resolve().parent.parent / "docs" / "learned-concepts.json"
    net = load_concepts(path)
    assert net.attributes == () and net.slips == ()  # pre-P6 files stay loadable
    assert "learned from" in net.source


def test_concepts_round_trip_with_attributes(tmp_path):
    from twinworld import ConceptNet
    from twinworld.concepts import load_concepts, save_concepts

    net = ConceptNet(attributes=(("letter", 2.0), ("position", 4.0)), slips=(("letter", 0.3),))
    assert load_concepts(save_concepts(net, tmp_path / "net.json")) == net


def test_relations_dispatch_grid_identical():
    from twinworld.analogy import relations

    state = parse_grid(T("30", "05"))
    assert relations(state) == {("left_of", 0, 1), ("above", 0, 1), ("same_shape", 0, 1)}


def test_byattr_select_str_and_rule_family():
    from twinworld import ByAttr
    from twinworld.concepts import rule_family
    from twinworld.mechanisms import Not, RecolourTo

    state = parse_grid(T("330", "000", "005"))
    picked = ByAttr("colour", 3).select(state.objects)
    assert [o.colour for o in picked] == [3]
    assert str(ByAttr("letter", "a")) == "letter-a objects"
    assert rule_family(ObjectRule(ByAttr("letter", "a"), RecolourTo(4))) == (
        "ByAttr[letter]*RecolourTo"
    )
    assert rule_family(ObjectRule(Not(ByAttr("letter", "a")), Delete())) == (
        "Not(ByAttr[letter])*Delete"
    )
    assert rule_family(ObjectRule(Not(ByColour(3)), Delete())) == "Not(ByColour)*Delete"


def test_structure_map_runs_on_relational_entities():
    from twinworld import structure_map
    from twinworld.backends.relational import state_from_towers

    before = state_from_towers([[1, 2], [3], []])
    after = state_from_towers([[1], [3, 2], []])
    pairs = [(x, y) for x, y in structure_map(before, after) if x is not None and y is not None]
    assert pairs, "relational entities must be mappable without grid attributes"
    mapped = {x.oid: y.oid for x, y in pairs}
    assert mapped.get(3) == 3  # the unmoved block maps to itself


def test_grid_conformance_battery_passes():
    report = conformance_battery(get_representation("grid"), SAMPLES, MECHANISMS)
    assert report.passed, str(report)
    by_name = {row.name: row for row in report.rows}
    for law in (
        "L1_parse_canon",
        "L2_key_abstraction_invariance",
        "L3_eq_hash_via_key",
        "L4_apply_canonical",
        "L5_preimage_sound",
        "L6_exact_preimage_spot",
        "L7_rebuild_closure",
    ):
        assert by_name[law].passed is True, f"{law}: {by_name[law].detail}"
