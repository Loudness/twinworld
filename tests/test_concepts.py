"""Phase A: the concept network as data — defaults reproduce the hand-coded
behaviour exactly; learning recovers planted corpus statistics; priors only
reorder candidates."""

from conftest import T

from dowhat import (
    DEFAULT_CONCEPTS,
    ByColour,
    ConceptNet,
    ObjectRule,
    RecolourTo,
    Task,
    induce_rules,
    learn_concepts,
    parse_grid,
    structure_map,
)
from dowhat.concepts import load_concepts, rule_family, save_concepts
from dowhat.mechanisms import Not
from dowhat.representation import attribute_score


def test_default_concepts_reproduce_hand_scores():
    a = parse_grid(T("330", "000", "003"))
    for x in a.objects:
        for y in a.objects:
            assert attribute_score(x, y) == attribute_score(x, y, DEFAULT_CONCEPTS)


def test_structure_map_default_concepts_identical():
    before = parse_grid(T("30", "00", "30", "00", "00"))
    after = parse_grid(T("00", "00", "30", "00", "30"))
    assert structure_map(before, after) == structure_map(before, after, DEFAULT_CONCEPTS)


def test_induce_rules_default_path_byte_identical(recolor_task, three_way_move_task):
    for task in (recolor_task, three_way_move_task):
        base = induce_rules(task)
        assert base == induce_rules(task, concepts=None)
        assert base == induce_rules(task, concepts=DEFAULT_CONCEPTS)


def _reliability_corpus() -> list[Task]:
    """Colour always preserved; shape scrambled. Objects have unique shapes
    (anchoring colour stats) and unique colours (anchoring shape stats)."""
    tasks = []
    pairs = [
        # dot stays a dot; the domino becomes a bar (shape changes, colour kept)
        (T("200", "000", "330"), T("200", "000", "333")),
        (T("020", "000", "033"), T("020", "000", "333")),
    ]
    for grid_in, grid_out in pairs:
        tasks.append(
            Task(train=((grid_in, grid_out),) * 2, test=((grid_in, grid_out),))
        )
    return tasks


def test_learn_concepts_recovers_planted_reliability():
    net = learn_concepts(_reliability_corpus())
    assert net.colour > net.shape  # colour is the reliable attribute here
    assert net.slip_colour == 0.0
    assert net.slip_shape > 0.0
    assert "learned from" in net.source


def test_learn_concepts_counts_rule_family_priors(recolor_task):
    net = learn_concepts([recolor_task])
    families = dict(net.priors)
    assert families.get("ByColour*RecolourTo", 0) > 0


def test_priors_reorder_candidates_without_filtering(recolor_task):
    base = induce_rules(recolor_task)
    negated_first = ConceptNet(
        priors=(("Not(ByColour)*RecolourTo", 0.9), ("ByColour*RecolourTo", 0.1))
    )
    reordered = induce_rules(recolor_task, concepts=negated_first)
    assert sorted(map(str, base)) == sorted(map(str, reordered))  # same set
    assert isinstance(reordered[0].selector, Not)  # the planted prior leads


def test_rule_family_names():
    assert rule_family(ObjectRule(ByColour(3), RecolourTo(4))) == "ByColour*RecolourTo"
    assert rule_family(ObjectRule(Not(ByColour(3)), RecolourTo(4))) == (
        "Not(ByColour)*RecolourTo"
    )


def test_concepts_json_round_trip(tmp_path):
    net = ConceptNet(shape=1.5, priors=(("ByColour*Delete", 0.5),), source="test")
    path = save_concepts(net, tmp_path / "net.json")
    assert load_concepts(path) == net
