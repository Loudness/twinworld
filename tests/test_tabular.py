"""The tabular backend: identity laws, decision-list induction as program
search, recourse via backtracking, and Rashomon-set diagnosis."""

import pytest

import twinworld
from twinworld import IdentificationError, PertinentNegative, conformance_battery
from twinworld.backends.tabular import LABEL, FeatureState, SetLabelIf, rows_task
from twinworld.discriminate import diagnose
from twinworld.engine import solve_all

TAB = twinworld.get_representation("tabular")

SAMPLES = (
    {"income": 40, "debt": 10, LABEL: None},
    {"income": 55, "debt": 0, LABEL: "approve"},
    {"employed": "yes", "income": 20, LABEL: None},
)

MECHANISMS = (
    SetLabelIf("income", ">=", 50, "approve"),
    SetLabelIf("employed", "==", "yes", "approve"),
    SetLabelIf("debt", ">=", 5, "deny"),
)


def test_parse_sorts_and_keys():
    state = TAB.parse({"b": 1, "a": 2})
    assert state.values == (("a", 2), ("b", 1))
    assert TAB.canon({"b": 1, "a": 2}) == TAB.canon((("a", 2), ("b", 1)))


def test_entity_oids_are_stable_schema_indices():
    state = TAB.parse(SAMPLES[0])
    assert [(o.oid, o.attributes["name"]) for o in state.objects] == [
        (0, "debt"),
        (1, "income"),
        (2, LABEL),
    ]


def test_frame_is_feature_names():
    a = TAB.parse({"income": 40, LABEL: None})
    b = TAB.parse({"income": 99, LABEL: None})
    assert TAB.frame(a) == TAB.frame(b)
    assert TAB.frame(a) != TAB.frame(TAB.parse({"income": 40, "debt": 1, LABEL: None}))


def test_setlabelif_first_match_wins_and_passes_through():
    rule = SetLabelIf("income", ">=", 50, "approve")
    hit = TAB.parse({"income": 55, LABEL: None})
    assert rule.apply(hit).row()[LABEL] == "approve"
    miss = TAB.parse({"income": 20, LABEL: None})
    assert rule.apply(miss) == miss  # falls through, unchanged
    decided = TAB.parse({"income": 55, LABEL: "deny"})
    assert rule.apply(decided) == decided  # already decided: falls through


def test_setlabelif_exact_preimage_fired_and_passthrough():
    rule = SetLabelIf("income", ">=", 50, "approve")
    labelled = TAB.parse({"income": 55, LABEL: "approve"})
    pres = list(rule.preimage(labelled))
    assert TAB.parse({"income": 55, LABEL: None}) in pres  # the fired predecessor
    assert labelled in pres  # the pass-through predecessor (already decided)
    assert rule.exact_preimage


def test_conformance_battery_tabular_passes():
    report = conformance_battery(TAB, tuple(SAMPLES), mechanisms=MECHANISMS)
    assert report.passed, str(report)


def test_solve_induces_consistent_rule_list():
    task = rows_task(
        train=[
            ({"income": 60, "debt": 2}, "approve"),
            ({"income": 30, "debt": 2}, "deny"),
            ({"income": 55, "debt": 9}, "approve"),
        ],
        test=[({"income": 70, "debt": 1}, "approve")],
    )
    rep = twinworld.model(task, max_depth=2)
    assert rep.solution.test_traces[0].outcome.row()[LABEL] == "approve"
    for raw_in, raw_out in task.train:  # the induced list reproduces every row
        assert rep.solution.cache.run(TAB.parse(raw_in), rep.solution.program).outcome.key == (
            TAB.canon(raw_out)
        )


def test_backtracking_recourse_flips_label_frame_checked():
    task = rows_task(
        train=[
            ({"income": 60}, "approve"),
            ({"income": 30}, "deny"),
            ({"income": 55}, "approve"),
        ],
        test=[({"income": 20}, "deny")],
    )
    rep = twinworld.model(task, max_depth=2)
    edited = {"income": 65, LABEL: None}
    cfs = twinworld.compute(twinworld.identify(rep, twinworld.Backtracking(edited)))
    outcome = cfs.items[0].counterfactual.counterfactual.outcome
    assert outcome.row()[LABEL] == "approve"  # the recourse what-if flips the label
    with pytest.raises(IdentificationError, match="dimensions"):
        twinworld.identify(rep, twinworld.Backtracking({"income": 65, "new": 1, LABEL: None}))


def test_rashomon_solve_all_diagnose_splits_by_probe():
    # income >= 50 and savings >= 10 coincide on every demonstration
    task = rows_task(
        train=[
            ({"income": 60, "savings": 12}, "approve"),
            ({"income": 30, "savings": 3}, "deny"),
        ],
        test=[({"income": 55, "savings": 11}, "approve")],
    )
    fits = solve_all(task, TAB.candidate_primitives(task), max_depth=2)
    report = diagnose(task, fits)
    assert report.underdetermined  # the Rashomon set is real...
    assert report.probe is not None  # ...and a probe exhibits where readings part ways


def test_distance_gower_style_bounds():
    a = TAB.parse({"income": 40, "job": "clerk", LABEL: None})
    b = TAB.parse({"income": 60, "job": "cook", LABEL: None})
    d = TAB.distance(a, b)
    assert 0.0 < d <= 2.0  # one numeric (normalized ≤1) + one categorical (1)
    assert TAB.distance(a, a) == 0.0


def test_identify_pertinent_negative_rejected_without_capability():
    task = rows_task(train=[({"income": 60}, "approve")], test=[({"income": 70}, "approve")])
    rep = twinworld.model(task, max_depth=1)
    with pytest.raises(IdentificationError, match="addition catalogue"):
        twinworld.identify(rep, PertinentNegative())


def test_plausible_structural_only():
    assert TAB.plausible(TAB.parse({"a": 1, LABEL: None})) is True
    assert TAB.plausible(FeatureState((("b", 1), ("a", 2)))) is False  # unsorted schema
