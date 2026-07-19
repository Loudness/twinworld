"""P5 evaluation layer: plausible/reachable certificates, exact Pareto front,
the unified Alternatives container, and the diversity wiring."""

import random

from conftest import T

import twinworld
from twinworld import (
    Alternatives,
    Contrastive,
    MetricVector,
    PertinentNegative,
    Task,
    dominates,
    pareto_front,
    rank_preimages,
)
from twinworld.alternatives import GENERIC_POLICIES, as_policy, class_alternatives
from twinworld.mechanisms import ByColour, Delete, ObjectRule, Recolor
from twinworld.representation import parse_grid
from twinworld.select import POLICIES


def _vector(sparsity=1, proximity=1.0, plausible=None, applicable=True, validity=True):
    return MetricVector(validity, sparsity, proximity, 0, applicable, plausible, None)


# ------------------------------------------------------- certificates


def test_evaluate_gains_plausible_and_reachable(recolor_task):
    rep = twinworld.model(recolor_task)
    cfs = twinworld.compute(
        twinworld.identify(rep, twinworld.Interventional(step=0, alternative=Recolor(3, 9)))
    )
    m = cfs.items[0].metrics
    assert m.plausible is True  # the grid backend certifies every grid world
    assert m.reachable in (True, False)  # a certificate, not None, once searched is set
    assert "plausible" in m.as_dict() and "reachable" in m.as_dict()


def test_reachable_uses_search_snapshot_not_live_dag(recolor_task):
    """The regression the design review caught: live-DAG membership is vacuous
    (every computed CF lands in the cache), so reachable must come from the
    search-time snapshot and be False for outcomes search never visited."""
    rep = twinworld.model(recolor_task)  # analogy path: tiny searched frontier
    cfs = twinworld.compute(
        twinworld.identify(rep, twinworld.Interventional(step=0, alternative=Recolor(3, 9)))
    )
    item = cfs.items[0]
    assert item.counterfactual.applicable
    outcome_key = item.counterfactual.counterfactual.outcome.key
    assert outcome_key in rep.solution.dag  # the live DAG has it (it was computed)
    assert item.metrics.reachable is False  # ... but search never went there


def test_reachable_none_when_inapplicable(recolor_task):
    rep = twinworld.model(recolor_task)
    cfs = twinworld.compute(
        twinworld.identify(rep, twinworld.Interventional(step=0, alternative=Recolor(9, 1)))
    )
    m = cfs.items[0].metrics
    assert not m.applicable
    assert m.reachable is None and m.plausible is None


# ------------------------------------------------------- dominance / Pareto


def test_dominates_validity_hard_filter_and_none_plausible_neutral():
    ok = _vector(sparsity=2, proximity=5.0)
    broken = _vector(sparsity=1, proximity=0.0, validity=False)
    assert dominates(ok, broken)  # hard filter beats better numbers
    assert not dominates(broken, ok)
    better = _vector(sparsity=1, proximity=1.0)
    worse = _vector(sparsity=2, proximity=1.0)
    assert dominates(better, worse) and not dominates(worse, better)
    plausible = _vector(plausible=True)
    unknown = _vector(plausible=None)
    assert not dominates(plausible, unknown)  # None is neutral, not a loss
    assert not dominates(unknown, plausible)
    implausible = _vector(plausible=False)
    assert dominates(plausible, implausible)


def test_pareto_front_exact_and_order_preserving():
    a = _vector(sparsity=1, proximity=3.0)
    b = _vector(sparsity=2, proximity=1.0)  # trades sparsity for proximity: on the front
    c = _vector(sparsity=2, proximity=3.0)  # dominated by both
    front = pareto_front([a, b, c])
    assert front == (a, b)


# ------------------------------------------------------- Alternatives container


def test_alternatives_accessor_edit_sets_scores(recolor_task):
    rep = twinworld.model(recolor_task)
    foil = twinworld.compute(
        twinworld.identify(rep, twinworld.Interventional(step=0, alternative=Recolor(3, 9)))
    ).items[0]
    assert foil.counterfactual.applicable
    target = foil.counterfactual.counterfactual.outcome.key
    cfs = twinworld.compute(twinworld.identify(rep, Contrastive(target, on="train[0]")))
    alts = cfs.alternatives()
    assert alts.kind == "edit_sets"
    assert len(alts.items) == len(cfs.items) and len(alts.scores) == len(cfs.items)
    assert alts.column("sparsity") == tuple(i.metrics.sparsity for i in cfs.items)
    assert set(alts.ranked("min_proximity")) == set(range(len(cfs.items)))


def test_generic_policies_deterministic():
    items = (object(), object(), object())
    scores = (
        {"proximity": 2.0, "plausible": None},
        {"proximity": 1.0, "plausible": True},
        {"proximity": 3.0, "plausible": False},
    )
    alts = Alternatives("edit_sets", items, scores)
    assert GENERIC_POLICIES["min_proximity"](alts, random.Random(0)) == 1
    assert GENERIC_POLICIES["max_plausibility"](alts, random.Random(0)) == 1
    ranked = alts.ranked("min_proximity")
    assert ranked == (1, 0, 2)


def test_class_alternatives_reproduces_policy_choices(small_ambiguous_task):
    rep = twinworld.model(small_ambiguous_task)
    report = twinworld.assess(rep)
    assert report.underdetermined
    alts = class_alternatives(rep, report)
    assert alts.kind == "program_classes"
    assert len(alts.items) == len(report.discrimination.classes)
    rng = random.Random(3)
    for name in ("first", "shortest", "largest_class", "probe_stability"):
        wrapped = as_policy(POLICIES[name])
        assert wrapped(alts, rng) == POLICIES[name](rep, report, rng)


def test_rank_preimages_rank_of_true_origin():
    rule = ObjectRule(ByColour(3), Delete())
    pre = parse_grid(T("300", "000", "005"))
    observed = rule.apply(pre)
    alts = rank_preimages(rule, observed, limit=64)
    assert alts.kind == "preimages"
    assert any(candidate == pre for candidate in alts.items)  # the true origin is findable
    assert alts.column("order") == tuple(range(len(alts.items)))
    assert all(p >= 0 for p in alts.column("proximity"))


# ------------------------------------------------------- diversity + narrative


def test_diversity_wired_into_contrastive_sets(move_recolor_task):
    rep = twinworld.model(move_recolor_task)
    base = twinworld.compute(
        twinworld.identify(rep, twinworld.Interventional(step=0, alternative=Recolor(2, 9)))
    )
    assert base.diversity is None  # per-trace interventional sets stay unranked
    # a contrastive set with a single edit answer reports None (no pair to compare)
    foil = base.items[0]
    if foil.counterfactual.applicable:
        target = foil.counterfactual.counterfactual.outcome.key
        cfs = twinworld.compute(twinworld.identify(rep, Contrastive(target, on="train[0]")))
        applicable = sum(1 for i in cfs.items if i.counterfactual.applicable)
        if applicable >= 2:
            assert cfs.diversity is not None and cfs.diversity >= 0.0
        else:
            assert cfs.diversity is None


def test_pn_robustness_narrative_says_values():
    grid = T("00300", "00000", "00000", "00000", "00003")
    task = Task(train=((grid, T("00400", "00000", "00000", "00000", "00004")),), test=())
    rep = twinworld.model(task, primitives=[Recolor(3, 4)], induction="never", max_depth=1)
    pn = twinworld.compute(
        twinworld.identify(rep, PertinentNegative(max_cells=1, max_witnesses=2))
    )
    text = "\n".join(item.narrative for item in pn.items)
    assert "colour(s)" not in text
    assert ("value(s)" in text) or ("load-bearing" in text)
