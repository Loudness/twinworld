"""Experiment 3: certified minimal counterfactuals, responsibility,
underdetermination diagnosis, and the ground-truth benchmark."""

import random

import pytest
from conftest import T

import dowhat
from dowhat import (
    ByColour,
    Contrastive,
    IdentificationError,
    ObjectRule,
    Recolor,
    RecolourTo,
    Task,
    induce_rules,
)
from dowhat.benchmark import generate_task, random_task
from dowhat.discriminate import diagnose, probes
from dowhat.engine import solve_all
from dowhat.metrics import diversity, edited_steps, responsibility_profile


# ----------------------------------------------------- contrastive: k=1 case


def test_contrastive_k1_certified(move_recolor_task):
    rep = dowhat.model(move_recolor_task)
    # foil: the bar ends colour 7 instead of 6 (7 is in the task palette, so
    # the pool contains a mechanism that can produce it — reachable by design)
    target = T("0000", "0770", "0000", "0007")
    cfs = dowhat.compute(dowhat.identify(rep, Contrastive(target, on="train[0]")))
    assert cfs.items
    for item in cfs.items:
        assert item.metrics.sparsity == 1
        assert "k=1, certified minimal" in item.narrative
        assert item.counterfactual.counterfactual.outcome.key == target
    # only the recolour step can bear responsibility for the colour foil
    assert cfs.responsibility == {0: 0.0, 1: 1.0}


# ------------------------------------------- contrastive: certified k=2 case


def test_contrastive_k2_certified(three_way_move_task):
    rep = dowhat.model(three_way_move_task)
    # foil: colour 2 moves DOWN and colour 6 moves RIGHT; colour 3 as factual.
    # No single edit can change two colours' motions, so k=2 is forced.
    target = T("00000", "20000", "00300", "00006", "00000")
    cfs = dowhat.compute(dowhat.identify(rep, Contrastive(target, on="train[0]")))
    assert cfs.items
    for item in cfs.items:
        assert item.metrics.sparsity == 2
        assert "k=2, certified minimal" in item.narrative
    # Chockler-Halpern: joint minimal edit of size 2 -> responsibility 1/2 each
    assert cfs.responsibility == {0: 0.5, 1: 0.0, 2: 0.5}


# --------------------------------------------- contrastive: unreachable foil


def test_contrastive_unreachable_is_certified(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    target = T("99999", "99999", "99999", "99999", "99999")
    cfs = dowhat.compute(dowhat.identify(rep, Contrastive(target, on="train[0]")))
    (item,) = cfs.items
    assert not item.metrics.applicable
    assert "robust" in item.narrative and "certified" in item.narrative
    assert cfs.responsibility is None


def test_contrastive_identify_errors(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    factual_out = rep.solution.train_traces[0].outcome.grid
    with pytest.raises(IdentificationError, match="factual outcome"):
        dowhat.identify(rep, Contrastive(factual_out, on="train[0]"))
    with pytest.raises(IdentificationError, match="does not exist"):
        dowhat.identify(rep, Contrastive(factual_out, on="train[9]"))
    with pytest.raises(IdentificationError, match="trace reference"):
        dowhat.identify(rep, Contrastive(factual_out, on="banana"))


# --------------------------------------------------- metrics building blocks


def test_edited_steps_and_diversity(recolor_task):
    from dowhat.engine import intervene

    sol = dowhat.model(recolor_task, max_depth=1).solution
    trace = sol.train_traces[0]
    cf9 = intervene(sol, trace, 0, Recolor(3, 9))
    cf7 = intervene(sol, trace, 0, Recolor(3, 7))
    assert edited_steps(cf9) == frozenset({0})
    # two applicable counterfactuals whose outcomes differ in one object colour
    assert diversity([cf9, cf7]) == 1.0
    assert diversity([cf9]) == 0.0


def test_responsibility_profile_takes_max_over_sets(recolor_task):
    from dowhat.engine import intervene

    sol = dowhat.model(recolor_task, max_depth=1).solution
    trace = sol.train_traces[0]
    cf = intervene(sol, trace, 0, Recolor(3, 9))
    assert responsibility_profile([cf]) == {0: 1.0}
    assert responsibility_profile([]) == {}


# ------------------------------------------------ underdetermination probing


@pytest.fixture
def ambiguous_task() -> Task:
    """Largest object == the colour-2 object in every train pair: 'recolour the
    largest' and 'recolour colour 2' fit equally. Only a counterfactual probe
    (e.g. deleting the bar) separates them."""
    return Task(
        train=(
            (T("22200", "00000", "00030"), T("55500", "00000", "00030")),
            (T("00000", "02220", "30000"), T("00000", "05550", "30000")),
        ),
        test=((T("00222", "30000", "00000"), T("00555", "30000", "00000")),),
        task_id="synthetic-ambiguous",
    )


def test_solve_all_finds_the_competing_hypotheses(ambiguous_task):
    rules = induce_rules(ambiguous_task)
    fits = solve_all(ambiguous_task, rules, max_depth=1)
    # at least the colour-based and the size-based readings both fit
    assert (ObjectRule(ByColour(2), RecolourTo(5)),) in fits
    assert (ObjectRule(dowhat.Largest(), RecolourTo(5)),) in fits
    assert len(fits) >= 2


def test_diagnose_separates_hypotheses_with_a_probe(ambiguous_task):
    fits = solve_all(ambiguous_task, induce_rules(ambiguous_task), max_depth=1)
    report = diagnose(ambiguous_task, fits)
    assert report.underdetermined
    assert len(report.classes) >= 2
    assert report.probe is not None
    assert report.outputs[0] != report.outputs[1]  # the probe truly separates them


def test_diagnose_single_class_when_determined(three_way_move_task):
    # the three colour-specific moves admit only order permutations of one
    # program — behaviourally a single class on every probe
    fits = solve_all(
        three_way_move_task, induce_rules(three_way_move_task), max_depth=3
    )
    assert len(fits) >= 2  # several orderings fit ...
    report = diagnose(three_way_move_task, fits)
    assert not report.underdetermined  # ... but they are one behaviour
    assert report.probe is None


def test_probes_are_deterministic(recolor_task):
    assert probes(recolor_task) == probes(recolor_task)
    # base train inputs are always included
    grids = probes(recolor_task)
    assert dowhat.as_grid(recolor_task.train[0][0]) in grids


# ------------------------------------------------- ground-truth benchmark


def test_generate_task_ground_truth_by_construction():
    latent = (ObjectRule(ByColour(2), RecolourTo(8)),)
    inputs = [
        T("200", "000", "003"),
        T("020", "000", "300"),
        T("002", "030", "000"),
    ]
    task = generate_task(latent, inputs)
    assert task is not None
    assert task.train[0][1] == T("800", "000", "003")
    assert len(task.train) == 2 and len(task.test) == 1


def test_benchmark_end_to_end_minimality_gap_zero():
    rng = random.Random(0)
    for _ in range(50):
        instance = random_task(rng)
        if instance is None:
            continue
        task, latent = instance
        try:
            rep = dowhat.model(task)
        except dowhat.UnsolvedTaskError:
            continue
        # foil: outcome of a one-edit variant of the induced program, using a
        # mechanism FROM the contrastive pool, so the true minimal k is 1 by
        # construction; the generator must certify k=1
        trace = rep.solution.train_traces[0]
        pool = [*induce_rules(task, rep.abstraction), *rep.primitives]
        alt = None
        for mech in pool:
            if mech == trace.mechanisms[-1]:
                continue
            candidate = rep.solution.cache.run(
                trace.states[0], trace.mechanisms[:-1] + (mech,)
            )
            if candidate is not None and candidate.outcome.key != trace.outcome.key:
                alt = candidate
                break
        if alt is None:
            continue
        cfs = dowhat.compute(
            dowhat.identify(rep, Contrastive(alt.outcome.grid, on="train[0]"))
        )
        assert cfs.items and cfs.items[0].metrics.applicable
        assert all(i.metrics.sparsity == 1 for i in cfs.items)  # minimality gap 0
        return  # one full pass is the smoke test
    pytest.skip("no benchmark instance materialized in 50 draws")