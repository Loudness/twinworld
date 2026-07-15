import pytest
from conftest import T

import twinworld
from twinworld import (
    Backtracking,
    ByColour,
    IdentificationError,
    Interventional,
    ObjectRule,
    Recolor,
    RecolourTo,
)


def test_end_to_end_pipeline(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    assert rep.solution.program == (ObjectRule(ByColour(3), RecolourTo(4)),)
    assert rep.solution.strategy == "analogy"

    identified = twinworld.identify(rep, Interventional(step=0, alternative=Recolor(3, 9)))
    cfs = twinworld.compute(identified)
    # one counterfactual per train trace + per test trace
    assert len(cfs.items) == len(recolor_task.train) + len(recolor_task.test)
    for item in cfs.items:
        assert item.metrics.validity is False
        assert item.metrics.sparsity == 1

    report = twinworld.refute(rep)
    assert report.passed


def test_golden_contrastive_narrative(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    cfs = twinworld.compute(twinworld.identify(rep, Interventional(0, Recolor(3, 9))))
    text = cfs.items[0].narrative
    assert text == (
        "train[0]: at step 0 the solver applied [recolour colour-3 objects to 4] "
        "rather than [recolor(3->9)]; had it chosen [recolor(3->9)], the task would "
        "NO LONGER be solved (sparsity 1 edit, outcome proximity 1)."
    )


def test_identify_rejects_out_of_range_step(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="does not exist"):
        twinworld.identify(rep, Interventional(step=5, alternative=Recolor(3, 9)))


def test_identify_rejects_factual_alternative(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    factual = rep.solution.program[0]
    with pytest.raises(IdentificationError, match="factual"):
        twinworld.identify(rep, Interventional(step=0, alternative=factual))


def test_identify_rejects_dimension_change(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="dimensions"):
        twinworld.identify(rep, Backtracking(edited_input=T("33", "00")))


def test_backtracking_counterfactual(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    edited = T("00000", "03300", "03000", "00000", "00050")  # spectator moved
    cfs = twinworld.compute(twinworld.identify(rep, Backtracking(edited)))
    (item,) = cfs.items
    assert item.counterfactual.mode == "backtracking"
    assert item.metrics.proximity > 0  # outcome differs where the spectator sits
    assert "rerunning the same program" in item.narrative


def test_unknown_backend_rejected(recolor_task):
    rep = twinworld.model(recolor_task, max_depth=1)
    identified = twinworld.identify(rep, Interventional(0, Recolor(3, 9)))
    with pytest.raises(ValueError, match="unknown backend"):
        twinworld.compute(identified, backend="cf.magic")
