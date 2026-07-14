import pytest
from conftest import T

import dowhat
from dowhat import Backtracking, IdentificationError, Interventional, Recolor


def test_end_to_end_pipeline(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    assert rep.solution.program == (Recolor(3, 4),)

    identified = dowhat.identify(rep, Interventional(step=0, alternative=Recolor(3, 9)))
    cfs = dowhat.compute(identified)
    # one counterfactual per train trace + per test trace
    assert len(cfs.items) == len(recolor_task.train) + len(recolor_task.test)
    for item in cfs.items:
        assert item.metrics.validity is False
        assert item.metrics.sparsity == 1

    report = dowhat.refute(rep)
    assert report.passed


def test_golden_contrastive_narrative(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    cfs = dowhat.compute(dowhat.identify(rep, Interventional(0, Recolor(3, 9))))
    text = cfs.items[0].narrative
    assert text == (
        "train[0]: at step 0 the solver applied [recolor(3->4)] rather than "
        "[recolor(3->9)]; had it chosen [recolor(3->9)], the task would NO LONGER "
        "be solved (sparsity 1 edit, outcome proximity 1)."
    )


def test_identify_rejects_out_of_range_step(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="does not exist"):
        dowhat.identify(rep, Interventional(step=5, alternative=Recolor(3, 9)))


def test_identify_rejects_factual_alternative(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="factual"):
        dowhat.identify(rep, Interventional(step=0, alternative=Recolor(3, 4)))


def test_identify_rejects_dimension_change(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="dimensions"):
        dowhat.identify(rep, Backtracking(edited_input=T("33", "00")))


def test_backtracking_counterfactual(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    edited = T("00000", "03300", "03000", "00000", "00050")  # spectator moved
    cfs = dowhat.compute(dowhat.identify(rep, Backtracking(edited)))
    (item,) = cfs.items
    assert item.counterfactual.mode == "backtracking"
    assert item.metrics.proximity > 0  # outcome differs where the spectator sits
    assert "rerunning the same program" in item.narrative


def test_unknown_backend_rejected(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    identified = dowhat.identify(rep, Interventional(0, Recolor(3, 9)))
    with pytest.raises(ValueError, match="unknown backend"):
        dowhat.compute(identified, backend="cf.magic")
