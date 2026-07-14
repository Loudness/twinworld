"""Milestone 2: multi-abstraction model() and counterfactual re-segmentation."""

import pytest
from conftest import T

import dowhat
from dowhat import IdentificationError, Representational, Task, Translate


@pytest.fixture
def composite_task() -> Task:
    """A 2-3 composite must move right as ONE thing while a 7 stays put.

    Only the colour-blind composite abstraction (mcc) sees the {2,3} pair as a
    single object, so only mcc solves this at depth 1: under cc4/cc8 the parts
    are separate objects and moving the 2 collides with the 3.
    """
    return Task(
        train=(
            (T("00000", "23000", "00007"), T("00000", "02300", "00007")),
            (T("00000", "02300", "70000"), T("00000", "00230", "70000")),
        ),
        test=((T("23000", "00000", "00070"), T("02300", "00000", "00070")),),
        task_id="synthetic-composite",
    )


def test_model_records_all_abstractions(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    # a plain recolour is representation-independent: every scheme solves it
    assert set(rep.solutions) == {"cc4", "cc8", "mcc"}
    assert rep.failures == {}
    assert rep.abstraction == "cc4"  # tie on program length breaks to listed order


def test_model_selects_the_only_working_abstraction(composite_task):
    rep = dowhat.model(composite_task, max_depth=1)
    assert rep.abstraction == "mcc"
    assert rep.solution.program == (Translate(0, 1, colour=2),)
    assert set(rep.failures) == {"cc4", "cc8"}


def test_resegmentation_robust_when_alternative_solves(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    cfs = dowhat.compute(dowhat.identify(rep, Representational("mcc")))
    (item,) = cfs.items
    assert item.metrics.validity is True
    assert item.metrics.proximity == 0.0  # same outcome, re-derived under mcc
    assert "robust to re-segmentation" in item.narrative


def test_resegmentation_load_bearing_when_alternative_fails(composite_task):
    rep = dowhat.model(composite_task, max_depth=1)
    cfs = dowhat.compute(dowhat.identify(rep, Representational("cc4")))
    (item,) = cfs.items
    assert item.metrics.validity is False
    assert not item.metrics.applicable
    assert "load-bearing" in item.narrative


def test_resegmentation_lazy_solve_for_unattempted_abstraction(recolor_task):
    rep = dowhat.model(recolor_task, abstractions=("cc4",), max_depth=1)
    assert "mcc" not in rep.solutions
    cfs = dowhat.compute(dowhat.identify(rep, Representational("mcc")))
    assert cfs.items[0].metrics.validity is True
    assert "mcc" in rep.solutions  # cached after the lazy refit


def test_identify_rejects_unknown_and_factual_abstraction(recolor_task):
    rep = dowhat.model(recolor_task, max_depth=1)
    with pytest.raises(IdentificationError, match="unknown abstraction"):
        dowhat.identify(rep, Representational("voxels"))
    with pytest.raises(IdentificationError, match="factual"):
        dowhat.identify(rep, Representational(rep.abstraction))
