"""Phase 8: diagnose() as a confidence gate — the solver knows when it doesn't know."""

import pytest
from conftest import T

import twinworld
from twinworld import Task


@pytest.fixture
def treacherous_task() -> Task:
    """Largest == colour-2 in every train pair, but the TEST input separates
    the two readings: there the largest object is colour 3. Whatever the
    solver picks, the demonstrations cannot justify it — the gate must say so
    without ever looking at the test output."""
    return Task(
        train=(
            (T("220000", "000000", "000300"), T("550000", "000000", "000300")),
            (T("000000", "002200", "300000"), T("000000", "005500", "300000")),
        ),
        test=(
            # colour-3 bar of size 3 (now the largest) + a colour-2 dot
            (T("333000", "000000", "000020"), T("333000", "000000", "000050")),
        ),
        task_id="synthetic-treacherous",
    )


def test_gate_low_when_classes_disagree_on_test(treacherous_task):
    rep = twinworld.model(treacherous_task)
    report = twinworld.assess(rep)
    assert report.underdetermined
    assert not report.unanimous_on_test
    assert report.confidence == "low"
    assert len(set(report.predictions)) >= 2  # genuinely different test answers
    prediction, _ = twinworld.predict(rep)
    assert prediction is None  # abstention


def test_gate_high_despite_ambiguity_when_test_is_unanimous():
    # largest == colour-2 on trains AND on the test input: the hypotheses
    # disagree on probes, but every class gives the same test answer
    task = Task(
        train=(
            (T("220000", "000000", "000300"), T("550000", "000000", "000300")),
            (T("000000", "002200", "300000"), T("000000", "005500", "300000")),
        ),
        test=((T("000022", "000000", "300000"), T("000055", "000000", "300000")),),
        task_id="synthetic-unanimous",
    )
    rep = twinworld.model(task)
    report = twinworld.assess(rep)
    assert report.underdetermined  # the ambiguity is real ...
    assert report.unanimous_on_test  # ... but harmless for THIS test input
    assert report.confidence == "high"
    prediction, _ = twinworld.predict(rep)
    assert prediction == (twinworld.as_grid(task.test[0][1]),)


def test_gate_high_when_determined(three_way_move_task):
    rep = twinworld.model(three_way_move_task)
    report = twinworld.assess(rep)
    assert not report.underdetermined
    assert report.confidence == "high"
    prediction, _ = twinworld.predict(rep)
    assert prediction == (twinworld.as_grid(three_way_move_task.test[0][1]),)
