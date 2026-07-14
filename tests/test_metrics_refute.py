from conftest import T

from dowhat import Recolor, Task, Translate, candidate_primitives, parse_grid
from dowhat.engine import intervene, solve
from dowhat.metrics import evaluate, proximity, sparsity, validity
from dowhat.refute import refutation_battery


def fit(task, max_depth=2):
    return solve(task, candidate_primitives(task.colours(), background=0), max_depth=max_depth)


def test_sparsity_counts_action_edits():
    a, b, c = Recolor(1, 2), Recolor(3, 4), Translate(1, 0, colour=5)
    assert sparsity((a, b), (a, b)) == 0
    assert sparsity((a, b), (a, c)) == 1
    assert sparsity((a, b), (c,)) == 2  # one substitution + one length edit


def test_proximity_zero_iff_same_outcome():
    s = parse_grid(T("0300", "0000", "0050"))
    assert proximity(s, s) == 0.0
    recoloured = parse_grid(T("0400", "0000", "0050"))
    assert proximity(s, recoloured) == 1.0  # one object differs in colour only
    moved = parse_grid(T("3000", "0000", "0050"))
    assert proximity(s, moved) == 1.0  # one object differs in location only


def test_validity_certificate(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    trace = sol.train_traces[0]
    same = intervene(sol, trace, 0, sol.program[0])
    assert validity(sol, same) is True  # factual program certifiably solves
    broken = intervene(sol, trace, 0, Recolor(3, 9))
    assert validity(sol, broken) is False  # certificate, not estimate


def test_evaluate_bundles_metrics(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    cf = intervene(sol, sol.train_traces[0], 0, Recolor(3, 9))
    m = evaluate(sol, cf)
    assert m.validity is False
    assert m.sparsity == 1
    assert m.proximity > 0
    assert m.applicable and m.divergence_step == 0


def test_placebo_refuter_passes_on_honest_program(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    report = refutation_battery(sol)
    (row,) = report.rows
    assert row.name == "placebo_intervention"
    assert row.passed is True
    assert report.passed


def test_placebo_refuter_skips_without_spectator():
    task = Task(
        train=((T("300", "000", "000"), T("400", "000", "000")),),
        test=((T("030", "000", "000"), T("040", "000", "000")),),
        task_id="no-spectator",
    )
    sol = fit(task, max_depth=1)
    (row,) = refutation_battery(sol).rows
    assert row.passed is None  # skipped, honestly reported
