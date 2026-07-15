import pytest
from conftest import T

from twinworld import Recolor, Translate, UnsolvedTaskError, as_grid, candidate_primitives
from twinworld.engine import backtrack, intervene, solve


def fit(task, max_depth=2):
    primitives = candidate_primitives(task.colours(), background=0)
    return solve(task, primitives, max_depth=max_depth)


def test_solves_depth1_recolor(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    assert sol.program == (Recolor(3, 4),)
    assert sol.test_traces[0].outcome.key == as_grid(recolor_task.test[0][1])


def test_solves_depth2_move_recolor(move_recolor_task):
    sol = fit(move_recolor_task)
    assert len(sol.program) == 2
    for trace, (_, expected) in zip(sol.train_traces, move_recolor_task.train):
        assert trace.outcome.key == as_grid(expected)
    assert sol.test_traces[0].outcome.key == as_grid(move_recolor_task.test[0][1])


def test_unsolvable_raises(unsolvable_task):
    with pytest.raises(UnsolvedTaskError):
        fit(unsolvable_task)


def test_trajectory_dag_records_expansions(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    # every factual transition is an edge in the DAG
    for trace in sol.train_traces:
        for a, b in zip(trace.states, trace.states[1:]):
            assert sol.dag.has_edge(a.key, b.key)
    # and the DAG holds more than the solution: failed candidates were recorded too
    assert sol.dag.number_of_nodes() > sum(len(t.states) for t in sol.train_traces)


def test_galles_pearl_composition(recolor_task):
    """do(A_t = factual mechanism) changes nothing."""
    sol = fit(recolor_task, max_depth=1)
    trace = sol.train_traces[0]
    cf = intervene(sol, trace, 0, trace.mechanisms[0])
    assert cf.applicable and cf.counterfactual == trace


def test_galles_pearl_effectiveness(move_recolor_task):
    """After do(A_t = m), the counterfactual world actually uses m at t."""
    sol = fit(move_recolor_task)
    trace = sol.train_traces[0]
    alternative = Recolor(2, 9) if sol.program[0] != Recolor(2, 9) else Recolor(2, 8)
    cf = intervene(sol, trace, 0, alternative)
    assert cf.program[0] == alternative
    if cf.applicable:
        assert cf.counterfactual.states[1] == alternative.apply(trace.states[0])


def test_twin_world_shares_prefix_by_reference(move_recolor_task):
    sol = fit(move_recolor_task)
    trace = sol.train_traces[0]
    # move the colour-7 spectator up: applicable at states[1] whatever program was induced
    alt = Translate(-1, 0, colour=7)
    cf = intervene(sol, trace, 1, alt)
    assert cf.applicable
    assert cf.counterfactual.states[0] is trace.states[0]
    assert cf.counterfactual.states[1] is trace.states[1]
    assert cf.divergence_step == 1


def test_inapplicable_alternative_yields_inapplicable_world(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    trace = sol.train_traces[0]
    cf = intervene(sol, trace, 0, Recolor(9, 1))  # no colour-9 object exists
    assert not cf.applicable and cf.counterfactual is None


def test_backtracking_reruns_same_laws(recolor_task):
    sol = fit(recolor_task, max_depth=1)
    trace = sol.train_traces[0]
    edited = T("00000", "03300", "03000", "00000", "00000")  # spectator removed
    cf = backtrack(sol, trace, edited)
    assert cf.mode == "backtracking"
    assert cf.applicable
    assert cf.program == trace.mechanisms
    assert cf.counterfactual.outcome.key == T("00000", "04400", "04000", "00000", "00000")
