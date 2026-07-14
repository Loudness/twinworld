"""Phase 7: clingo searches the program space; the engine verifies."""

import pytest
from conftest import T

import dowhat
from dowhat import ByColour, Not, ObjectRule, RecolourTo, Task
from dowhat.discriminate import probes, signature
from dowhat.engine import ApplyCache

pytest.importorskip("clingo")

from dowhat.asp_solver import asp_solve  # noqa: E402


def test_asp_solves_recolour_and_engine_verifies(recolor_task):
    result = asp_solve(recolor_task, max_depth=1)
    assert result.depth == 1
    assert result.verified >= 1
    assert result.proposed >= result.verified
    assert (ObjectRule(ByColour(3), RecolourTo(4)),) in result.programs


def test_asp_model_strategy_agrees_with_analogy_when_determined(three_way_move_task):
    # recolor_task is underdetermined (ByColour vs Not(ByColour(spectator))), so
    # ASP may legitimately land in another behavioural class there; the
    # three-way task has a single class, so the strategies MUST agree on it.
    via_asp = dowhat.model(three_way_move_task, induction="asp")
    via_analogy = dowhat.model(three_way_move_task)
    assert via_asp.solution.strategy == "asp"
    probe_grids = probes(three_way_move_task, via_asp.abstraction)
    cache = ApplyCache()
    assert signature(
        via_asp.solution.program, probe_grids, via_asp.abstraction, cache
    ) == signature(via_analogy.solution.program, probe_grids, via_analogy.abstraction, cache)


def test_asp_solves_three_way_at_depth_three(three_way_move_task):
    result = asp_solve(three_way_move_task, max_depth=3)
    assert result.depth == 3
    assert result.verified >= 1
    rep = dowhat.model(three_way_move_task, induction="asp")
    expected = dowhat.as_grid(three_way_move_task.test[0][1])
    assert rep.solution.test_traces[0].outcome.key == expected


def test_asp_searches_negation_space():
    task = Task(  # recolour everything except the largest (colour varies)
        train=(
            (T("22200", "00000", "30044"), T("22200", "00000", "50055")),
            (T("00000", "06660", "20070"), T("00000", "06660", "50050")),
        ),
        test=((T("33300", "00000", "20004"), T("33300", "00000", "50005")),),
        task_id="synthetic-except-largest",
    )
    result = asp_solve(task, max_depth=1)
    assert (ObjectRule(Not(dowhat.Largest()), RecolourTo(5)),) in result.programs


def test_asp_declares_its_fragment():
    # a multi-colour composite (mcc abstraction) is outside the solid-object fragment
    task = Task(
        train=((T("230", "000", "000"), T("023", "000", "000")),),
        test=((T("023", "000", "000"), T("002", "300", "000")),),
        task_id="composite",
    )
    result = asp_solve(task, abstraction="mcc", max_depth=1)
    assert result.programs == () and result.depth is None
