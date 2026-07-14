"""Phase 10: LLM baseline — prompt/parse round trips are endpoint-free; one
live smoke test runs only when the local endpoint is reachable."""

import pytest
from conftest import T

from dowhat import Task
from dowhat.baselines.llm import (
    build_prompt,
    grid_text,
    llm_predict,
    parse_grid_text,
    reachable,
)
from dowhat.domains.causalarc import CfDemo


def test_grid_text_round_trip():
    grid = T("340", "005")
    assert parse_grid_text(grid_text(grid)) == grid


def test_parse_grid_text_takes_the_last_clean_block():
    reply = "Sure! Working through it...\n123\n\nFinal answer:\n```\n34\n05\n```\nDone."
    assert parse_grid_text(reply) == T("34", "05")
    assert parse_grid_text("no grids here") is None
    assert parse_grid_text("12\n345") is None  # ragged rows are not a grid


def test_build_prompt_carries_pairs_and_counterfactuals(recolor_task):
    prompt = build_prompt(recolor_task)
    assert grid_text(recolor_task.train[0][0]) in prompt
    assert "Test input:" in prompt
    demos = (
        (CfDemo("do_hard (color 3 = 0)", T("00", "00"), T("00", "00")),),
    ) + ((),) * (len(recolor_task.train) - 1)
    with_cf = build_prompt(recolor_task, cf_demos=demos)
    assert "do_hard (color 3 = 0)" in with_cf


def test_reachable_false_on_dead_port():
    assert not reachable("http://localhost:9/v1/chat/completions", timeout=0.5)


LIVE = reachable()


@pytest.mark.skipif(not LIVE, reason="local LLM endpoint not reachable")
def test_llm_predict_live_smoke():
    identity = Task(
        train=(
            (T("120", "000"), T("120", "000")),
            (T("003", "400"), T("003", "400")),
        ),
        test=((T("050", "006"), T("050", "006")),),
        task_id="identity-smoke",
    )
    guess = llm_predict(identity, timeout=240)
    assert guess is not None  # a grid came back and parsed (correctness not asserted)
