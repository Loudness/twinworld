"""Phase 10: CausalARC adapter — pure parsing plus a network-gated live load."""

import pytest

from dowhat import as_grid
from dowhat.domains.causalarc import load_causalarc, parse_causalarc


def test_parse_causalarc_is_pure_and_network_free():
    raw = {
        "SCMtest": {
            "scm": "def f(): ...",
            "train": [
                {
                    "input": [[1, 0], [0, 0]],
                    "output": [[2]],
                    "cf_types": ["do_hard (color 1 = 0)"],
                    "cf_inputs": [[[0, 0], [0, 0]]],
                    "cf_outputs": [[[0]]],
                }
            ],
            "test": [{"input": [[0, 1], [0, 0]], "output": [[2]]}],
        }
    }
    (ct,) = parse_causalarc(raw)
    assert ct.task.task_id == "SCMtest"
    assert ct.task.train[0][1] == as_grid([[2]])
    assert ct.scm_source.startswith("def f")
    (demos,) = ct.cf_demos
    assert demos[0].kind == "do_hard (color 1 = 0)"
    assert demos[0].input == as_grid([[0, 0], [0, 0]])
    assert demos[0].output == as_grid([[0]])


def test_load_causalarc_live():
    try:
        tasks = load_causalarc("counting", timeout=60)
    except Exception as err:  # offline / HF down: the adapter is network-optional
        pytest.skip(f"CausalARC dataset unavailable: {err}")
    assert len(tasks) == 10
    assert all(len(ct.task.train) == 5 and len(ct.task.test) == 1 for ct in tasks)
    assert any(demo for ct in tasks for demos in ct.cf_demos for demo in demos)
