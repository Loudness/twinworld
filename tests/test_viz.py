"""Visualization: renderers pin the HTML contract, reports compose the pipeline,
and the server routes are pure functions — no sockets, browser, or arckit."""

import html

import twinworld
from conftest import T
from twinworld.domains.blocks import build_grid, candidate_moves, task_from_towers
from twinworld.representation import as_grid, parse_grid
from twinworld.viz import (
    PALETTE,
    VizApp,
    full_report,
    grid_html,
    report_html,
    save_report,
    state_html,
)

# ------------------------------------------------------------------ renderers


def test_grid_html_cells():
    doc = grid_html(T("012", "345"))
    assert doc.count("<td") == 6
    for colour in range(6):
        assert PALETTE[colour] in doc


def test_grid_html_escapes_caption():
    doc = grid_html(T("0"), caption="recolor(3->9) <x>")
    assert "recolor(3-&gt;9) &lt;x&gt;" in doc


def test_grid_html_diff_marks_changed_cells():
    doc = grid_html(T("012", "345"), diff_against=T("092", "375"))
    assert doc.count('class="d"') == 2
    shape_mismatch = grid_html(T("012", "345"), diff_against=T("01"))
    assert 'class="d"' not in shape_mismatch


def test_grid_html_out_of_range_colour():
    assert "#FFFFFF" in grid_html(as_grid([[12]]))


def test_state_html_outlines():
    doc = state_html(parse_grid(T("330", "000", "003")))
    assert "2 object(s)" in doc
    assert "bt" in doc and "bl" in doc  # every object cell borders the background


# -------------------------------------------------------------------- reports


def test_report_solved_sections(recolor_task):
    rep = twinworld.model(recolor_task)
    doc = report_html(rep)
    assert doc.startswith("<!doctype")
    assert doc.count("<style") == 1
    assert html.escape(str(rep.solution.program[0])) in doc
    for title in (
        "Demonstrations",
        "Segmentation",
        "Solving traces",
        "Interventional",
        "Pertinent negatives",
        "re-segmentation",
        "Confidence gate",
        "Refutation battery",
    ):
        assert title in doc


def test_report_test_prediction_tick(recolor_task):
    assert "matches expected" in full_report(recolor_task)


def test_report_unsolved_degrades(unsolvable_task):
    doc = full_report(unsolvable_task)
    assert "no program exists" in doc
    assert "expected outcome" in doc  # honest banner, not a crash
    assert doc.count('<table class="g"') >= 2  # the pairs still render
    assert "Segmentation" in doc  # what the solver saw, even without a fit
    assert "Interventional" not in doc


def test_report_contrastive_with_foil(move_recolor_task):
    doc = full_report(
        move_recolor_task, foil=T("0000", "0770", "0000", "0007"), foil_on="train[0]"
    )
    assert "certified minimal" in doc
    assert "responsibility" in doc


def test_report_blocks_world():
    task = task_from_towers(
        train=[
            ([[1, 2], [], []], [[], [2], [1]]),
            ([[1, 2], [3], []], [[], [3, 2], [1]]),
        ],
        test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
    )
    doc = full_report(
        task,
        primitives=candidate_moves(task),
        induction="never",
        max_depth=2,
        foil=build_grid([[], [5, 2, 1], []]),
        foil_on="test[0]",
    )
    assert "move block" in doc
    assert "Contrastive" in doc


# --------------------------------------------------------------------- server


def test_route_index_and_404(recolor_task):
    app = VizApp(tasks_fn=lambda: {"synthetic-recolor": recolor_task})
    status, body = app.route("/")
    assert status == 200
    assert "/task/synthetic-recolor" in body
    assert app.route("/nope")[0] == 404
    assert app.route("/task/missing")[0] == 404


def test_route_task_page_caches(recolor_task):
    calls = []

    def tasks_fn():
        calls.append(1)
        return {"synthetic-recolor": recolor_task}

    app = VizApp(tasks_fn=tasks_fn)
    status, first = app.route("/task/synthetic-recolor")
    assert status == 200
    assert "Demonstrations" in first
    _, second = app.route("/task/synthetic-recolor")
    assert second == first
    assert len(calls) == 1


def test_route_index_marks_visited_fit_status(recolor_task, unsolvable_task):
    app = VizApp(tasks_fn=lambda: {"a": recolor_task, "b": unsolvable_task})
    before = app.route("/")[1]
    assert 'class="pass">fits<' not in before
    app.route("/task/a")
    app.route("/task/b")
    after = app.route("/")[1]
    assert 'class="pass">fits<' in after
    assert 'class="fail">no fit<' in after


def test_route_index_lists_known_fitting_starters(recolor_task):
    app = VizApp(tasks_fn=lambda: {"a79310a0": recolor_task, "zzz": recolor_task})
    body = app.route("/")[1]
    assert "start with a task known to fit" in body
    no_starters = VizApp(tasks_fn=lambda: {"zzz": recolor_task})
    assert "start with" not in no_starters.route("/")[1]


def test_scan_populates_status_without_rendering(recolor_task, unsolvable_task):
    app = VizApp(tasks_fn=lambda: {"a": recolor_task, "b": unsolvable_task})
    assert app.scan() == (1, 2)
    assert app._pages == {}  # pages stay lazy; scan records status only
    body = app.route("/")[1]
    assert 'class="pass">fits<' in body
    assert 'class="fail">no fit<' in body
    assert "1/2 scanned task(s) fit" in body


def test_route_missing_arckit_message():
    def tasks_fn():
        raise ModuleNotFoundError("No module named 'arckit'")

    app = VizApp(tasks_fn=tasks_fn)
    status, body = app.route("/")
    assert status == 500
    assert "pip install" in body


def test_save_report_round_trip(tmp_path):
    doc = grid_html(T("01"))
    path = save_report(doc, tmp_path / "r.html")
    assert path.read_text(encoding="utf-8") == doc
