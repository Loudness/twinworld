"""Self-contained HTML visualization of the model → identify → compute → refute pipeline.

Pure stdlib; report rendering works on any :class:`Task` and needs no arckit.
Save a single-file page with ``save_report(full_report(task), "report.html")``
or open one directly with ``show(full_report(task))``. ``python -m dowhat.viz``
serves the ARC corpus at http://127.0.0.1:8008 (requires the ``arc`` extra);
tasks are fitted on first click and cached.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import tempfile
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote, unquote, urlsplit

from .api import (
    CausalRepresentation,
    Contrastive,
    CounterfactualItem,
    IdentificationError,
    Interventional,
    PertinentNegative,
    Representational,
    assess,
    compute,
    identify,
    model,
    refute,
)
from .engine import Task, Trace, UnsolvedTaskError
from .metrics import MetricVector
from .representation import ABSTRACTIONS, Grid, StateGraph, as_grid, parse_grid

PALETTE = {
    0: "#252525", 1: "#0074D9", 2: "#FF4136", 3: "#37D449", 4: "#FFDC00",
    5: "#E6E6E6", 6: "#F012BE", 7: "#FF871E", 8: "#54D2EB", 9: "#8D1D2C",
}
_UNKNOWN = "#FFFFFF"

_CSS = """
body{background:#EEEFF6;color:#1b1b1f;font:14px/1.45 system-ui,sans-serif;
  max-width:1100px;margin:24px auto;padding:0 16px}
h1{font-size:22px}
h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid #cfd2e0;padding-bottom:4px}
h3{font-size:14px;margin:14px 0 4px}
code{background:#e4e6f0;border-radius:4px;padding:1px 5px;font-size:13px}
table.g{border-collapse:collapse;background:#404040}
table.g td{width:var(--s,16px);height:var(--s,16px);padding:0;box-sizing:border-box;
  border:1px solid #404040}
table.g td.d{box-shadow:inset 0 0 0 2px #fff,inset 0 0 0 3px #000}
table.g td.bt{border-top:2px solid #fff}table.g td.br{border-right:2px solid #fff}
table.g td.bb{border-bottom:2px solid #fff}table.g td.bl{border-left:2px solid #fff}
figure.grid{display:inline-block;margin:4px;text-align:center;vertical-align:top}
figure.grid figcaption{font-size:11px;color:#555;max-width:180px;margin-top:3px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:6px 0}
.arrow{max-width:170px;font-size:12px;color:#333;text-align:center}
.card{background:#fff;border-radius:8px;padding:10px 14px;margin:10px 0;
  box-shadow:0 1px 2px rgba(20,24,60,.08)}
.metrics{font-size:12px;color:#555;margin-top:6px}
.pass{color:#0a7a33;font-weight:600}.fail{color:#b3122e;font-weight:600}
.skip{color:#777;font-weight:600}
.hi{color:#0a7a33;font-weight:600}.lo{color:#b3122e;font-weight:600}
.banner{background:#fcf3e2;border:1px solid #e3c98f;border-radius:8px;padding:10px 14px}
table.t{border-collapse:collapse;background:#fff;font-size:13px}
table.t td,table.t th{border:1px solid #d5d8e6;padding:4px 10px;text-align:left}
ul{margin:6px 0}
"""

# ------------------------------------------------------------- pure renderers


def _esc(x: object) -> str:
    return html.escape(str(x))


def _cell_px(grid: Grid) -> int:
    return max(6, min(22, 264 // max(len(grid), len(grid[0]), 1)))


def grid_html(grid: Grid, caption: str | None = None, diff_against: Grid | None = None) -> str:
    """One grid as a colour table; cells that differ from ``diff_against`` get a ring."""
    grid = as_grid(grid)
    diff = None
    if diff_against is not None:
        diff_against = as_grid(diff_against)
        if len(diff_against) == len(grid) and len(diff_against[0]) == len(grid[0]):
            diff = diff_against
    rows = []
    for r, row in enumerate(grid):
        cells = []
        for c, v in enumerate(row):
            marked = ' class="d"' if diff is not None and diff[r][c] != v else ""
            cells.append(f'<td{marked} style="background:{PALETTE.get(v, _UNKNOWN)}"></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = f'<table class="g" style="--s:{_cell_px(grid)}px">' + "".join(rows) + "</table>"
    cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    return f'<figure class="grid">{table}{cap}</figure>'


_SIDES = (("bt", (-1, 0)), ("br", (0, 1)), ("bb", (1, 0)), ("bl", (0, -1)))


def state_html(state: StateGraph, caption: str | None = None) -> str:
    """The state's grid with white outlines around each segmented object."""
    grid = state.grid
    oid = {cell: o.oid for o in state.objects for cell in o.cells}
    rows = []
    for r, row in enumerate(grid):
        cells = []
        for c, v in enumerate(row):
            classes = []
            here = oid.get((r, c))
            if here is not None:
                classes = [cls for cls, (dr, dc) in _SIDES if oid.get((r + dr, c + dc)) != here]
            attr = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f'<td{attr} style="background:{PALETTE.get(v, _UNKNOWN)}"></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = f'<table class="g" style="--s:{_cell_px(grid)}px">' + "".join(rows) + "</table>"
    if caption is None:
        caption = f"{state.abstraction} — {len(state.objects)} object(s)"
    return f'<figure class="grid">{table}<figcaption>{_esc(caption)}</figcaption></figure>'


def trace_html(trace: Trace, caption: str | None = None) -> str:
    """state → mechanism → state … for one solving trace."""
    parts = [grid_html(trace.states[0].grid, caption)]
    for mech, state in zip(trace.mechanisms, trace.states[1:]):
        parts.append(f'<div class="arrow">→<br><code>{_esc(mech)}</code></div>')
        parts.append(grid_html(state.grid))
    return _row(*parts)


def _row(*parts: str) -> str:
    return '<div class="row">' + "".join(parts) + "</div>"


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_esc(title)}</h2>{body}</section>"


def _metrics_html(m: MetricVector) -> str:
    prox = "∞" if math.isinf(m.proximity) else f"{m.proximity:g}"
    validity = {True: "true", False: "FALSE", None: "n/a"}[m.validity]
    return (
        f'<div class="metrics">validity {validity} · sparsity {m.sparsity} · '
        f"proximity {prox} · fork at step {m.divergence_step} · "
        f"applicable {str(m.applicable).lower()}</div>"
    )


def _item_html(item: CounterfactualItem, show_inputs: bool = False) -> str:
    """One counterfactual as a card: narrative, factual-vs-CF grids, metrics."""
    cf = item.counterfactual
    parts = [f"<p>{_esc(item.narrative)}</p>"]
    if cf.applicable:
        grids = []
        if show_inputs:
            grids.append(grid_html(cf.factual.states[0].grid, "factual input"))
            grids.append(
                grid_html(
                    cf.counterfactual.states[0].grid,
                    "counterfactual input",
                    diff_against=cf.factual.states[0].grid,
                )
            )
        grids.append(grid_html(cf.factual.outcome.grid, "factual outcome"))
        grids.append(
            grid_html(
                cf.counterfactual.outcome.grid,
                "counterfactual outcome",
                diff_against=cf.factual.outcome.grid,
            )
        )
        parts.append(_row(*grids))
    parts.append(_metrics_html(item.metrics))
    return '<div class="card">' + "".join(parts) + "</div>"


# ------------------------------------------------------------ report sections


def _header(rep: CausalRepresentation) -> str:
    sol = rep.solution
    steps = " ; ".join(f"<code>{_esc(m)}</code>" for m in sol.program)
    also = sorted(set(rep.solutions) - {rep.abstraction})
    status = (
        f"<p>abstraction <code>{_esc(rep.abstraction)}</code> · strategy "
        f"<code>{_esc(sol.strategy)}</code> · {sol.programs_tried} program(s) tried"
    )
    if also:
        status += " · also solved under " + ", ".join(f"<code>{_esc(a)}</code>" for a in also)
    lines = [f"<p>program: {steps}</p>", status + "</p>"]
    if rep.failures:
        fails = "".join(
            f"<li><code>{_esc(a)}</code>: {_esc(msg)}</li>"
            for a, msg in sorted(rep.failures.items())
        )
        lines.append(f"<p>abstractions with no fit:</p><ul>{fails}</ul>")
    return "".join(lines)


def _test_traces(rep: CausalRepresentation) -> list[Trace | None]:
    """One trace per test input (Solution.test_traces drops inapplicable inputs)."""
    sol = rep.solution
    return [
        sol.cache.run(parse_grid(test_in, rep.abstraction), sol.program)
        for test_in, _ in rep.task.test
    ]


def _demos(task: Task, rep: CausalRepresentation | None) -> str:
    rows = []
    for i, (grid_in, grid_out) in enumerate(task.train):
        rows.append(
            _row(
                grid_html(grid_in, f"train[{i}] input"),
                '<div class="arrow">→</div>',
                grid_html(grid_out, f"train[{i}] output"),
            )
        )
    predicted = _test_traces(rep) if rep is not None else [None] * len(task.test)
    for i, (test_in, test_out) in enumerate(task.test):
        parts = [
            grid_html(test_in, f"test[{i}] input"),
            '<div class="arrow">→</div>',
            grid_html(test_out, f"test[{i}] expected (held out)"),
        ]
        if rep is not None:
            trace = predicted[i]
            if trace is None:
                parts.append(
                    '<div class="arrow"><span class="fail">✗</span> program inapplicable</div>'
                )
            else:
                ok = trace.outcome.key == as_grid(test_out)
                tick = (
                    '<span class="pass">✓ matches expected</span>'
                    if ok
                    else '<span class="fail">✗ differs from expected</span>'
                )
                parts.append(
                    grid_html(trace.outcome.grid, f"test[{i}] predicted", diff_against=test_out)
                )
                parts.append(f'<div class="arrow">{tick}</div>')
        rows.append(_row(*parts))
    return "".join(rows)


def _segmentation(task: Task, chosen: str | None) -> str:
    grid_in = task.train[0][0]
    figs = []
    for name in ABSTRACTIONS:
        state = parse_grid(grid_in, name)
        suffix = " (chosen)" if name == chosen else ""
        figs.append(state_html(state, f"{name} — {len(state.objects)} object(s){suffix}"))
    note = (
        "<p>train[0] input under each registered segmentation — a recorded, "
        "revisable modelling decision.</p>"
    )
    return note + _row(*figs)


def _traces(rep: CausalRepresentation) -> str:
    parts = [
        trace_html(trace, f"train[{i}]")
        for i, trace in enumerate(rep.solution.train_traces)
    ]
    parts += [
        trace_html(trace, f"test[{i}]")
        for i, trace in enumerate(_test_traces(rep))
        if trace is not None
    ]
    return "".join(parts)


_MAX_STEPS = 6
_MAX_ALTERNATIVES = 24


def _interventions(rep: CausalRepresentation) -> str:
    """Per step: how many primitive alternatives preserve success, plus one fork."""
    sol = rep.solution
    blocks = [
        "<p>each fork replaces one program step with a primitive alternative in a "
        "twin world that shares the prefix; validity asks whether the edited "
        "program still solves the whole task.</p>"
    ]
    for step in range(min(len(sol.program), _MAX_STEPS)):
        tested = valid = 0
        breaking: CounterfactualItem | None = None
        replaceable: CounterfactualItem | None = None
        for alt in rep.primitives:
            if tested >= _MAX_ALTERNATIVES:
                break
            try:
                identified = identify(rep, Interventional(step=step, alternative=alt))
            except IdentificationError:
                continue
            tested += 1
            item = compute(identified).items[0]
            if not item.counterfactual.applicable:
                continue
            if item.metrics.validity:
                valid += 1
                replaceable = replaceable or item
            elif breaking is None:
                breaking = item
        verdict = f"{valid}/{tested} tested alternatives preserve success"
        if valid == 0:
            verdict += " — necessary within the tested set"
        pick = breaking or replaceable
        fork = (
            _item_html(pick)
            if pick is not None
            else "<p>no tested alternative even applies in that state.</p>"
        )
        blocks.append(
            f"<h3>step {step}: <code>{_esc(sol.program[step])}</code></h3>"
            f"<p>{verdict}.</p>{fork}"
        )
    return "".join(blocks)


def _pertinent(rep: CausalRepresentation) -> str:
    query = PertinentNegative(on="train[0]", max_cells=1, max_witnesses=4)
    cfs = compute(identify(rep, query))
    return "".join(_item_html(item, show_inputs=True) for item in cfs.items)


def _resegmentation(rep: CausalRepresentation) -> str:
    cards = []
    for name in ABSTRACTIONS:
        if name == rep.abstraction:
            continue
        cfs = compute(identify(rep, Representational(name)))
        cards.append(_item_html(cfs.items[0]))
    return "".join(cards)


def _contrast_section(rep: CausalRepresentation, foil: Grid, foil_on: str) -> str:
    head = _row(grid_html(foil, f"foil for {foil_on}"))
    try:
        cfs = compute(identify(rep, Contrastive(as_grid(foil), on=foil_on, k_max=2)))
    except IdentificationError as err:
        return head + f'<div class="card"><p>{_esc(err)}</p></div>'
    body = "".join(_item_html(item) for item in cfs.items[:3])
    if len(cfs.items) > 3:
        body += f"<p>(… {len(cfs.items) - 3} more)</p>"
    if cfs.responsibility is not None:
        cells = "".join(
            f"<tr><td>step {t}</td><td>{r:g}</td></tr>"
            for t, r in sorted(cfs.responsibility.items())
        )
        body += (
            '<table class="t"><tr><th>step</th><th>responsibility</th></tr>'
            + cells
            + "</table>"
        )
    return head + body


def _gate(rep: CausalRepresentation) -> str:
    report = assess(rep)
    cls = "hi" if report.confidence == "high" else "lo"
    parts = [f'<p class="{cls}">{_esc(report)}</p>']
    for i, outs in enumerate(report.predictions[:4]):
        figs = []
        for j, out in enumerate(outs):
            if out is None:
                figs.append(f'<div class="arrow">class {i} → test[{j}]: inapplicable</div>')
            else:
                figs.append(grid_html(out, f"class {i} → test[{j}]"))
        parts.append(_row(*figs))
    if len(report.predictions) > 4:
        parts.append(f"<p>(… {len(report.predictions) - 4} more classes)</p>")
    if report.probe is not None:
        parts.append(_row(grid_html(report.probe, "probe on which the classes part ways")))
    return "".join(parts)


def _refutation(rep: CausalRepresentation) -> str:
    report = refute(rep)
    mark = {True: ("PASS", "pass"), False: ("FAIL", "fail"), None: ("SKIP", "skip")}
    rows = "".join(
        f"<tr><td>{_esc(row.name)}</td>"
        f'<td class="{mark[row.passed][1]}">{mark[row.passed][0]}</td>'
        f"<td>{_esc(row.detail)}</td></tr>"
        for row in report.rows
    )
    return (
        '<table class="t"><tr><th>refuter</th><th></th><th>detail</th></tr>'
        + rows
        + "</table><p>passing is necessary, not sufficient.</p>"
    )


# ----------------------------------------------------- composition, save, show


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def report_html(
    rep: CausalRepresentation, *, foil: Grid | None = None, foil_on: str = "test[0]"
) -> str:
    """The full process page for a fitted representation."""
    task = rep.task
    sections = [
        f"<h1>dowhat — task {_esc(task.task_id)}</h1>",
        _header(rep),
        _section("Demonstrations and held-out prediction", _demos(task, rep)),
        _section("Segmentation — plural abstractions", _segmentation(task, rep.abstraction)),
        _section("Solving traces", _traces(rep)),
        _section("Interventional counterfactuals — is each step necessary?", _interventions(rep)),
        _section("Pertinent negatives — what must stay absent?", _pertinent(rep)),
        _section("Counterfactual re-segmentation", _resegmentation(rep)),
    ]
    if foil is not None:
        sections.append(
            _section("Contrastive — why not this instead?", _contrast_section(rep, foil, foil_on))
        )
    sections.append(_section("Confidence gate", _gate(rep)))
    sections.append(_section("Refutation battery", _refutation(rep)))
    return _page(f"dowhat — {task.task_id}", "".join(sections))


def unsolved_report_html(task: Task, error: str) -> str:
    body = (
        f"<h1>dowhat — task {_esc(task.task_id)}</h1>"
        f'<div class="banner"><strong>no fit:</strong> {_esc(error)}<br>'
        "This is the expected outcome for most ARC tasks: the transformation is "
        "outside the current rule vocabulary, and the solver says so rather than "
        "guessing. Below is what it still sees.</div>"
        + _section("Demonstrations", _demos(task, None))
        + _section("Segmentation — plural abstractions", _segmentation(task, None))
    )
    return _page(f"dowhat — {task.task_id}", body)


def full_report(
    task: Task, *, foil: Grid | None = None, foil_on: str = "test[0]", **model_kwargs
) -> str:
    """model() the task and render the whole pipeline; degrades when nothing fits."""
    try:
        rep = model(task, **model_kwargs)
    except UnsolvedTaskError as err:
        return unsolved_report_html(task, str(err))
    return report_html(rep, foil=foil, foil_on=foil_on)


def save_report(html_doc: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html_doc, encoding="utf-8")
    return path


def show(html_doc: str, path: str | Path | None = None) -> Path:
    """Write the page (to ``path`` or a temp file) and open it in the browser."""
    if path is None:
        fd, name = tempfile.mkstemp(suffix=".html", prefix="dowhat-")
        os.close(fd)
        path = name
    path = save_report(html_doc, path)
    webbrowser.open(path.resolve().as_uri())
    return path


# ------------------------------------------------------------------ the server

# ARC training ids that fit under default model() settings, from corpus runs
# (only a handful of the 1000 do — that gap is the research programme, not a
# bug). b230c067 fits the demonstrations but is underdetermined: its page
# shows the confidence gate abstaining.
_KNOWN_FITTING = ("25ff71a9", "42a50994", "5582e5ca", "a79310a0", "b230c067", "e0fb7511")


@dataclass
class VizApp:
    """Routes for the local viewer; the pure ``route`` keeps it testable without sockets."""

    tasks_fn: Callable[[], dict[str, Task]]
    title: str = "dowhat — ARC tasks"
    _tasks: dict[str, Task] | None = None
    _pages: dict[str, str] = field(default_factory=dict)
    _status: dict[str, str] = field(default_factory=dict)

    def _corpus(self) -> dict[str, Task]:
        if self._tasks is None:
            self._tasks = self.tasks_fn()
        return self._tasks

    def route(self, path: str) -> tuple[int, str]:
        path = unquote(urlsplit(path).path)
        try:
            if path == "/":
                return 200, self._index()
            if path.startswith("/task/"):
                return self._task_page(path[len("/task/") :])
            return 404, _page("not found", f"<h1>404</h1><p>no route {_esc(path)}</p>")
        except ModuleNotFoundError as err:
            return 500, _page(
                "missing extra",
                f"<h1>arckit is not installed</h1><p>{_esc(err)} — install the ARC "
                "extra: <code>pip install 'dowhat[arc]'</code></p>",
            )
        except Exception as err:  # a bad task must not kill the server
            return 500, _page("error", f"<h1>error</h1><p>{_esc(err)}</p>")

    def _index(self) -> str:
        ids = sorted(self._corpus())
        intro = (
            f"<p>{len(ids)} task(s) — the first click fits a task and may take a few "
            f"seconds; pages are cached. Most ARC tasks are outside the current rule "
            f"vocabulary and render an honest no-fit page.</p>"
        )
        starters = [tid for tid in _KNOWN_FITTING if tid in ids]
        if starters:
            links = ", ".join(
                f'<a href="/task/{quote(tid)}">{_esc(tid)}</a>' for tid in starters
            )
            intro += f"<p>start with a task known to fit: {links}</p>"
        items = "".join(
            f'<li><a href="/task/{quote(tid)}">{_esc(tid)}</a>{self._chip(tid)}</li>'
            for tid in ids
        )
        return _page(self.title, f"<h1>{_esc(self.title)}</h1>{intro}<ul>{items}</ul>")

    def _chip(self, tid: str) -> str:
        status = self._status.get(tid)
        if status == "fits":
            return ' <span class="pass">fits</span>'
        if status == "no fit":
            return ' <span class="fail">no fit</span>'
        return ""

    def _task_page(self, tid: str) -> tuple[int, str]:
        corpus = self._corpus()
        if tid not in corpus:
            return 404, _page("not found", f"<h1>404</h1><p>unknown task {_esc(tid)}</p>")
        if tid not in self._pages:
            try:
                rep = model(corpus[tid])
            except UnsolvedTaskError as err:
                self._pages[tid] = unsolved_report_html(corpus[tid], str(err))
                self._status[tid] = "no fit"
            else:
                self._pages[tid] = report_html(rep)
                self._status[tid] = "fits"
        return 200, self._pages[tid]


def _arc_app(dataset: str = "train") -> VizApp:
    def tasks_fn() -> dict[str, Task]:
        from .domains.arc import iter_tasks

        return {t.task_id: t for t in iter_tasks(dataset)}

    return VizApp(tasks_fn=tasks_fn, title=f"dowhat — ARC {dataset} tasks")


class _Handler(BaseHTTPRequestHandler):
    app: VizApp

    def do_GET(self):
        status, body = self.app.route(self.path)
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(port: int = 8008, dataset: str = "train", open_browser: bool = False) -> None:
    """Single-threaded local viewer (a slow first fit blocks other requests)."""
    handler = type("Handler", (_Handler,), {"app": _arc_app(dataset)})
    server = HTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"dowhat viz: serving ARC {dataset} tasks at {url} (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve dowhat process pages for the ARC corpus."
    )
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--dataset", choices=("train", "eval"), default="train")
    parser.add_argument("--open", action="store_true", help="open the index in a browser")
    args = parser.parse_args(argv)
    serve(port=args.port, dataset=args.dataset, open_browser=args.open)


if __name__ == "__main__":
    main()
