# AGENTS.md

Agent-facing guide for working in this repo. For the full conceptual story, read `README.md`.

## What this is

`twinworld` is a Python library for counterfactual reasoning and explanation over symbolic
state-transition traces. It exposes a DoWhy-style staged API (`model → identify → compute →
refute`), but the underlying model is a deterministic symbolic solver, so counterfactuals are
point-identified certificates rather than statistical estimates. The solving core is
non-connectionist — no neural nets or LLMs in the solving path. This is part of a PhD project;
module docstrings reference the thesis experiments they support.

## Setup & commands

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e '.[arc,dev]'   # setup
./.venv/bin/pytest                                                 # run tests
./.venv/bin/ruff check .                                           # lint
./.venv/bin/python examples/vertical_slice.py                      # end-to-end smoke run
./.venv/bin/python -m twinworld.viz                                # corpus browser at http://127.0.0.1:8008
```

- Python ≥3.11. The only core runtime dependency is `networkx`.
- Optional extras: `arc` (arckit, bundled ARC corpus), `asp` (clingo solver), `dev` (pytest, ruff).
- CI (`.github/workflows/ci.yml`): Python 3.11–3.13 matrix, installs `.[dev,asp]`, runs
  `ruff check .` then `pytest -q`. The `arc` extra is **not** installed in CI, so corpus-backed
  paths are not exercised there.

## Architecture map

- `src/twinworld/engine.py` — the causal core. A `Trace` (states + mechanism applications) *is*
  the structural causal model. `Task`, `Trace`, `Solution`, `solve`, `abduce_inputs`.
- `src/twinworld/api.py` — public staged verbs `model` / `identify` / `compute` / `refute`, plus
  the `assess` / `predict` confidence gate. Query types: `Interventional`, `Contrastive`,
  `Representational`.
- `src/twinworld/backend.py` + `src/twinworld/backends/` — the `Representation` contract between
  the core and the state substrate. Backends are registered by name and resolved lazily:
  `grid` (ARC default), `relational` (STRIPS worlds), `sequence` (letter strings), `tabular`
  (feature rows), `graph` (labelled graphs). Backend laws L1–L7 are checked by
  `conformance_battery`.
- `src/twinworld/domains/` — thin adapters mapping real problems onto a backend: `arc.py`
  (arckit), `blocks.py` (blocks world on the relational backend), `causalarc.py` (runtime-fetched
  CausalARC tasks).
- Supporting modules: `mechanisms.py` (pure state functions with `apply`/`preimage`),
  `analogy.py` (SME structure mapping), `concepts.py` (learned concept network), `copycat.py`
  (stochastic correspondence), `discriminate.py` (probe-based underdetermination diagnosis),
  `select.py` (calibrated abstention), `alternatives.py` (alternative-set ranking),
  `metrics.py`, `refute.py` (placebo battery), `asp.py` / `asp_solver.py` (optional clingo
  cross-check and program induction), `viz.py` (self-contained HTML reports, pure stdlib),
  `baselines/llm.py` (local-endpoint LLM baseline).

## Testing

- `pytest` from the repo root (`testpaths = ["tests"]`). Shared fixtures live in
  `tests/conftest.py` as compact grid-literal tasks. Test files map roughly 1:1 to modules.
- clingo-dependent tests auto-skip via `pytest.importorskip("clingo")` when the `asp` extra is
  missing.
- The live test in `tests/test_llm_baseline.py` auto-skips unless a local OpenAI-compatible
  endpoint is reachable (`TWINWORLD_LLM_URL`, default `http://localhost:11434/v1/chat/completions`;
  no API keys involved).
- When adding a backend, run it through `conformance_battery` in its tests.

## Conventions & gotchas

- src layout; the package is typed (`py.typed`).
- Keep the solving core stdlib + networkx only. No new hard dependencies — anything heavy or
  optional goes behind an extra with a lazy import.
- Lint is ruff only (`line-length = 100`, configured in `pyproject.toml`). No mypy, black, or
  pre-commit.
- `docs/` is gitignored — docs referenced by the README exist only locally / in the sdist.
- CausalARC data is GPL-3.0: it is fetched at runtime into `~/.cache/twinworld`, never vendored
  into the repo.
- Generated `examples/report_*.html` files are excluded from the sdist (`MANIFEST.in`).
