"""CausalARC adapter (Maasch et al. 2025, arXiv:2509.03636).

ARC-style tasks sampled from fully specified structural causal models, with
labelled counterfactual demonstration pairs per train pair (hard and soft
interventions, geometric transforms). The dataset is GPL-3.0: it is fetched
at runtime as *evaluation data* and cached locally — never vendored, and the
``scm`` field (Python source of the ground-truth SCM) is kept as opaque
metadata, never executed.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..engine import Task
from ..representation import Grid, as_grid

BASE_URL = (
    "https://huggingface.co/datasets/jmaasch/causal_arc/resolve/main/"
    "static_evaluation_set/v0_09-01-25/{category}/causal_arc_{category}.json"
)
CACHE_DIR = Path("~/.cache/twinworld").expanduser()


@dataclass(frozen=True)
class CfDemo:
    """One labelled counterfactual demonstration attached to a train pair."""

    kind: str  # e.g. "do_hard (color 2 = 0)", "do_soft (geometric transform = rot90)"
    input: Grid
    output: Grid


@dataclass(frozen=True)
class CausalTask:
    task: Task
    scm_source: str  # ground-truth SCM as source text — metadata only, never executed
    cf_demos: tuple[tuple[CfDemo, ...], ...]  # one tuple per train pair


def parse_causalarc(raw: dict) -> list[CausalTask]:
    """Parse the raw CausalARC JSON dict into CausalTasks (pure, network-free)."""
    out = []
    for task_id, entry in raw.items():
        train = tuple(
            (as_grid(p["input"]), as_grid(p["output"])) for p in entry["train"]
        )
        test = tuple((as_grid(p["input"]), as_grid(p["output"])) for p in entry["test"])
        demos = tuple(
            tuple(
                CfDemo(kind, as_grid(ci), as_grid(co))
                for kind, ci, co in zip(
                    p.get("cf_types", ()), p.get("cf_inputs", ()), p.get("cf_outputs", ())
                )
            )
            for p in entry["train"]
        )
        out.append(
            CausalTask(
                Task(train=train, test=test, task_id=str(task_id)),
                entry.get("scm", ""),
                demos,
            )
        )
    return out


def load_causalarc(category: str = "counting", timeout: int = 120) -> list[CausalTask]:
    """Load a CausalARC category, downloading to a local cache on first use."""
    cache = CACHE_DIR / f"causal_arc_{category}.json"
    if not cache.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = BASE_URL.format(category=category)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            cache.write_bytes(response.read())
    return parse_causalarc(json.loads(cache.read_text()))
