"""LLM baseline over a local OpenAI-compatible endpoint (stdlib only).

The comparison subject for the thesis's "another way to measure intelligence"
claim: the LLM sees the same demonstration pairs as the symbolic solver and
must emit the test output grid. Defaults target a llama.cpp/ollama-style
server; override with DOWHAT_LLM_URL / DOWHAT_LLM_MODEL. Reasoning models are
handled by reading the final ``content`` field only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..engine import Task
from ..representation import Grid, as_grid

DEFAULT_URL = os.environ.get("DOWHAT_LLM_URL", "http://localhost:11434/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("DOWHAT_LLM_MODEL", "qwen")


def grid_text(grid: Grid) -> str:
    return "\n".join("".join(str(c) for c in row) for row in grid)


def parse_grid_text(text: str) -> Grid | None:
    """Extract the LAST contiguous block of equal-width digit rows."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        stripped = line.strip().replace(" ", "")
        if stripped and all(ch.isdigit() for ch in stripped):
            blocks[-1].append(stripped)
        elif blocks[-1]:
            blocks.append([])
    candidates = [b for b in blocks if b and len({len(row) for row in b}) == 1]
    if not candidates:
        return None
    return as_grid([[int(ch) for ch in row] for row in candidates[-1]])


def build_prompt(task: Task, cf_demos=None) -> str:
    """Demonstrations (+ optional labelled counterfactual pairs) -> test input."""
    parts = [
        "You are solving an abstract reasoning puzzle. Each grid is rows of "
        "digits 0-9. Infer the transformation from the examples and apply it "
        "to the test input.",
    ]
    for i, (grid_in, grid_out) in enumerate(task.train):
        parts.append(f"Example {i + 1} input:\n{grid_text(as_grid(grid_in))}")
        parts.append(f"Example {i + 1} output:\n{grid_text(as_grid(grid_out))}")
        if cf_demos and i < len(cf_demos):
            for demo in cf_demos[i]:
                parts.append(
                    f"Counterfactual for example {i + 1} [{demo.kind}] input:\n"
                    f"{grid_text(demo.input)}"
                )
                parts.append(f"Counterfactual output:\n{grid_text(demo.output)}")
    parts.append(f"Test input:\n{grid_text(as_grid(task.test[0][0]))}")
    parts.append("Answer with ONLY the test output grid as rows of digits.")
    return "\n\n".join(parts)


def chat(
    prompt: str,
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    max_tokens: int = 2048,
    thinking: bool = False,
) -> str | None:
    body_dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if not thinking:
        # Qwen-style reasoning models otherwise burn the whole token budget on
        # reasoning_content and never emit the answer (llama.cpp honours this)
        body_dict["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body_dict).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def llm_predict(task: Task, cf_demos=None, **chat_kwargs) -> Grid | None:
    """One gated prediction: prompt -> chat -> parsed grid (None on any failure)."""
    reply = chat(build_prompt(task, cf_demos), **chat_kwargs)
    if reply is None:
        return None
    return parse_grid_text(reply)


def reachable(url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """Fast connectivity probe (used by tests to skip when the server is down)."""
    base = url.split("/v1/")[0] + "/v1/models"
    try:
        with urllib.request.urlopen(base, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
