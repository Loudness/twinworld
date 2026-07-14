"""Ground-truth counterfactual benchmark (thesis Experiment 3).

In tabular ML the true minimal counterfactual is unknowable; here tasks are
GENERATED from a known latent program, so program recovery, underdetermination
rate, and counterfactual minimality can be scored exactly — the measurement
contribution the research digest flags as unoccupied. re-arc/arc-dsl latent
programs drop into the same harness later; this module supplies its own
generator so the benchmark exists today.
"""

from __future__ import annotations

import random

from .engine import ApplyCache, Program, Task
from .mechanisms import (
    All,
    ByColour,
    Delete,
    Largest,
    ObjectRule,
    RecolourTo,
    Smallest,
    TranslateBy,
)
from .representation import Grid, as_grid, parse_grid

SHAPES = (
    frozenset({(0, 0)}),
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0)}),
)
GRID_COLOURS = (2, 3, 4, 6)  # generator palette; 5/8/9 stay free as recolour targets


def generate_task(latent: Program, inputs: list[Grid], abstraction: str = "cc4") -> Task | None:
    """Build a task whose ground-truth transformation IS ``latent``.

    The last input becomes the test pair. None when the latent program is
    inapplicable to any input (caller resamples).
    """
    if len(inputs) < 3:
        return None
    cache = ApplyCache()
    pairs = []
    for grid in inputs:
        trace = cache.run(parse_grid(grid, abstraction), latent)
        if trace is None or trace.outcome.key == as_grid(grid):
            return None  # inapplicable or a no-op world: not a usable demonstration
        pairs.append((as_grid(grid), trace.outcome.key))
    return Task(train=tuple(pairs[:-1]), test=(pairs[-1],), task_id="synthetic-latent")


def random_grid(
    rng: random.Random, size: int = 7, n_objects: int = 3, colours=GRID_COLOURS
) -> Grid | None:
    """Random grid of well-separated solid objects (halo keeps every
    abstraction scheme agreeing on the segmentation)."""
    rows = [[0] * size for _ in range(size)]
    placed: set[tuple[int, int]] = set()
    for colour in rng.sample(colours, k=min(n_objects, len(colours))):
        shape = rng.choice(SHAPES)
        for _ in range(20):
            r0, c0 = rng.randrange(size - 2), rng.randrange(size - 2)
            cells = {(r0 + r, c0 + c) for r, c in shape}
            halo = {
                (r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            }
            if halo & placed:
                continue
            for r, c in cells:
                rows[r][c] = colour
            placed |= cells
            break
        else:
            return None
    return as_grid(rows)


def random_latent(rng: random.Random, colours=GRID_COLOURS) -> Program:
    """Sample a latent program from the ObjectRule language (depth 1-2)."""
    selectors = [All(), Largest(), Smallest(), *[ByColour(c) for c in colours]]
    transforms = [
        TranslateBy(0, 1),
        TranslateBy(1, 0),
        TranslateBy(0, -1),
        RecolourTo(8),
        RecolourTo(5),
        Delete(),
    ]
    depth = rng.choice((1, 1, 2))
    return tuple(ObjectRule(rng.choice(selectors), rng.choice(transforms)) for _ in range(depth))


def random_task(rng: random.Random, n_train: int = 3) -> tuple[Task, Program] | None:
    """One benchmark instance: (task, its ground-truth latent program)."""
    inputs = [random_grid(rng) for _ in range(n_train + 1)]
    if any(g is None for g in inputs):
        return None
    latent = random_latent(rng)
    task = generate_task(latent, inputs)
    if task is None:
        return None
    return task, latent
