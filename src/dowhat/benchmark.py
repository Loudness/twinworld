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
    _HYPOTHESIS_SHAPES,
    All,
    ByColour,
    Delete,
    Largest,
    Not,
    ObjectRule,
    RecolourTo,
    Smallest,
    TranslateBy,
)
from .representation import Grid, StateGraph, as_grid, parse_grid

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


# --------------------------------------- controlled underdetermination


_BIG_SHAPE = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})  # 2x2: a unique largest
_SMALL_SHAPES = (
    frozenset({(0, 0)}),
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0)}),
)


def random_ambiguous_task(
    rng: random.Random,
    treacherous: bool = True,
    n_train: int = 3,
    latent_bias: float = 0.5,
) -> tuple[Task, Program] | None:
    """A controlled failure-class-3 instance: on every train input the largest
    object IS the colour-c object, so the size reading and the colour reading
    fit identically; the held-out test input makes them diverge when
    ``treacherous`` (a different colour carries the big shape) and keeps them
    agreeing otherwise. The latent is the COLOUR reading with probability
    ``latent_bias``, else the SIZE reading — known ground truth for scoring
    selection policies."""
    colours = list(GRID_COLOURS)
    rng.shuffle(colours)
    c, others = colours[0], colours[1:3]
    inputs = []
    for k in range(n_train + 1):
        divergent = treacherous and k == n_train
        big_colour = others[0] if divergent else c
        # one shared small shape: equal sizes keep third readings (Not(Smallest))
        # coincident with Largest, so the collision stays a clean two-way split
        small = rng.choice(_SMALL_SHAPES)
        spec = [
            (big_colour, _BIG_SHAPE),
            (c if divergent else others[0], small),
            (others[1], small),
        ]
        grid = _place_objects(rng, 7, spec)
        if grid is None:
            return None
        state = parse_grid(grid)
        top = max(o.size for o in state.objects)
        largest = [o for o in state.objects if o.size == top]
        if len(largest) != 1 or largest[0].colour != big_colour:
            return None  # collision (or divergence) failed; caller resamples
        inputs.append(grid)
    selector = ByColour(c) if rng.random() < latent_bias else Largest()
    transform = rng.choice([RecolourTo(8), RecolourTo(5), Delete(), TranslateBy(0, 1)])
    latent: Program = (ObjectRule(selector, transform),)
    task = generate_task(latent, inputs)
    if task is None:
        return None
    return task, latent


# ------------------------------------------------- abduction ground truth


def _place_objects(
    rng: random.Random, size: int, spec: list[tuple[int, frozenset]]
) -> Grid | None:
    """Place the given (colour, shape) objects halo-separated; None on failure.

    Separate from random_grid on purpose: sharing would change random_grid's
    rng consumption and silently reseed every existing benchmark."""
    rows = [[0] * size for _ in range(size)]
    placed: set[tuple[int, int]] = set()
    for colour, shape in spec:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        for _ in range(30):
            r0 = rng.randrange(size - height + 1)
            c0 = rng.randrange(size - width + 1)
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


def random_delete_instance(
    rng: random.Random,
    size: int = 10,
    n_deleted: int = 1,
    n_palette: int = 2,
    family: str = "bycolour",
) -> tuple[ObjectRule, StateGraph, StateGraph] | None:
    """A ground-truth abduction instance: (rule, pre, observed) with
    ``rule.apply(pre) == observed`` and every deleted shape drawn from the
    preimage hypothesis catalogue (so the true origin is recoverable in
    principle — the question examples/abduction_scaling.py measures is at
    what rank and cost)."""
    palette = list(GRID_COLOURS[: max(1, n_palette)])
    shapes = [frozenset(sh) for sh in _HYPOTHESIS_SHAPES]
    dot = frozenset({(0, 0)})
    bar = frozenset({(0, 0), (0, 1), (0, 2)})
    if family == "bycolour":
        target = palette[0]
        rule = ObjectRule(ByColour(target), Delete())
        deleted = [(target, rng.choice(shapes)) for _ in range(n_deleted)]
        survivors = [(c, rng.choice(shapes)) for c in palette[1:]]
    elif family == "smallest":
        rule = ObjectRule(Smallest(), Delete())
        deleted = [(rng.choice(palette), dot) for _ in range(n_deleted)]
        survivors = [(c, bar) for c in palette]  # strictly larger than any dot
    elif family == "not_bycolour":
        if len(palette) < 2:
            return None
        keep = palette[0]
        rule = ObjectRule(Not(ByColour(keep)), Delete())
        deleted = [(rng.choice(palette[1:]), rng.choice(shapes)) for _ in range(n_deleted)]
        survivors = [(keep, rng.choice(shapes))]
    else:
        raise ValueError(f"unknown family {family!r}")
    grid = _place_objects(rng, size, survivors + deleted)
    if grid is None:
        return None
    pre = parse_grid(grid)
    observed = rule.apply(pre)
    if observed is None or observed == pre:
        return None
    return rule, pre, observed
