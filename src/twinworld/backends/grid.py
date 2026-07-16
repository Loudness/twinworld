"""The grid representation backend — ARC's substrate, registered as ``"grid"``.

A thin object gathering the existing grid functions behind the
:class:`~twinworld.backend.Representation` contract. Behaviour is byte-identical
to the pre-backend code: every method delegates to the module that always
owned it.
"""

from __future__ import annotations

from typing import Callable, Collection, Iterator, Mapping, Sequence

from ..analogy import GRID_TRANSFORM_FAMILIES
from ..backend import Addition, register
from ..mechanisms import _rebuild, candidate_primitives
from ..representation import (
    ABSTRACTIONS,
    MAX_COLOURS,
    Cell,
    Grid,
    Obj,
    StateGraph,
    as_grid,
    infer_background,
    match_objects,
    parse_grid,
)

MAX_PROBE_OBJECTS = 6  # per train input; keeps the probe set small and readable

# hypothesis-space footprints for pertinent-negative additions, by cell count
_ADDITION_SHAPES: dict[int, tuple[tuple[Cell, ...], ...]] = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)), ((0, 0), (1, 0))),
    3: (((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0))),
}


def _paint(grid: Grid, obj: Obj, colour: int) -> Grid:
    rows = [list(r) for r in grid]
    for r, c in obj.cells:
        rows[r][c] = colour
    return as_grid(rows)


def _nudge(grid: Grid, obj: Obj, dr: int, dc: int, background: int) -> Grid | None:
    h, w = len(grid), len(grid[0])
    targets = {(r + dr, c + dc) for r, c in obj.cells}
    if not all(0 <= r < h and 0 <= c < w for r, c in targets):
        return None
    if any((r, c) not in obj.cells and grid[r][c] != background for r, c in targets):
        return None
    rows = [list(r) for r in grid]
    for r, c in obj.cells:
        rows[r][c] = background
    for r, c, colour in obj.pixels:
        rows[r + dr][c + dc] = colour
    return as_grid(rows)


def _recolour_cells(state: StateGraph, obj: Obj | None, colour: int) -> Grid:
    rows = [list(row) for row in state.grid]
    if obj is not None:
        for r, c in obj.cells:
            rows[r][c] = colour
    return as_grid(rows)


def _find(state: StateGraph, obj: Obj) -> Obj | None:
    """Locate the same (untouched) object in another state by cells+colour."""
    for candidate in state.objects:
        if candidate.cells == obj.cells and candidate.colour == obj.colour:
            return candidate
    return None


class GridRepresentation:
    name = "grid"
    default_abstractions = ("cc4", "cc8", "mcc")
    transform_families = GRID_TRANSFORM_FAMILIES

    @property
    def abstractions(self) -> Mapping[str, object]:
        return ABSTRACTIONS

    def parse(
        self, raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> StateGraph:
        background = (context or {}).get("background")
        return parse_grid(raw, abstraction or self.default_abstractions[0], background=background)

    def canon(self, raw) -> Grid:
        return as_grid(raw)

    def raw_of(self, state: StateGraph) -> Grid:
        return state.grid

    def frame(self, state: StateGraph) -> tuple[int, int]:
        return (state.height, state.width)

    def rebuild(self, template: StateGraph, entities: Sequence[Obj]) -> StateGraph | None:
        return _rebuild(template, list(entities))

    def candidate_primitives(self, task) -> list:
        return candidate_primitives(
            task.colours(), background=infer_background(as_grid(task.train[0][0]))
        )

    def task_values(self, task) -> frozenset[int]:
        return frozenset(task.colours())

    def attr_domain(self, attr: str) -> tuple[int, ...] | None:
        return tuple(range(MAX_COLOURS)) if attr == "colour" else None

    def fresh_value(self, attr: str, used: Collection) -> int | None:
        if attr != "colour":
            return None
        return next((c for c in range(MAX_COLOURS - 1, -1, -1) if c not in used), None)

    def relations(self, state: StateGraph) -> set[tuple[str, int, int]]:
        from ..analogy import relations

        return relations(state)

    def overlap(self, a: Obj, b: Obj) -> float:
        return len(a.cells & b.cells) / len(a.cells | b.cells)

    # ------------------------------------------------ optional capabilities

    def probe_perturbations(self, state: StateGraph, used: Collection) -> Iterator[Grid]:
        """Object-level probe variants of one input, in the historical order:
        per object — delete, recolour to a task-unused colour, nudge right, nudge down."""
        base = state.grid
        fresh = self.fresh_value("colour", used)
        fresh = 9 if fresh is None else fresh
        for o in state.objects[:MAX_PROBE_OBJECTS]:
            for candidate in (
                _paint(base, o, state.background),  # delete it
                _paint(base, o, fresh),  # recolour to unused colour
                _nudge(base, o, 0, 1, state.background),  # nudge right
                _nudge(base, o, 1, 0, state.background),  # nudge down
            ):
                if candidate is not None:
                    yield candidate

    def addition_values(self, state: StateGraph, task) -> list[int]:
        """Colours a pertinent-negative addition may take: the task palette
        plus one task-unused colour, when any exists."""
        palette = sorted(task.colours() - {state.background})
        fresh = next(
            (c for c in range(MAX_COLOURS - 1, -1, -1) if c not in task.colours()), None
        )
        return palette + ([fresh] if fresh is not None else [])

    def addition_catalogue(
        self, state: StateGraph, max_size: int, separated: bool, values: Sequence[int]
    ) -> Iterator[Addition]:
        """The bounded pertinent-negative hypothesis space, streamed in the
        historical scan order: footprint size, then anchor cell, then shape,
        then colour. ``group`` is the anchor (one witness per anchor)."""
        grid, background = state.grid, state.background
        occupied = {cell for o in state.objects for cell in o.cells}
        excluded = occupied
        if separated:
            excluded = {
                (r + dr, c + dc)
                for r, c in occupied
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
        free = {
            (r, c)
            for r in range(state.height)
            for c in range(state.width)
            if (r, c) not in excluded and grid[r][c] == background
        }
        anchors = sorted(free)[:80]
        for size in range(1, min(max_size, max(_ADDITION_SHAPES)) + 1):
            for anchor_r, anchor_c in anchors:
                for shape in _ADDITION_SHAPES[size]:
                    cells = [(anchor_r + r, anchor_c + c) for r, c in shape]
                    if not all(cell in free for cell in cells):
                        continue
                    for colour in values:
                        rows = [list(row) for row in grid]
                        for r, c in cells:
                            rows[r][c] = colour
                        yield Addition(
                            raw=as_grid(rows),
                            phrase=f"a colour-{colour} object occupied {cells}",
                            size=size,
                            group=(anchor_r, anchor_c),
                        )

    def placebo_edit(
        self, state: StateGraph, spectator: Obj, forbidden: Collection[int]
    ) -> tuple[Grid, int, Callable[[StateGraph], Grid]] | None:
        """Recolour one program-irrelevant object; returns the edited input,
        the placebo colour, and a function computing the expected pass-through
        outcome key from the factual outcome."""
        free = [
            c
            for c in range(MAX_COLOURS)
            if c not in forbidden and c != spectator.colour and c != state.background
        ]
        if not free:
            return None
        colour = free[0]

        def expect(outcome: StateGraph) -> Grid:
            return _recolour_cells(outcome, _find(outcome, spectator), colour)

        return _recolour_cells(state, spectator, colour), colour, expect

    def distance(self, a: StateGraph, b: StateGraph) -> float:
        """Approximate object-graph edit distance: matched objects contribute
        the number of differing properties (colour, shape, location);
        unmatched objects cost 2 each. Greedy matching — an upper bound."""
        cost = 0.0
        for x, y in match_objects(a, b):
            if x is None or y is None:
                cost += 2.0
                continue
            cost += (x.colour != y.colour) + (x.shape != y.shape) + (x.location != y.location)
        return cost


GRID = register(GridRepresentation())
