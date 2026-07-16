"""Blocks-world domain — the non-ARC generality proof, now NATIVE.

The canonical STRIPS planning domain on the relational representation backend:
states are tower tuples (block ids stacked per column, unbounded height), and
the only mechanism is MoveBlock — which, unlike every ARC mechanism, has
*preconditions* (the block must be clear; the destination must have room) and
*context-dependent effects* (the block lands on top of whatever the
destination holds, so the same move shifts a block by a different amount in
different states — provably inexpressible as a translation rule). The core —
engine, all query types, metrics, refuters, diagnosis, viz — is used
unchanged; this module supplies only the primitive library and the task
builder.

The historical GRID serialization (columns rendered into a colour grid,
gravity down) is retained: ``task_from_towers(..., representation="grid")``
and ``MoveBlock``'s grid branch keep it runnable, and the τ-abstraction
cross-check (tests/test_blocks_tau.py) verifies that native and serialized
runs commute — Beckers & Halpern's constructive abstraction, as a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from ..backends.relational import RelationalState, state_from_towers
from ..engine import Task
from ..mechanisms import PreimageBudget
from ..representation import Grid, StateGraph, as_grid, parse_grid

Towers = Sequence[Sequence[int]]  # towers[c] = blocks in column c, bottom -> top


def build_grid(towers: Towers, height: int = 4) -> Grid:
    width = len(towers)
    rows = [[0] * width for _ in range(height)]
    for c, tower in enumerate(towers):
        if len(tower) > height:
            raise ValueError(f"tower in column {c} exceeds height {height}")
        for i, block in enumerate(tower):  # i = 0 is the bottom block
            rows[height - 1 - i][c] = block
    return as_grid(rows)


def towers_of(grid: Grid) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    result = []
    for c in range(width):
        tower = []
        for r in range(height - 1, -1, -1):
            if grid[r][c] == 0:
                break
            tower.append(grid[r][c])
        result.append(tower)
    return result


def task_from_towers(
    train: Sequence[tuple[Towers, Towers]],
    test: Sequence[tuple[Towers, Towers]],
    height: int = 4,
    representation: str = "relational",
) -> Task:
    """A planning task: demonstration instance(s) the same plan must solve.

    Native (default): tower payloads on the relational backend, columns
    unbounded. ``representation="grid"`` serializes to colour grids of
    ``height`` rows instead — the τ-abstraction escape hatch (``height`` is
    ignored on the native path).
    """
    if representation == "grid":
        return Task(
            train=tuple((build_grid(a, height), build_grid(b, height)) for a, b in train),
            test=tuple((build_grid(a, height), build_grid(b, height)) for a, b in test),
            task_id="blocks-world",
        )

    def tup(towers: Towers) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(t) for t in towers)

    return Task(
        train=tuple((tup(a), tup(b)) for a, b in train),
        test=tuple((tup(a), tup(b)) for a, b in test),
        task_id="blocks-world",
        representation="relational",
    )


@dataclass(frozen=True)
class MoveBlock:
    """Move ``block`` onto the top of ``to_column`` — with STRIPS preconditions.

    Inapplicable (None) when the block is missing or duplicated, buried under
    another block, already in the destination column, or the destination is
    full. The landing row depends on the destination stack's height: the same
    move is a different displacement in different states.
    """

    block: int
    to_column: int

    exact_preimage = True  # a just-moved block came from the top of SOME column

    def apply(self, s: StateGraph | RelationalState) -> StateGraph | RelationalState | None:
        """Dual-dispatch: native tower logic for relational states, the
        historical grid logic for grid serializations — one mechanism, two
        representations, which is what makes the τ-abstraction cross-check a
        statement about the SAME plan."""
        if isinstance(s, RelationalState):
            return self._apply_relational(s)
        return self._apply_grid(s)

    def _apply_relational(self, s: RelationalState) -> RelationalState | None:
        if not 0 <= self.to_column < s.columns:
            return None
        hits = [c for c, tower in enumerate(s.towers) if self.block in tower]
        if len(hits) != 1 or s.towers[hits[0]].count(self.block) != 1:
            return None  # the plan presumes exactly one such block
        c = hits[0]
        if c == self.to_column:
            return None
        if s.towers[c][-1] != self.block:
            return None  # buried: not clear
        if s.height is not None and len(s.towers[self.to_column]) >= s.height:
            return None  # destination column is full (never, when unbounded)
        towers = [list(t) for t in s.towers]
        towers[c].pop()
        towers[self.to_column].append(self.block)
        return state_from_towers(towers, height=s.height, abstraction=s.abstraction)

    def _apply_grid(self, s: StateGraph) -> StateGraph | None:
        grid = s.grid
        h, w = s.height, s.width
        if not 0 <= self.to_column < w:
            return None
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == self.block]
        if len(cells) != 1:
            return None  # the plan presumes exactly one such block
        (r, c) = cells[0]
        if c == self.to_column:
            return None
        if r > 0 and grid[r - 1][c] != s.background:
            return None  # buried: not clear
        landing = next(
            (rr for rr in range(h - 1, -1, -1) if grid[rr][self.to_column] == s.background),
            None,
        )
        if landing is None:
            return None  # destination column is full
        rows = [list(row) for row in grid]
        rows[r][c] = s.background
        rows[landing][self.to_column] = self.block
        return parse_grid(as_grid(rows), abstraction=s.abstraction, background=s.background)

    def preimage(
        self, s: StateGraph | RelationalState, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph | RelationalState]:
        # budget ignored: undo-to-each-column is an exact, tiny enumeration
        """Exact abduction: the block now tops ``to_column``; undo to the top
        of every other column and keep the candidates that re-apply to s."""
        for source in range(_column_count(s)):
            if source == self.to_column:
                continue
            pre = MoveBlock(self.block, source).apply(s)
            if pre is not None and self.apply(pre) == s:
                yield pre

    def touched(self, attr: str) -> frozenset | None:
        return frozenset({self.block}) if attr == "block" else None

    def __str__(self) -> str:
        return f"move block {self.block} to column {self.to_column}"


def _column_count(s: StateGraph | RelationalState) -> int:
    return s.columns if isinstance(s, RelationalState) else s.width


def candidate_moves(task: Task) -> list[MoveBlock]:
    """The domain's primitive library: every block onto every column."""
    raw = task.train[0][0]
    width = len(raw) if task.representation == "relational" else len(raw[0])
    blocks = sorted(task.colours() - {0})
    return [MoveBlock(b, c) for b in blocks for c in range(width)]
