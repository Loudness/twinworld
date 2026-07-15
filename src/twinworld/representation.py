"""Object-centric state representation (thesis Experiment 1).

A grid is parsed into a :class:`StateGraph` — a set of typed objects carrying
the thesis property ontology (location, colour, size, shape, symmetry,
rotation-invariant shape signature) — under a pluggable
:class:`AbstractionScheme`. Objects hold per-pixel colours, so multi-colour
composites are first-class. The rendered grid is the canonical identity of a
state; segmentation is an observation of it, recorded on the state and
revisable — which is what makes counterfactual re-segmentation possible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from .concepts import ConceptNet

Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]
Pixel = tuple[int, int, int]

# The domain contract: colour ids are 0..MAX_COLOURS-1 (the ARC convention).
# Core preimage/refuter enumerations range over this palette, and
# infer_background breaks frequency ties in favour of 0 — domain plugins must
# stay within these ids (blocks world does; see README "Using twinworld in
# your own domain").
MAX_COLOURS = 10  # (row, col, colour)


def as_grid(rows: Iterable[Iterable[int]]) -> Grid:
    return tuple(tuple(int(c) for c in row) for row in rows)


@dataclass(frozen=True)
class Obj:
    """One object: a set of coloured pixels. All other properties are derived."""

    oid: int
    pixels: frozenset[Pixel]

    @classmethod
    def solid(cls, oid: int, colour: int, cells: Iterable[Cell]) -> "Obj":
        return cls(oid, frozenset((r, c, colour) for r, c in cells))

    @cached_property
    def cells(self) -> frozenset[Cell]:
        return frozenset((r, c) for r, c, _ in self.pixels)

    @cached_property
    def colours(self) -> frozenset[int]:
        return frozenset(col for _, _, col in self.pixels)

    @cached_property
    def colour(self) -> int:
        """Dominant colour (most pixels; ties break to the lowest colour)."""
        counts = Counter(col for _, _, col in self.pixels)
        return min(counts, key=lambda c: (-counts[c], c))

    @cached_property
    def location(self) -> Cell:
        return (min(r for r, _ in self.cells), min(c for _, c in self.cells))

    @property
    def size(self) -> int:
        return len(self.cells)

    @cached_property
    def shape(self) -> frozenset[Cell]:
        """Cells normalized to the object's top-left corner (translation-invariant)."""
        r0, c0 = self.location
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    @cached_property
    def shape_signature(self) -> tuple[Cell, ...]:
        """Canonical shape under the dihedral group: invariant to rotation and
        reflection — two objects are 'the same shape, rotated' iff signatures match."""

        def norm(cells: list[Cell]) -> tuple[Cell, ...]:
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            return tuple(sorted((r - r0, c - c0) for r, c in cells))

        variants = []
        cur = list(self.shape)
        for _ in range(4):
            cur = [(c, -r) for r, c in cur]  # rotate 90°
            variants.append(norm(cur))
            variants.append(norm([(r, -c) for r, c in cur]))  # + mirror
        return min(variants)

    @cached_property
    def symmetries(self) -> frozenset[str]:
        """Shape self-symmetries within the bounding box (colour-blind)."""
        cells = self.shape
        h = max(r for r, _ in cells)
        w = max(c for _, c in cells)
        syms = set()
        if all((h - r, c) in cells for r, c in cells):
            syms.add("horizontal")  # top-bottom mirror
        if all((r, w - c) in cells for r, c in cells):
            syms.add("vertical")  # left-right mirror
        if all((h - r, w - c) in cells for r, c in cells):
            syms.add("rot180")
        return frozenset(syms)


class AbstractionScheme(Protocol):
    name: str

    def segment(self, grid: Grid, background: int) -> list[frozenset[Pixel]]:
        """Partition non-background pixels into object pixel-sets."""
        ...


class ConnectedComponents:
    """Objects are connected components of non-background cells.

    ``diagonal`` widens adjacency to 8 neighbours; ``colour_blind`` joins cells
    regardless of colour (ARGA's multi-colour composite abstraction).
    """

    def __init__(self, name: str, diagonal: bool = False, colour_blind: bool = False):
        self.name = name
        self._offsets = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if diagonal else [])
        )
        self._colour_blind = colour_blind

    def segment(self, grid: Grid, background: int) -> list[frozenset[Pixel]]:
        h, w = len(grid), len(grid[0])
        seen: set[Cell] = set()
        components: list[frozenset[Pixel]] = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == background or (r, c) in seen:
                    continue
                stack, comp = [(r, c)], set()
                seen.add((r, c))
                while stack:
                    cr, cc = stack.pop()
                    comp.add((cr, cc, grid[cr][cc]))
                    for dr, dc in self._offsets:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                            continue
                        if grid[nr][nc] == background:
                            continue
                        if not self._colour_blind and grid[nr][nc] != grid[cr][cc]:
                            continue
                        seen.add((nr, nc))
                        stack.append((nr, nc))
                components.append(frozenset(comp))
        return components


ABSTRACTIONS: dict[str, AbstractionScheme] = {
    "cc4": ConnectedComponents("cc4"),
    "cc8": ConnectedComponents("cc8", diagonal=True),
    "mcc": ConnectedComponents("mcc", colour_blind=True),
}


def infer_background(grid: Grid) -> int:
    """Most frequent colour; ties broken in favour of 0 (ARC convention)."""
    counts: dict[int, int] = {}
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=lambda c: (counts[c], c == 0))


@dataclass(frozen=True)
class StateGraph:
    """A state: dimensions, background colour, objects, and the abstraction used.

    Equality and hashing go through the rendered grid — two states are the same
    state iff they look the same, regardless of how they were segmented.
    """

    height: int
    width: int
    background: int
    objects: tuple[Obj, ...]
    abstraction: str

    @cached_property
    def grid(self) -> Grid:
        rows = [[self.background] * self.width for _ in range(self.height)]
        for obj in self.objects:
            for r, c, colour in obj.pixels:
                rows[r][c] = colour
        return as_grid(rows)

    @property
    def key(self) -> Grid:
        return self.grid

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateGraph) and self.grid == other.grid

    def __hash__(self) -> int:
        return hash(self.grid)

    def colours(self) -> set[int]:
        return {col for o in self.objects for col in o.colours}

    def to_networkx(self):
        """Object-relation view for interop; nodes carry the property ontology."""
        import networkx as nx

        g = nx.MultiDiGraph(abstraction=self.abstraction, background=self.background)
        for o in self.objects:
            g.add_node(
                o.oid,
                colour=o.colour,
                colours=o.colours,
                location=o.location,
                size=o.size,
                shape=o.shape,
                shape_signature=o.shape_signature,
                symmetries=o.symmetries,
            )
        for a in self.objects:
            for b in self.objects:
                if a.oid < b.oid:
                    if a.colour == b.colour:
                        g.add_edge(a.oid, b.oid, relation="same_colour")
                    if a.shape == b.shape:
                        g.add_edge(a.oid, b.oid, relation="same_shape")
        return g


def parse_grid(grid: Grid, abstraction: str = "cc4", background: int | None = None) -> StateGraph:
    grid = as_grid(grid)
    scheme = ABSTRACTIONS[abstraction]
    bg = infer_background(grid) if background is None else background
    pixel_sets = sorted(scheme.segment(grid, bg), key=lambda p: min((r, c) for r, c, _ in p))
    objects = tuple(Obj(i, pixels) for i, pixels in enumerate(pixel_sets))
    return StateGraph(len(grid), len(grid[0]), bg, objects, abstraction)


def attribute_score(x: Obj, y: Obj, concepts: ConceptNet | None = None) -> float:
    """Property-level similarity of two objects: shared shape, colour,
    location, and cell overlap (IoU). Used for identity matching and as the
    local-match score inside structure mapping. ``concepts`` swaps the
    hand-coded weights for learned ones (None keeps the historical 4/2/1/3)."""
    if concepts is None:
        w_shape, w_colour, w_location, w_iou = 4.0, 2.0, 1.0, 3.0
    else:
        w_shape, w_colour, w_location, w_iou = (
            concepts.shape, concepts.colour, concepts.location, concepts.iou,
        )
    s = 0.0
    s += w_shape if x.shape == y.shape else 0
    s += w_colour if x.colour == y.colour else 0
    s += w_location if x.location == y.location else 0
    s += w_iou * len(x.cells & y.cells) / len(x.cells | y.cells)
    return s


def match_objects(a: StateGraph, b: StateGraph) -> list[tuple[Obj | None, Obj | None]]:
    """Greedy persistent-identity matching between two states.

    Pairs are scored by :func:`attribute_score`; each object is used at most
    once. Unmatched objects pair with None (appeared / disappeared).
    """
    candidates = sorted(
        ((attribute_score(x, y), x.oid, y.oid, x, y) for x in a.objects for y in b.objects),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[Obj | None, Obj | None]] = []
    for s, _, _, x, y in candidates:
        if s == 0 or x.oid in used_a or y.oid in used_b:
            continue
        pairs.append((x, y))
        used_a.add(x.oid)
        used_b.add(y.oid)
    pairs.extend((x, None) for x in a.objects if x.oid not in used_a)
    pairs.extend((None, y) for y in b.objects if y.oid not in used_b)
    return pairs
