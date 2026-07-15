"""Transformation mechanisms (thesis Experiment 2 substrate).

A :class:`Mechanism` is a pure function on states with an explicit inverse-set:
``apply`` returns the successor state or ``None`` when inapplicable, and
``preimage`` lazily enumerates predecessor states — the exact-abduction hook
that statistical counterfactual engines lack for discrete, non-invertible
operations. ``exact_preimage`` declares whether the enumeration is exhaustive.

Applied states are canonicalized by render → re-parse under the state's own
abstraction, so the grid remains the ground truth of what a mechanism did.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Protocol, runtime_checkable

from .representation import MAX_COLOURS, Obj, StateGraph, as_grid, parse_grid


@dataclass(frozen=True)
class PreimageBudget:
    """Caps for non-injective preimage enumeration.

    The enumeration is catalogue-bounded, not exhaustive; the budget makes the
    bound an explicit, experimentable quantity (examples/abduction_scaling.py).
    Measured finding: capping SINGLE-object hypotheses at ``anchors`` free
    cells made most true origins hard-unreachable on large grids (the cap has
    a top-left bias), so singles now stream over every free cell and
    ``anchors`` bounds only the pool that two-object PAIRS are drawn from;
    ``cap_singles=True`` restores the historical behaviour for comparison."""

    anchors: int = 40  # free cells anchoring the PAIR hypothesis pool (Delete)
    pairs: int = 200  # two-object hypotheses examined (Delete)
    recolour_objects: int = 6  # uniformly-target objects considered (RecolourTo)
    cap_singles: bool = False  # True = pre-fix behaviour: singles also capped


DEFAULT_PREIMAGE_BUDGET = PreimageBudget()


@runtime_checkable
class Mechanism(Protocol):
    exact_preimage: bool

    def apply(self, s: StateGraph) -> StateGraph | None: ...

    def preimage(
        self, s: StateGraph, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph]: ...


# hypothesis-space footprints for abduction through deletion
_HYPOTHESIS_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)


def _rebuild(s: StateGraph, objects: list[Obj]) -> StateGraph | None:
    """Render transformed objects and re-parse canonically; None on collisions
    or out-of-bounds pixels."""
    painted: dict[tuple[int, int], int] = {}
    for o in objects:
        for r, c, colour in o.pixels:
            if not (0 <= r < s.height and 0 <= c < s.width):
                return None
            if (r, c) in painted:
                return None
            painted[(r, c)] = colour
    rows = [[s.background] * s.width for _ in range(s.height)]
    for (r, c), colour in painted.items():
        rows[r][c] = colour
    return parse_grid(as_grid(rows), abstraction=s.abstraction, background=s.background)


@dataclass(frozen=True)
class Identity:
    exact_preimage = True

    def apply(self, s: StateGraph) -> StateGraph:
        return s

    def preimage(
        self, s: StateGraph, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph]:
        yield s

    def __str__(self) -> str:
        return "identity"


@dataclass(frozen=True)
class Recolor:
    """Substitute colour ``src`` with ``dst`` everywhere on the grid.

    Colour substitution is representation-independent: under a single-colour
    abstraction it recolours whole objects; under a multi-colour composite it
    recolours the matching pixels inside composites.
    """

    src: int
    dst: int

    # Preimages are enumerated at object granularity (uniformly dst-coloured
    # objects flipped back); cell-granular preimages are not enumerated.
    exact_preimage = False

    def apply(self, s: StateGraph) -> StateGraph | None:
        if self.src == self.dst or self.src == s.background:
            return None
        if all(self.src != v for row in s.grid for v in row):
            return None  # no-op applications are inapplicable, not silent identities
        rows = [[self.dst if v == self.src else v for v in row] for row in s.grid]
        return parse_grid(as_grid(rows), abstraction=s.abstraction, background=s.background)

    def preimage(
        self, s: StateGraph, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph]:
        # budget accepted for protocol uniformity; the exact subset enumeration
        # here is bounded by the dst-uniform object count, not by a cap
        if self.src == self.dst or any(self.src in o.colours for o in s.objects):
            return  # apply leaves no src-coloured pixel behind
        uniform_dst = [o for o in s.objects if o.colours == frozenset({self.dst})]
        others = [o for o in s.objects if o.colours != frozenset({self.dst})]
        for k in range(len(uniform_dst) + 1):
            for flipped in combinations(uniform_dst, k):
                back = [Obj.solid(o.oid, self.src, o.cells) for o in flipped]
                kept = [o for o in uniform_dst if o not in flipped]
                pre = _rebuild(s, others + kept + back)
                if pre is not None and self.apply(pre) == s:
                    yield pre

    def __str__(self) -> str:
        return f"recolor({self.src}->{self.dst})"


@dataclass(frozen=True)
class Translate:
    """Move every object of dominant colour ``colour`` (or all objects if None)
    by (dr, dc).

    Inapplicable when a moved pixel would leave the grid or land on another object.
    """

    dr: int
    dc: int
    colour: int | None = None

    exact_preimage = True

    def _moves(self, o: Obj) -> bool:
        return self.colour is None or o.colour == self.colour

    def apply(self, s: StateGraph) -> StateGraph | None:
        if self.dr == 0 and self.dc == 0:
            return None
        if not any(self._moves(o) for o in s.objects):
            return None  # no-op applications are inapplicable, not silent identities
        objects = [
            Obj(o.oid, frozenset((r + self.dr, c + self.dc, col) for r, c, col in o.pixels))
            if self._moves(o)
            else o
            for o in s.objects
        ]
        return _rebuild(s, objects)

    def preimage(
        self, s: StateGraph, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph]:
        inverse = Translate(-self.dr, -self.dc, self.colour)
        pre = inverse.apply(s)
        if pre is not None and self.apply(pre) == s:
            yield pre

    def __str__(self) -> str:
        target = "all" if self.colour is None else f"colour {self.colour}"
        return f"translate({target} by {self.dr},{self.dc})"


# --------------------------------------------------------------------------
# Object-level rules (thesis Experiment 2): a selector picks objects, a
# transform rewrites each. Rules are ordinary Mechanisms, so the engine,
# counterfactuals, metrics and refuters work on them unchanged.


@dataclass(frozen=True)
class All:
    def select(self, objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
        return tuple(objects)

    def __str__(self) -> str:
        return "all objects"


@dataclass(frozen=True)
class ByColour:
    colour: int

    def select(self, objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
        return tuple(o for o in objects if o.colour == self.colour)

    def __str__(self) -> str:
        return f"colour-{self.colour} objects"


@dataclass(frozen=True)
class Largest:
    def select(self, objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
        if not objects:
            return ()
        top = max(o.size for o in objects)
        return tuple(o for o in objects if o.size == top)

    def __str__(self) -> str:
        return "largest object(s)"


@dataclass(frozen=True)
class Smallest:
    def select(self, objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
        if not objects:
            return ()
        low = min(o.size for o in objects)
        return tuple(o for o in objects if o.size == low)

    def __str__(self) -> str:
        return "smallest object(s)"


@dataclass(frozen=True)
class Not:
    """Selector negation (thesis Experiment 4): the complement of ``inner``."""

    inner: "Selector"

    def select(self, objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
        excluded = set(self.inner.select(objects))
        return tuple(o for o in objects if o not in excluded)

    def __str__(self) -> str:
        return f"objects other than {self.inner}"


Selector = All | ByColour | Largest | Smallest | Not


@dataclass(frozen=True)
class TranslateBy:
    dr: int
    dc: int

    def transform(self, o: Obj) -> Obj | None:
        return Obj(o.oid, frozenset((r + self.dr, c + self.dc, col) for r, c, col in o.pixels))

    def __str__(self) -> str:
        return f"move %s by ({self.dr},{self.dc})"


@dataclass(frozen=True)
class RecolourTo:
    colour: int

    def transform(self, o: Obj) -> Obj | None:
        return Obj(o.oid, frozenset((r, c, self.colour) for r, c, _ in o.pixels))

    def __str__(self) -> str:
        return f"recolour %s to {self.colour}"


@dataclass(frozen=True)
class Delete:
    def transform(self, o: Obj) -> Obj | None:
        return None

    def __str__(self) -> str:
        return "delete %s"


ObjectTransform = TranslateBy | RecolourTo | Delete


@dataclass(frozen=True)
class ObjectRule:
    """Apply ``transform`` to every object picked by ``selector``."""

    selector: Selector
    transform: ObjectTransform

    @property
    def exact_preimage(self) -> bool:
        return isinstance(self.transform, TranslateBy)

    def apply(self, s: StateGraph) -> StateGraph | None:
        selected = set(self.selector.select(s.objects))
        if not selected:
            return None
        objects: list[Obj] = []
        for o in s.objects:
            if o in selected:
                t = self.transform.transform(o)
                if t is not None:
                    objects.append(t)
            else:
                objects.append(o)
        out = _rebuild(s, objects)
        if out is None or out == s:
            return None  # no-op applications are inapplicable, not silent identities
        return out

    def preimage(
        self, s: StateGraph, budget: PreimageBudget | None = None
    ) -> Iterator[StateGraph]:
        budget = budget or DEFAULT_PREIMAGE_BUDGET
        # Translations: exact — the selector re-selects the same objects
        # (colour and size are translation-invariant), so undoing is sound.
        if isinstance(self.transform, TranslateBy):
            inverse = ObjectRule(
                self.selector, TranslateBy(-self.transform.dr, -self.transform.dc)
            )
            pre = inverse.apply(s)
            if pre is not None and self.apply(pre) == s:
                yield pre
            return
        # RecolourTo: bounded abduction — flip subsets of uniformly-target
        # objects back to candidate original colours (a ByColour selector pins
        # the colour; otherwise all colours are candidates, one shared colour
        # per flip set), each candidate verified by re-application.
        if isinstance(self.transform, RecolourTo):
            target = self.transform.colour
            uniform = [o for o in s.objects if o.colours == frozenset({target})]
            uniform = uniform[: budget.recolour_objects]
            if isinstance(self.selector, ByColour):
                originals = [self.selector.colour]
            elif isinstance(self.selector, Not) and isinstance(self.selector.inner, ByColour):
                originals = [
                    c for c in range(MAX_COLOURS) if c not in (self.selector.inner.colour, target)
                ]
            else:
                originals = [c for c in range(MAX_COLOURS) if c != target]
            for k in range(1, len(uniform) + 1):
                for flipped in combinations(uniform, k):
                    kept = [o for o in s.objects if o not in flipped]
                    for colour in originals:
                        back = [Obj.solid(o.oid, colour, o.cells) for o in flipped]
                        pre = _rebuild(s, kept + back)
                        if pre is not None and self.apply(pre) == s:
                            yield pre
            return
        # Delete: abduction through deletion needs a hypothesis space over
        # what was deleted. The catalogue below is bounded (small shapes,
        # selector-pinned colours, separated placements) — but apply(pre) == s
        # is a COMPLETE verifier, so every selector-consistency constraint
        # (the deleted objects are exactly what the selector picks; no
        # survivor is selectable) comes for free from re-application.
        if isinstance(self.transform, Delete):
            yield from self._deleted_hypotheses(s, budget)
        return

    def _deleted_hypotheses(
        self, s: StateGraph, budget: PreimageBudget = DEFAULT_PREIMAGE_BUDGET
    ) -> Iterator[StateGraph]:
        if isinstance(self.selector, ByColour):
            colours = [self.selector.colour]
        elif isinstance(self.selector, Not) and isinstance(self.selector.inner, ByColour):
            colours = [
                c for c in range(1, MAX_COLOURS)
                if c not in (self.selector.inner.colour, s.background)
            ]
        else:
            # bounded to the surviving palette (Occam); the verifier would
            # accept other colours too, but the catalogue must stop somewhere
            colours = sorted({o.colour for o in s.objects}) or [1]
        occupied = {cell for o in s.objects for cell in o.cells}
        halo = {
            (r + dr, c + dc)
            for r, c in occupied
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        }
        grid = s.grid
        free = {
            (r, c)
            for r in range(s.height)
            for c in range(s.width)
            if (r, c) not in halo and grid[r][c] == s.background
        }
        free_sorted = sorted(free)

        def hypotheses(anchors: list) -> list[frozenset]:
            out = []
            for anchor_r, anchor_c in anchors:
                for shape in _HYPOTHESIS_SHAPES:
                    cells = [(anchor_r + r, anchor_c + c) for r, c in shape]
                    if all(cell in free for cell in cells):
                        out.append(frozenset(cells))
            # Occam ordering: smallest hypothesised objects first
            out.sort(key=lambda cells: (len(cells), sorted(cells)))
            return out

        # pairs draw from a capped pool; singles cover every free cell unless
        # the historical cap is explicitly requested (see PreimageBudget)
        pair_pool = hypotheses(free_sorted[: budget.anchors])
        singles = pair_pool if budget.cap_singles else hypotheses(free_sorted)
        next_oid = max((o.oid for o in s.objects), default=-1) + 1

        def candidate(cell_groups: tuple[frozenset, ...], colour: int) -> StateGraph | None:
            added = [
                Obj.solid(next_oid + i, colour, cells)
                for i, cells in enumerate(cell_groups)
            ]
            return _rebuild(s, list(s.objects) + added)

        # smaller hypotheses first: one deleted object, then two
        for cells in singles:
            for colour in colours:
                pre = candidate((cells,), colour)
                if pre is not None and self.apply(pre) == s:
                    yield pre
        seen_pairs = 0
        for i, a in enumerate(pair_pool):
            for b in pair_pool[i + 1 :]:
                if seen_pairs >= budget.pairs:
                    return
                near = {
                    (r + dr, c + dc) for r, c in a for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                }
                if near & b:
                    continue  # keep the two hypothesised objects separated
                seen_pairs += 1
                for colour in colours:
                    pre = candidate((a, b), colour)
                    if pre is not None and self.apply(pre) == s:
                        yield pre

    def __str__(self) -> str:
        return str(self.transform) % str(self.selector)


def candidate_primitives(
    colours: set[int], background: int, max_shift: int = 3
) -> list[Mechanism]:
    """Enumerate a task-scoped primitive vocabulary (the slice's 'auto-assign' step)."""
    palette = sorted(colours - {background})
    prims: list[Mechanism] = []
    prims += [Recolor(a, b) for a in palette for b in sorted(colours | {background}) if a != b]
    shifts = [d for d in range(-max_shift, max_shift + 1) if d != 0]
    targets: list[int | None] = [None, *palette]  # None: move all objects, any colour
    prims += [Translate(dr, 0, col) for dr in shifts for col in targets]
    prims += [Translate(0, dc, col) for dc in shifts for col in targets]
    return prims
