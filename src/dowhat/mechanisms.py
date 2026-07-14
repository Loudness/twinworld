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

from .representation import Obj, StateGraph, as_grid, parse_grid


@runtime_checkable
class Mechanism(Protocol):
    exact_preimage: bool

    def apply(self, s: StateGraph) -> StateGraph | None: ...

    def preimage(self, s: StateGraph) -> Iterator[StateGraph]: ...


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

    def preimage(self, s: StateGraph) -> Iterator[StateGraph]:
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

    def preimage(self, s: StateGraph) -> Iterator[StateGraph]:
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

    def preimage(self, s: StateGraph) -> Iterator[StateGraph]:
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


Selector = All | ByColour | Largest | Smallest


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

    def preimage(self, s: StateGraph) -> Iterator[StateGraph]:
        # Exact only for translations: the selector re-selects the same objects
        # (colour and size are translation-invariant), so undoing is sound.
        if not isinstance(self.transform, TranslateBy):
            return
        inverse = ObjectRule(self.selector, TranslateBy(-self.transform.dr, -self.transform.dc))
        pre = inverse.apply(s)
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
