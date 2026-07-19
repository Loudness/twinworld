"""The relational representation backend — STRIPS-style column worlds,
registered as ``"relational"``.

States are stacks of uniquely-identified blocks: the semantic content is the
ground-atom set (``at(block, column, level)`` with derived ``on``/``clear``),
canonically serialized as tower tuples. The key is the tower serialization
plus the height bound, so identity is abstraction-independent (law L2), and
``height=None`` means the world is UNBOUNDED — a state the grid serialization
cannot express (docs/beyond-grids.md §5.1). Two segmentation schemes make
counterfactual re-segmentation meaningful here too: ``"consts"`` (one entity
per block) and ``"towers"`` (one entity per occupied column).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Collection, Hashable, Iterator, Mapping, Sequence

from ..backend import Addition, register

Towers = tuple[tuple[int, ...], ...]  # towers[c] = blocks in column c, bottom -> top
Atom = tuple  # ("at", block, column, level)


@dataclass(frozen=True)
class RelEntity:
    """One relational entity: a block (``"consts"``) or a column (``"towers"``)."""

    oid: int
    extent: tuple
    size: int
    _attrs: tuple[tuple[str, Hashable], ...]

    @cached_property
    def attributes(self) -> dict[str, Hashable]:
        return dict(self._attrs)


@dataclass(frozen=True, eq=False)
class RelationalState:
    """A column world. Equality and hashing go through ``key`` — the tower
    serialization plus the height bound — regardless of segmentation."""

    towers: Towers
    height: int | None = None  # None: unbounded columns (grid-inexpressible)
    abstraction: str = "consts"

    representation = "relational"

    @property
    def columns(self) -> int:
        return len(self.towers)

    @cached_property
    def universe(self) -> frozenset[int]:
        return frozenset(b for tower in self.towers for b in tower)

    @cached_property
    def atoms(self) -> frozenset[Atom]:
        return frozenset(
            ("at", block, c, level)
            for c, tower in enumerate(self.towers)
            for level, block in enumerate(tower)
        )

    @property
    def key(self) -> tuple:
        return ("relational", self.towers, self.height)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RelationalState) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    @cached_property
    def objects(self) -> tuple[RelEntity, ...]:
        if self.abstraction == "towers":
            return tuple(
                RelEntity(
                    oid=c,
                    extent=(c, tower),
                    size=len(tower),
                    _attrs=(
                        ("column", c),
                        ("height", len(tower)),
                        ("blocks", tower),
                        ("top", tower[-1]),
                    ),
                )
                for c, tower in enumerate(self.towers)
                if tower
            )
        entities = []
        for c, tower in enumerate(self.towers):
            for level, block in enumerate(tower):
                entities.append(
                    RelEntity(
                        oid=block,
                        extent=("at", block, c, level),
                        size=1,
                        _attrs=(
                            ("block", block),
                            ("column", c),
                            ("level", level),
                            ("on", tower[level - 1] if level > 0 else None),
                            ("clear", level == len(tower) - 1),
                        ),
                    )
                )
        return tuple(sorted(entities, key=lambda e: e.oid))


def state_from_towers(
    towers: Sequence[Sequence[int]],
    height: int | None = None,
    abstraction: str = "consts",
) -> RelationalState:
    return RelationalState(tuple(tuple(t) for t in towers), height, abstraction)


class _Scheme:
    def __init__(self, name: str):
        self.name = name


class RelationalRepresentation:
    name = "relational"
    default_abstractions = ("consts",)
    placebo_attr = "block"  # the attribute the placebo refuter perturbs
    transform_families: tuple = ()  # no object-rule vocabulary: plans use domain primitives
    abstractions: Mapping[str, object] = {"consts": _Scheme("consts"), "towers": _Scheme("towers")}

    def parse(
        self, raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> RelationalState:
        if abstraction is not None and abstraction not in self.abstractions:
            raise KeyError(abstraction)
        return RelationalState(
            towers=tuple(tuple(int(b) for b in tower) for tower in raw),
            height=(context or {}).get("height"),
            abstraction=abstraction or self.default_abstractions[0],
        )

    def canon(self, raw) -> tuple:
        return self.parse(raw).key

    def raw_of(self, state: RelationalState) -> Towers:
        return state.towers

    def frame(self, state: RelationalState) -> tuple:
        return (state.universe, state.columns, state.height)

    def rebuild(
        self, template: RelationalState, entities: Sequence[RelEntity]
    ) -> RelationalState | None:
        placed: dict[int, dict[int, int]] = {c: {} for c in range(template.columns)}
        for e in entities:
            ext = e.extent
            if len(ext) == 4 and ext[0] == "at":  # a block entity
                _, block, c, level = ext
                if not 0 <= c < template.columns or level in placed[c]:
                    return None
                placed[c][level] = block
            elif len(ext) == 2:  # a column entity
                c, blocks = ext
                if not 0 <= c < template.columns:
                    return None
                for level, block in enumerate(blocks):
                    if level in placed[c]:
                        return None
                    placed[c][level] = block
            else:
                return None
        towers = []
        for c in range(template.columns):
            levels = placed[c]
            if sorted(levels) != list(range(len(levels))):
                return None  # a floating block: gravity violated
            towers.append(tuple(levels[i] for i in range(len(levels))))
        if template.height is not None and any(len(t) > template.height for t in towers):
            return None
        return RelationalState(tuple(towers), template.height, template.abstraction)

    def candidate_primitives(self, task) -> list:
        from ..domains.blocks import candidate_moves

        return candidate_moves(task)

    def task_values(self, task) -> frozenset[int]:
        return frozenset(task.colours())

    def attr_domain(self, attr: str) -> None:
        return None  # block ids are unbounded

    def fresh_value(self, attr: str, used: Collection) -> int | None:
        if attr != "block":
            return None
        ints = [v for v in used if isinstance(v, int)]
        return max(ints, default=0) + 1

    def relations(self, state: RelationalState) -> set[tuple[str, int, int]]:
        rels: set[tuple[str, int, int]] = set()
        for tower in state.towers:
            for level in range(1, len(tower)):
                rels.add(("on", tower[level], tower[level - 1]))
        return rels

    def overlap(self, a: RelEntity, b: RelEntity) -> float:
        return 1.0 if a.extent == b.extent else 0.0

    # ------------------------------------------------ optional capabilities

    def probe_perturbations(self, state: RelationalState, used: Collection) -> Iterator[Towers]:
        """Per clear (top) block, in column order: remove it, then move it to
        each other column; finally add one fresh block on top of each column."""
        fresh = self.fresh_value("block", used)
        clear = [(c, tower[-1]) for c, tower in enumerate(state.towers) if tower]
        for c, block in clear[:6]:
            removed = [list(t) for t in state.towers]
            removed[c] = removed[c][:-1]
            yield tuple(tuple(t) for t in removed)
            for dest in range(state.columns):
                if dest == c:
                    continue
                moved = [list(t) for t in state.towers]
                moved[c] = moved[c][:-1]
                moved[dest] = [*moved[dest], block]
                yield tuple(tuple(t) for t in moved)
        for c in range(state.columns):
            added = [list(t) for t in state.towers]
            added[c] = [*added[c], fresh]
            yield tuple(tuple(t) for t in added)

    def addition_values(self, state: RelationalState, task) -> list[int]:
        return [self.fresh_value("block", state.universe | self.task_values(task))]

    def addition_catalogue(
        self, state: RelationalState, max_size: int, separated: bool, values: Sequence[int]
    ) -> Iterator[Addition]:
        """Pertinent-negative additions: one new block on top of one column.
        ``values`` are block ids (the PertinentNegative ``colours`` tuple,
        read as ids in this backend); ids already in the world are skipped
        (a duplicate id is not an addition, it breaks block identity).
        ``max_size``/``separated`` have no relational analogue."""
        del max_size, separated
        for c in range(state.columns):
            for v in values:
                if v in state.universe:
                    continue
                towers = [list(t) for t in state.towers]
                towers[c] = [*towers[c], v]
                yield Addition(
                    raw=tuple(tuple(t) for t in towers),
                    phrase=f"block {v} sat on top of column {c}",
                    size=1,
                    group=("col", c),
                )

    def placebo_edit(
        self, state: RelationalState, spectator, forbidden: Collection
    ) -> tuple[Towers, int, object] | None:
        """Rename one program-irrelevant block to a fresh id; the plan must
        pass the rename through unchanged. (The rename changes the frame —
        legal here: backtracking through the refuter has no frame gate.)"""
        old = spectator.attributes["block"]
        used = set(state.universe) | {v for v in forbidden if isinstance(v, int)}
        fresh = self.fresh_value("block", used)

        def rename(towers: Towers) -> Towers:
            return tuple(tuple(fresh if b == old else b for b in t) for t in towers)

        def expect(outcome: RelationalState) -> tuple:
            return ("relational", rename(outcome.towers), outcome.height)

        return rename(state.towers), fresh, expect

    def plausible(self, state: RelationalState) -> bool:
        """Constraint-consistency certificate: every block id occurs exactly
        once and no tower exceeds the height bound (when one is declared)."""
        blocks = [b for tower in state.towers for b in tower]
        if len(blocks) != len(set(blocks)):
            return False
        return state.height is None or all(len(t) <= state.height for t in state.towers)

    def render_raw(self, raw, caption: str | None = None, diff_against=None) -> str:
        """Towers as a coloured column table (block id shown in the cell).
        ``diff_against`` is accepted for interface parity and ignored —
        documented as this renderer's weakest point."""
        from ..viz import _UNKNOWN, PALETTE, _esc

        towers = [list(t) for t in raw]
        height = max([len(t) for t in towers] + [1])
        rows = []
        for level in range(height - 1, -1, -1):
            cells = []
            for tower in towers:
                if level < len(tower):
                    block = tower[level]
                    colour = PALETTE.get(block % 10, _UNKNOWN)
                    cells.append(f'<td style="background:{colour}" title="block {block}"></td>')
                else:
                    cells.append("<td></td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        table = '<table class="g" style="--s:22px">' + "".join(rows) + "</table>"
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return f'<figure class="grid">{table}{cap}</figure>'

    def render_html(self, state: RelationalState, caption: str | None = None) -> str:
        if caption is None:
            caption = f"{state.abstraction} — {len(state.objects)} entit(y/ies)"
        return self.render_raw(state.towers, caption)

    def render_key(self, key, caption: str | None = None) -> str:
        return self.render_raw(key[1], caption)


RELATIONAL = register(RelationalRepresentation())
