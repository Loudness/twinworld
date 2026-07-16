"""The tabular representation backend — feature rows, registered as ``"tabular"``.

States are feature rows (sorted name/value tuples); the "solver" is a
DETERMINISTIC surrogate: an ordered rule list whose steps are
:class:`SetLabelIf` mechanisms — each decision-list rule is one program step,
so ``solve()`` literally induces the decision list from labelled rows, and the
whole counterfactual suite applies: Backtracking is the recourse what-if,
contrastive edits are certified rule substitutions, and Rashomon-set
underdetermination (several rule lists fitting the same rows) is diagnosed by
the ordinary probe machinery.

Deliberate deviation, documented: a rule-list step that does not fire PASSES
THE ROW THROUGH unchanged (like Identity) instead of the usual
no-op-means-inapplicable convention — a decision list must fall through to the
next rule. Deferred by design (docs/beyond-grids.md §5.3): class-REGION
contrastive targets and minimal input-edit search; range-based plausibility is
also deferred because the schema never reaches ``parse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Collection, Hashable, Iterator, Mapping, Sequence

from ..backend import register

Row = tuple[tuple[str, Hashable], ...]  # sorted (name, value) pairs

LABEL = "label"  # the annotated field the rule list writes


@dataclass(frozen=True)
class FeatureEntity:
    """One feature of a row. ``oid`` is the feature's index in the sorted
    schema — stable across a task because the frame pins the schema."""

    oid: int
    extent: tuple
    size: int
    _attrs: tuple[tuple[str, Hashable], ...]

    @cached_property
    def attributes(self) -> dict[str, Hashable]:
        return dict(self._attrs)


@dataclass(frozen=True, eq=False)
class FeatureState:
    """A feature row. Equality and hashing go through ``key`` — the sorted
    name/value tuple."""

    values: Row
    abstraction: str = "features"

    representation = "tabular"

    @property
    def key(self) -> tuple:
        return ("tabular", self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FeatureState) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def row(self) -> dict[str, Hashable]:
        return dict(self.values)

    @cached_property
    def objects(self) -> tuple[FeatureEntity, ...]:
        return tuple(
            FeatureEntity(
                oid=i,
                extent=(name, value),
                size=1,
                _attrs=(("name", name), ("value", value)),
            )
            for i, (name, value) in enumerate(self.values)
        )


@dataclass(frozen=True)
class SetLabelIf:
    """One decision-list rule: when the row is still unlabelled and the
    condition holds, write ``label``; otherwise pass the row through
    (see the module docstring for the convention deviation)."""

    feature: str
    op: str  # ">=" (numeric) | "==" (categorical)
    threshold: Hashable
    label: Hashable
    label_field: str = LABEL

    exact_preimage = True  # predecessors = {the row itself, the unlabelled row}

    def _matches(self, value) -> bool:
        if self.op == ">=":
            return isinstance(value, (int, float)) and value >= self.threshold
        return value == self.threshold

    def apply(self, s: FeatureState) -> FeatureState | None:
        row = s.row()
        if self.feature not in row or self.label_field not in row:
            return None  # schema mismatch: genuinely inapplicable
        if row[self.label_field] is not None:
            return s  # already decided: fall through
        if not self._matches(row[self.feature]):
            return s  # rule does not fire: fall through
        row[self.label_field] = self.label
        return FeatureState(tuple(sorted(row.items())), s.abstraction)

    def preimage(self, s: FeatureState, budget=None) -> Iterator[FeatureState]:
        # budget accepted for protocol uniformity; the predecessor set is tiny
        if self.apply(s) == s:
            yield s  # the pass-through predecessor
        row = s.row()
        if row.get(self.label_field) == self.label and self._matches(row.get(self.feature)):
            row[self.label_field] = None
            candidate = FeatureState(tuple(sorted(row.items())), s.abstraction)
            if self.apply(candidate) == s:
                yield candidate

    def __str__(self) -> str:
        return f"set {self.label_field} to {self.label!r} if {self.feature} {self.op} {self.threshold!r}"


class _Scheme:
    def __init__(self, name: str):
        self.name = name


class TabularRepresentation:
    name = "tabular"
    default_abstractions = ("features",)
    transform_families: tuple = ()  # induction is blind enumeration over thresholds
    abstractions: Mapping[str, object] = {"features": _Scheme("features")}

    def parse(
        self, raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> FeatureState:
        if abstraction is not None and abstraction not in self.abstractions:
            raise KeyError(abstraction)
        pairs = raw.items() if isinstance(raw, Mapping) else raw
        return FeatureState(
            tuple(sorted((str(n), v) for n, v in pairs)),
            abstraction or self.default_abstractions[0],
        )

    def canon(self, raw) -> tuple:
        return self.parse(raw).key

    def raw_of(self, state: FeatureState) -> Row:
        return state.values

    def frame(self, state: FeatureState) -> tuple[str, ...]:
        return tuple(name for name, _ in state.values)  # the schema is the frame

    def rebuild(
        self, template: FeatureState, entities: Sequence[FeatureEntity]
    ) -> FeatureState | None:
        seen: dict[str, Hashable] = {}
        for entity in entities:
            name, value = entity.extent
            if name in seen:
                return None
            seen[name] = value
        return FeatureState(tuple(sorted(seen.items())), template.abstraction)

    def candidate_primitives(self, task) -> list[SetLabelIf]:
        """Decision-list vocabulary anchored to OBSERVED values: numeric
        features get ``>=`` thresholds, categorical ones ``==`` tests, labels
        come from the train outputs — finite and principled, which is what
        keeps blind rule-list induction tractable here."""
        observed: dict[str, set] = {}
        labels: set = set()
        for raw_in, raw_out in task.train:
            for name, value in self.parse(raw_in).values:
                if name != LABEL:
                    observed.setdefault(name, set()).add(value)
            label = dict(self.parse(raw_out).values).get(LABEL)
            if label is not None:
                labels.add(label)
        rules = []
        for name in sorted(observed):
            for value in sorted(observed[name], key=repr):
                op = ">=" if isinstance(value, (int, float)) else "=="
                for label in sorted(labels, key=repr):
                    rules.append(SetLabelIf(name, op, value, label))
        return rules

    def task_values(self, task) -> frozenset:
        return frozenset(
            value
            for pairs in (task.train, task.test)
            for pair in pairs
            for raw in pair
            for _, value in self.parse(raw).values
        )

    def attr_domain(self, attr: str) -> None:
        return None  # numeric domains are unenumerable

    def fresh_value(self, attr: str, used: Collection) -> str:
        return "<fresh>"

    def relations(self, state: FeatureState) -> set[tuple[str, int, int]]:
        return set()  # schema-declared relations are future work

    def overlap(self, a: FeatureEntity, b: FeatureEntity) -> float:
        return 1.0 if a.extent == b.extent else 0.0

    # ------------------------------------------------ optional capabilities

    def probe_perturbations(self, state: FeatureState, used: Collection) -> Iterator[Row]:
        """Per feature (first 6, label excluded): nudge numerics by ±1 and
        swap categoricals to an unused token."""
        names = [name for name, _ in state.values if name != LABEL][:6]
        row = state.row()
        for name in names:
            value = row[name]
            if isinstance(value, (int, float)):
                for nudged in (value + 1, value - 1):
                    yield tuple(sorted({**row, name: nudged}.items()))
            else:
                yield tuple(sorted({**row, name: self.fresh_value("value", used)}.items()))

    def distance(self, a: FeatureState, b: FeatureState) -> float:
        """Gower-style row distance: categorical mismatch costs 1, numeric
        differences are self-normalized to [0, 1], missing features cost 2."""
        row_a, row_b = a.row(), b.row()
        cost = 0.0
        for name in set(row_a) | set(row_b):
            if name not in row_a or name not in row_b:
                cost += 2.0
                continue
            va, vb = row_a[name], row_b[name]
            if va == vb:
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                cost += min(1.0, abs(va - vb) / max(abs(va), abs(vb), 1))
            else:
                cost += 1.0
        return cost

    def plausible(self, state: FeatureState) -> bool:
        """Structural certificate only (the schema never reaches parse, so
        range checks are deferred): unique sorted names, scalar values."""
        names = [name for name, _ in state.values]
        if names != sorted(names) or len(names) != len(set(names)):
            return False
        return all(
            value is None or isinstance(value, (int, float, str))
            for _, value in state.values
        )

    def render_raw(self, raw, caption: str | None = None, diff_against=None) -> str:
        from ..viz import _esc

        state = raw if isinstance(raw, FeatureState) else self.parse(raw)
        rows = "".join(
            f"<tr><td>{_esc(name)}</td><td>{_esc(value)}</td></tr>"
            for name, value in state.values
        )
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return f'<figure class="grid"><table class="t">{rows}</table>{cap}</figure>'

    def render_html(self, state: FeatureState, caption: str | None = None) -> str:
        return self.render_raw(state, caption)

    def render_key(self, key, caption: str | None = None) -> str:
        return self.render_raw(FeatureState(key[1]), caption)


TABULAR = register(TabularRepresentation())


def rows_task(
    train: Sequence[tuple[Mapping, Hashable]],
    test: Sequence[tuple[Mapping, Hashable]],
    task_id: str = "tabular",
):
    """A decision-list task from (row, label) pairs: the input is the row with
    ``label=None``, the expected output is the same row labelled."""
    from ..engine import Task

    def pair(row: Mapping, label: Hashable) -> tuple[Row, Row]:
        raw_in = tuple(sorted({**dict(row), LABEL: None}.items()))
        raw_out = tuple(sorted({**dict(row), LABEL: label}.items()))
        return raw_in, raw_out

    return Task(
        train=tuple(pair(row, label) for row, label in train),
        test=tuple(pair(row, label) for row, label in test),
        task_id=task_id,
        representation="tabular",
    )
