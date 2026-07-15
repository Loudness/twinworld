"""Learned concept network (thesis Experiment 2 — the digest's open question).

Copycat's Slipnet was hand-coded and never learned (Hofstadter & Mitchell
1994; Mitchell 2021); this library's analog — the attribute weights in
:func:`~twinworld.representation.attribute_score`, the uniform relation weight,
and the unordered rule-candidate vocabulary — has been hand-coded too. Here
those numbers become DATA: :func:`learn_concepts` estimates them from a task
corpus and :class:`ConceptNet` carries them explicitly (JSON-serializable, per
the digest's "explicit declarative data" requirement).

The estimation is deliberately non-circular: correspondences are anchored by
leave-one-attribute-out uniqueness (colour statistics come only from pairs
anchored by shape alone; shape statistics only from pairs anchored by colour
alone), never by the scores being learned. Weights are clipped log-odds of
agreement on anchored pairs versus random cross pairs. Slips — the corpus
probability that a TRUE correspondence changes an attribute — quantify
Copycat's slippage for :func:`twinworld.copycat.copycat_map`. Rule-family priors
only reorder induced candidates; the engine remains the verifier.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .engine import ApplyCache, Task
from .representation import StateGraph, as_grid, parse_grid

if TYPE_CHECKING:
    from .mechanisms import ObjectRule

_EPS = 0.05  # log-odds smoothing
_MAX_OBJECTS = 30  # states larger than this are skipped during learning
_RELATION_NAMES = ("above", "larger", "left_of", "same_colour", "same_shape")


@dataclass(frozen=True)
class ConceptNet:
    """The concept network as explicit data; defaults = the hand-coded values."""

    shape: float = 4.0
    colour: float = 2.0
    location: float = 1.0
    iou: float = 3.0
    relations: tuple[tuple[str, float], ...] = tuple((n, 3.0) for n in _RELATION_NAMES)
    slip_shape: float = 0.0  # P(shape changes | true correspondence)
    slip_colour: float = 0.0
    slip_move: float = 0.0
    slip_delete: float = 0.0
    priors: tuple[tuple[str, float], ...] = ()  # (selector*transform family, freq)
    source: str = "hand-coded"

    def relation_weight(self, name: str) -> float:
        return dict(self.relations).get(name, 0.0)

    def prior(self, rule: ObjectRule) -> float:
        return dict(self.priors).get(rule_family(rule), 0.0)


DEFAULT_CONCEPTS = ConceptNet()


def save_concepts(net: ConceptNet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(asdict(net), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_concepts(path: str | Path) -> ConceptNet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["relations"] = tuple((n, w) for n, w in raw["relations"])
    raw["priors"] = tuple((f, p) for f, p in raw["priors"])
    return ConceptNet(**raw)


def rule_family(rule: ObjectRule) -> str:
    """'ByColour*RecolourTo', 'Not(Largest)*Delete', ... — the prior's key space."""
    sel = rule.selector
    inner = getattr(sel, "inner", None)
    sel_name = f"Not({type(inner).__name__})" if inner is not None else type(sel).__name__
    return f"{sel_name}*{type(rule.transform).__name__}"


# ---------------------------------------------------------------- estimation


def _unique_match(a: StateGraph, b: StateGraph, key) -> list[tuple]:
    """Pairs (x, y) where key(x) occurs exactly once in each state and matches."""
    a_by: dict = {}
    b_by: dict = {}
    for o in a.objects:
        a_by.setdefault(key(o), []).append(o)
    for o in b.objects:
        b_by.setdefault(key(o), []).append(o)
    pairs = [
        (xs[0], b_by[k][0])
        for k, xs in a_by.items()
        if len(xs) == 1 and len(b_by.get(k, ())) == 1
    ]
    return sorted(pairs, key=lambda p: p[0].oid)


def _log_odds(p_anchor: float, p_random: float) -> float:
    return max(0.0, math.log2((p_anchor + _EPS) / (p_random + _EPS)))


class _Stats:
    """Streaming counters for one attribute or relation: anchored vs random."""

    def __init__(self) -> None:
        self.hit = 0.0
        self.n = 0
        self.base_hit = 0.0
        self.base_n = 0

    def rate(self) -> tuple[float, float]:
        p = self.hit / self.n if self.n else 0.0
        q = self.base_hit / self.base_n if self.base_n else 0.0
        return p, q

    def weight(self) -> float | None:
        return _log_odds(*self.rate()) if self.n else None


def learn_concepts(tasks: Iterable[Task], abstraction: str = "cc4") -> ConceptNet:
    """Estimate a ConceptNet from a corpus's train pairs (never touches tests)."""
    from .analogy import induce_rules, relations

    attrs = {name: _Stats() for name in ("shape", "colour", "location", "iou")}
    rels = {name: _Stats() for name in _RELATION_NAMES}
    deleted = _Stats()
    families: dict[str, int] = {}
    n_tasks = 0

    for task in tasks:
        n_tasks += 1
        cache = ApplyCache()
        for rule in induce_rules(task, abstraction):
            if all(
                (t := cache.run(parse_grid(i, abstraction), (rule,))) is not None
                and t.outcome.key == as_grid(o)
                for i, o in task.train
            ):
                fam = rule_family(rule)
                families[fam] = families.get(fam, 0) + 1

        for grid_in, grid_out in task.train:
            a = parse_grid(grid_in, abstraction)
            b = parse_grid(grid_out, abstraction)
            if not a.objects or not b.objects:
                continue
            if max(len(a.objects), len(b.objects)) > _MAX_OBJECTS:
                continue

            # leave-one-attribute-out anchors
            by_shape = _unique_match(a, b, lambda o: o.shape)  # colour left free
            by_colour = _unique_match(a, b, lambda o: o.colour)  # shape left free
            strict = _unique_match(a, b, lambda o: (o.shape, o.colour))

            for x, y in by_shape:
                attrs["colour"].hit += x.colour == y.colour
                attrs["colour"].n += 1
            for x, y in by_colour:
                attrs["shape"].hit += x.shape == y.shape
                attrs["shape"].n += 1
            for x, y in strict:
                attrs["location"].hit += x.location == y.location
                attrs["location"].n += 1
                attrs["iou"].hit += len(x.cells & y.cells) / len(x.cells | y.cells)
                attrs["iou"].n += 1

            # random cross-pair base rates
            for x in a.objects:
                for y in b.objects:
                    attrs["colour"].base_hit += x.colour == y.colour
                    attrs["shape"].base_hit += x.shape == y.shape
                    attrs["location"].base_hit += x.location == y.location
                    attrs["iou"].base_hit += len(x.cells & y.cells) / len(x.cells | y.cells)
            for name in ("shape", "colour", "location", "iou"):
                attrs[name].base_n += len(a.objects) * len(b.objects)

            # relation preservation across the strict anchor mapping
            mapping = {x.oid: y.oid for x, y in strict}
            rels_a, rels_b = relations(a), relations(b)
            n_b = len(b.objects)
            for name, i, j in rels_a:
                if i in mapping and j in mapping:
                    rels[name].hit += (name, mapping[i], mapping[j]) in rels_b
                    rels[name].n += 1
            if n_b > 1:
                for name in _RELATION_NAMES:
                    rels[name].base_hit += sum(1 for r, _, _ in rels_b if r == name)
                    rels[name].base_n += n_b * (n_b - 1)

            # deletion base rate: input objects with no colour or shape counterpart
            b_shapes = {o.shape for o in b.objects}
            b_colours = {o.colour for o in b.objects}
            for x in a.objects:
                deleted.hit += x.shape not in b_shapes and x.colour not in b_colours
                deleted.n += 1

    hand = DEFAULT_CONCEPTS
    weights = {
        name: (stats.weight() if stats.weight() is not None else getattr(hand, name))
        for name, stats in attrs.items()
    }
    relation_weights = tuple(
        (name, rels[name].weight() if rels[name].weight() is not None else 3.0)
        for name in _RELATION_NAMES
    )
    total_verified = sum(families.values())
    priors = tuple(
        sorted(((f, n / total_verified) for f, n in families.items()), key=lambda t: -t[1])
    )

    def slip(stats: _Stats) -> float:
        p, _ = stats.rate()
        return round(1.0 - p, 4) if stats.n else 0.0

    delete_rate = round(deleted.rate()[0], 4) if deleted.n else 0.0
    return ConceptNet(
        shape=round(weights["shape"], 4),
        colour=round(weights["colour"], 4),
        location=round(weights["location"], 4),
        iou=round(weights["iou"], 4),
        relations=tuple((n, round(w, 4)) for n, w in relation_weights),
        slip_shape=slip(attrs["shape"]),
        slip_colour=slip(attrs["colour"]),
        slip_move=slip(attrs["location"]),
        slip_delete=delete_rate,
        priors=priors,
        source=f"learned from {n_tasks} task(s), abstraction {abstraction}",
    )
