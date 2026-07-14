"""Analogy making (thesis Experiment 2).

Structure mapping in the SME tradition, scaled to the slice: local match
hypotheses are scored by object attributes, merged greedily into a one-to-one
mapping (Forbus & Oblinger's greedy merge), then improved by a bounded
swap-repair pass that rewards *relational* agreement — if left_of(a, b) holds
in the input, left_of(m(a), m(b)) should hold in the output (parallel
connectivity). The differences that survive the mapping ARE the observed
transformation (Lovett & Forbus's recipe for Raven's matrices): per-object
deltas are generalized across all train pairs into candidate
:class:`~dowhat.mechanisms.ObjectRule` programs, which the ordinary engine
then verifies. Analogy proposes; search disposes.

MAC/FAC content vectors (predicate-occurrence counts, cosine similarity)
support cheap cross-task retrieval of previously solved tasks.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .engine import Task
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
from .representation import Obj, StateGraph, attribute_score, parse_grid

RELATION_WEIGHT = 3.0  # one preserved relation outweighs a small attribute deficit
MAX_REPAIR_OBJECTS = 15  # beyond this, swap repair is skipped (attribute-greedy only)


def relations(state: StateGraph) -> set[tuple[str, int, int]]:
    """Directed qualitative relations between objects, keyed by oid."""
    rels: set[tuple[str, int, int]] = set()
    for a in state.objects:
        for b in state.objects:
            if a.oid == b.oid:
                continue
            ar, ac = a.location
            br, bc = b.location
            if max(c for _, c in a.cells) < bc:
                rels.add(("left_of", a.oid, b.oid))
            if max(r for r, _ in a.cells) < br:
                rels.add(("above", a.oid, b.oid))
            if a.oid < b.oid:
                if a.colour == b.colour:
                    rels.add(("same_colour", a.oid, b.oid))
                if a.shape == b.shape:
                    rels.add(("same_shape", a.oid, b.oid))
                if a.size > b.size:
                    rels.add(("larger", a.oid, b.oid))
    return rels


def _relational_score(
    mapping: dict[int, int], rels_a: set[tuple[str, int, int]], rels_b: set[tuple[str, int, int]]
) -> float:
    preserved = sum(
        1
        for rel, x, y in rels_a
        if x in mapping and y in mapping and (rel, mapping[x], mapping[y]) in rels_b
    )
    return RELATION_WEIGHT * preserved


def structure_map(a: StateGraph, b: StateGraph) -> list[tuple[Obj | None, Obj | None]]:
    """One-to-one object mapping maximizing attribute + relational agreement.

    Greedy merge on attribute score, then bounded swap-repair passes that
    accept any reassignment improving the global (attribute + relation) score.
    Returns the same pair structure as ``match_objects``.
    """
    a_objs = {o.oid: o for o in a.objects}
    b_objs = {o.oid: o for o in b.objects}
    candidates = sorted(
        ((attribute_score(x, y), x.oid, y.oid) for x in a.objects for y in b.objects),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    mapping: dict[int, int] = {}
    taken_b: set[int] = set()
    for s, xid, yid in candidates:
        if s <= 0 or xid in mapping or yid in taken_b:
            continue
        mapping[xid] = yid
        taken_b.add(yid)

    if 0 < len(mapping) and max(len(a.objects), len(b.objects)) <= MAX_REPAIR_OBJECTS:
        rels_a, rels_b = relations(a), relations(b)

        def total(m: dict[int, int]) -> float:
            attr = sum(attribute_score(a_objs[x], b_objs[y]) for x, y in m.items())
            return attr + _relational_score(m, rels_a, rels_b)

        best = total(mapping)
        for _ in range(3):  # bounded repair
            improved = False
            keys = sorted(mapping)
            for i, x1 in enumerate(keys):
                for x2 in keys[i + 1 :]:
                    trial = dict(mapping)
                    trial[x1], trial[x2] = mapping[x2], mapping[x1]
                    t = total(trial)
                    if t > best:
                        mapping, best, improved = trial, t, True
            if not improved:
                break

    pairs: list[tuple[Obj | None, Obj | None]] = [
        (a_objs[x], b_objs[y]) for x, y in sorted(mapping.items())
    ]
    matched_b = set(mapping.values())
    pairs.extend((o, None) for o in a.objects if o.oid not in mapping)
    pairs.extend((None, o) for o in b.objects if o.oid not in matched_b)
    return pairs


# ------------------------------------------------------------- rule induction


@dataclass(frozen=True)
class Delta:
    """What one input object became, per the structure mapping."""

    obj: Obj
    moved: tuple[int, int] | None  # location delta; None when deleted
    shape_stable: bool
    recoloured_to: int | None  # output is uniformly this (new) colour
    deleted: bool


def pair_deltas(state_in: StateGraph, state_out: StateGraph) -> list[Delta]:
    deltas = []
    for x, y in structure_map(state_in, state_out):
        if x is None:
            continue  # appearances are outside the current rule language
        if y is None:
            deltas.append(Delta(x, None, False, None, True))
            continue
        moved = (y.location[0] - x.location[0], y.location[1] - x.location[1])
        recoloured_to = None
        if y.colours == frozenset({y.colour}) and x.colours != y.colours:
            recoloured_to = y.colour
        deltas.append(Delta(x, moved, x.shape == y.shape, recoloured_to, False))
    return deltas


def induce_rules(task: Task, abstraction: str = "cc4") -> list[ObjectRule]:
    """Propose ObjectRules consistent with every train pair's structure mapping.

    A candidate needs its selector to pick at least one object in every train
    input and its transform to agree with every selected object's delta. The
    engine remains the verifier — bad proposals simply fail search.
    """
    inputs = [parse_grid(i, abstraction) for i, _ in task.train]
    outputs = [parse_grid(o, abstraction) for _, o in task.train]
    if any(not s.objects for s in inputs):
        return []
    all_deltas = [pair_deltas(a, b) for a, b in zip(inputs, outputs)]
    by_obj = [{d.obj.oid: d for d in ds} for ds in all_deltas]

    shared_colours = set.intersection(*({o.colour for o in s.objects} for s in inputs))
    selectors = [All(), *[ByColour(c) for c in sorted(shared_colours)], Largest(), Smallest()]

    rules: list[ObjectRule] = []
    for sel in selectors:
        selected = [
            [by_obj[k].get(o.oid) for o in sel.select(s.objects)]
            for k, s in enumerate(inputs)
        ]
        if any(not group or None in group for group in selected):
            continue
        flat: list[Delta] = [d for group in selected for d in group]

        moves = {d.moved for d in flat}
        if len(moves) == 1 and not any(d.deleted for d in flat):
            (move,) = moves
            if move != (0, 0) and all(d.shape_stable for d in flat):
                rules.append(ObjectRule(sel, TranslateBy(*move)))

        targets = {d.recoloured_to for d in flat}
        if len(targets) == 1 and None not in targets:
            (target,) = targets
            rules.append(ObjectRule(sel, RecolourTo(target)))

        if all(d.deleted for d in flat):
            rules.append(ObjectRule(sel, Delete()))

    unique: list[ObjectRule] = []
    for r in rules:
        if r not in unique:
            unique.append(r)
    return unique


# ------------------------------------------------------- MAC/FAC retrieval


def content_vector(state: StateGraph) -> Counter:
    """MAC-stage content vector: predicate-occurrence counts (Forbus et al.)."""
    v: Counter = Counter()
    v["objects"] = len(state.objects)
    for o in state.objects:
        v[("colour", o.colour)] += 1
        v[("size", min(o.size, 5))] += 1
        for sym in o.symmetries:
            v[("symmetry", sym)] += 1
    for rel, _, _ in relations(state):
        v[("rel", rel)] += 1
    return v


def similarity(v1: Counter, v2: Counter) -> float:
    """Cosine similarity of two content vectors (the MAC dot product)."""
    dot = sum(v1[k] * v2[k] for k in v1.keys() & v2.keys())
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def retrieve(
    probe: StateGraph, library: list[tuple[str, Counter]], k: int = 3
) -> list[tuple[str, float]]:
    """Rank stored task vectors by similarity to the probe state (MAC stage)."""
    pv = content_vector(probe)
    ranked = sorted(((tid, similarity(pv, v)) for tid, v in library), key=lambda t: -t[1])
    return ranked[:k]
