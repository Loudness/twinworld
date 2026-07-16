"""Representation backends: the contract between the core and a state substrate.

The core trusts exactly one thing about a state — a hashable canonical ``key``
— plus an object/attribute ontology (docs/beyond-grids.md). This module makes
that contract explicit: :class:`Representation` is what a substrate must
supply, ``REPRESENTATIONS`` is the registry the core dispatches through, and
:func:`conformance_battery` checks the key laws L1–L7 on a candidate backend
("measured, not asserted", applied to backends themselves).

This module imports nothing from the rest of twinworld; built-in backends are
resolved lazily by name so constructing ``Task(representation="relational")``
works without a prior import.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from itertools import islice
from typing import (
    Any,
    Collection,
    Hashable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

Key = Hashable  # canonical state identity token
Raw = Any  # backend-interpretable payload: a Grid, tower tuples, atoms, ...


@runtime_checkable
class Entity(Protocol):
    """One object in a state: an id, a name-keyed attribute ontology, a
    canonical footprint, and an ordinal magnitude (what Largest/Smallest use)."""

    oid: int

    @property
    def attributes(self) -> Mapping[str, Hashable]: ...

    @property
    def extent(self) -> Hashable: ...

    @property
    def size(self) -> int: ...


@runtime_checkable
class State(Protocol):
    """A parsed state. ``key`` is the identity the whole core verifies against;
    ``__eq__``/``__hash__`` must route through it (law L3). ``abstraction``
    records how this observation was segmented — identity must not depend on
    it (law L2)."""

    representation: str
    abstraction: str
    objects: tuple[Entity, ...]

    @property
    def key(self) -> Key: ...


class TransformFamily(Protocol):
    """One block of rule-induction vocabulary: turns a list of per-object
    deltas (analogy.Delta) into candidate object transforms."""

    def emit(self, deltas: Sequence) -> Iterator: ...


class Representation(Protocol):
    """What a state substrate owes the core.

    Required members cover parsing, identity, and the mechanism/vocabulary
    hooks. Optional capability members (probed by :func:`capabilities`) are:
    ``probe_perturbations``, ``addition_values`` + ``addition_catalogue``,
    ``placebo_edit``, ``render_raw``/``render_html``/``render_key``,
    ``render_text``, ``distance``, ``plausible``, ``generate``.
    """

    name: str
    default_abstractions: tuple[str, ...]
    transform_families: tuple[TransformFamily, ...]

    @property
    def abstractions(self) -> Mapping[str, object]: ...

    def parse(
        self, raw: Raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> State: ...

    def canon(self, raw: Raw) -> Key: ...

    def raw_of(self, state: State) -> Raw: ...

    def frame(self, state: State) -> Hashable | None: ...

    def rebuild(self, template: State, entities: Sequence[Entity]) -> State | None: ...

    def candidate_primitives(self, task) -> list: ...

    def task_values(self, task) -> frozenset: ...

    def attr_domain(self, attr: str) -> Sequence[Hashable] | None: ...

    def fresh_value(self, attr: str, used: Collection) -> Hashable | None: ...

    def relations(self, state: State) -> set[tuple[str, int, int]]: ...

    def overlap(self, a: Entity, b: Entity) -> float: ...


@dataclass(frozen=True)
class Addition:
    """One pertinent-negative catalogue item: an edited raw payload, the
    narrative fragment describing what was added, its footprint size, and a
    group token (one witness is kept per group — per anchor, historically)."""

    raw: Raw
    phrase: str
    size: int
    group: Hashable


REPRESENTATIONS: dict[str, Representation] = {}

_BUILTIN_MODULES = {
    "grid": "twinworld.backends.grid",
    "relational": "twinworld.backends.relational",
}

_CAPABILITY_MEMBERS = {
    "probes": "probe_perturbations",
    "pertinent_negative": "addition_catalogue",
    "placebo": "placebo_edit",
    "viz": "render_html",
    "text": "render_text",
    "distance": "distance",
    "plausibility": "plausible",
    "benchmark": "generate",
}


def register(rep: Representation) -> Representation:
    """Register a backend under ``rep.name`` (re-registering replaces it)."""
    REPRESENTATIONS[rep.name] = rep
    return rep


def get_representation(name: str) -> Representation:
    if name not in REPRESENTATIONS and name in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[name])  # registers itself on import
    if name not in REPRESENTATIONS:
        known = sorted(set(REPRESENTATIONS) | set(_BUILTIN_MODULES))
        raise KeyError(f"unknown representation {name!r}; known: {known}")
    return REPRESENTATIONS[name]


def representation_of(obj) -> Representation:
    """Resolve the backend of a Task, State, or name string."""
    if isinstance(obj, str):
        return get_representation(obj)
    return get_representation(obj.representation)


def capabilities(rep: Representation) -> frozenset[str]:
    return frozenset(cap for cap, member in _CAPABILITY_MEMBERS.items() if hasattr(rep, member))


@dataclass(frozen=True)
class ConformanceRow:
    name: str
    passed: bool | None  # None = not applicable on the given samples
    detail: str


@dataclass(frozen=True)
class ConformanceReport:
    rows: tuple[ConformanceRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed is not False for row in self.rows)

    def __str__(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}
        return "\n".join(f"[{mark[r.passed]}] {r.name}: {r.detail}" for r in self.rows)


def conformance_battery(
    rep: Representation,
    sample_raws: Sequence[Raw],
    mechanisms: Sequence = (),
    preimage_cap: int = 20,
) -> ConformanceReport:
    """Check the backend laws on sample payloads.

    L1 parse/canon agreement; L2 key invariance under abstraction choice;
    L3 eq/hash route through the key; L4 apply is deterministic and canonical;
    L5 preimages re-apply to the source key; L6 exact preimages contain the
    true predecessor; L7 rebuild round-trips a state from its own entities.
    """
    rows: list[ConformanceRow] = []

    def run(name: str, check) -> None:
        try:
            passed, detail = check()
        except Exception as exc:  # a law that raises is a law that fails
            passed, detail = False, f"raised {exc!r}"
        rows.append(ConformanceRow(name, passed, detail))

    def l1():
        bad = sum(1 for raw in sample_raws if rep.parse(raw).key != rep.canon(raw))
        return bad == 0, f"{len(sample_raws) - bad}/{len(sample_raws)} samples agree"

    def l2():
        names = tuple(rep.abstractions)
        bad = sum(1 for raw in sample_raws if len({rep.parse(raw, a).key for a in names}) != 1)
        return bad == 0, f"key stable across {len(names)} abstraction(s) on {len(sample_raws)} sample(s); {bad} unstable"

    def l3():
        for raw in sample_raws:
            first, second = rep.parse(raw), rep.parse(raw)
            if first != second or hash(first) != hash(second):
                return False, "re-parse is not equal/hash-stable"
            for a in rep.abstractions:
                if rep.parse(raw, a) != first:
                    return False, f"equality depends on abstraction {a!r}"
        return True, f"{len(sample_raws)} sample(s) equal and hash-stable"

    def l4():
        checked = 0
        for raw in sample_raws:
            state = rep.parse(raw)
            for mech in mechanisms:
                once, twice = mech.apply(state), mech.apply(state)
                if (once is None) != (twice is None):
                    return False, f"{mech} is nondeterministic"
                if once is None:
                    continue
                checked += 1
                if once.key != twice.key:
                    return False, f"{mech} is nondeterministic"
                if rep.parse(rep.raw_of(once), once.abstraction).key != once.key:
                    return False, f"{mech} result is not parse-canonical"
        if checked == 0:
            return None, "no applicable (state, mechanism) pair in the samples"
        return True, f"{checked} application(s) deterministic and canonical"

    def l5():
        checked = 0
        for raw in sample_raws:
            state = rep.parse(raw)
            for mech in mechanisms:
                for pre in islice(mech.preimage(state), preimage_cap):
                    checked += 1
                    redone = mech.apply(pre)
                    if redone is None or redone.key != state.key:
                        return False, f"{mech} yielded an unsound preimage"
        if checked == 0:
            return None, "no preimages yielded on the samples"
        return True, f"{checked} preimage(s) re-apply to the source key"

    def l6():
        checked = 0
        for raw in sample_raws:
            state = rep.parse(raw)
            for mech in mechanisms:
                if not getattr(mech, "exact_preimage", False):
                    continue
                result = mech.apply(state)
                if result is None:
                    continue
                checked += 1
                if not any(p == state for p in islice(mech.preimage(result), preimage_cap)):
                    return False, f"{mech} claims exact_preimage but misses the true predecessor"
        if checked == 0:
            return None, "no exact-preimage mechanism applied on the samples"
        return True, f"{checked} exact preimage(s) contain the true predecessor"

    def l7():
        for raw in sample_raws:
            state = rep.parse(raw)
            rebuilt = rep.rebuild(state, state.objects)
            if rebuilt is None or rebuilt.key != state.key:
                return False, "rebuild(state, state.objects) does not round-trip the key"
            if rep.frame(rebuilt) != rep.frame(state):
                return False, "rebuild changed the frame"
        return True, f"{len(sample_raws)} state(s) rebuild to themselves"

    run("L1_parse_canon", l1)
    run("L2_key_abstraction_invariance", l2)
    run("L3_eq_hash_via_key", l3)
    run("L4_apply_canonical", l4)
    run("L5_preimage_sound", l5)
    run("L6_exact_preimage_spot", l6)
    run("L7_rebuild_closure", l7)
    return ConformanceReport(tuple(rows))
