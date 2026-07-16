"""The sequence representation backend — letter strings, registered as
``"sequence"``.

Copycat's home domain (Hofstadter & Mitchell 1994) as a twinworld substrate:
states are symbol tuples, entities are letters (or runs), and the rule
vocabulary is successor/replace/delete under an EXPLICIT alphabet — so a
permuted alphabet is a different frozen mechanism, and the counterfactual
letter-string manipulations of Lewis & Mitchell (arXiv:2402.08955) become
ordinary Interventional queries (swap ``SuccessorT(standard)`` for
``SuccessorT(permuted)``) rather than prompt engineering.

Two segmentation schemes make re-segmentation meaningful: ``"letters"`` (one
entity per position) and ``"runs"`` (maximal same-letter or successor runs —
Copycat's groups).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Collection, Hashable, Iterator, Mapping, Sequence

from ..backend import Addition, register
from ..concepts import ConceptNet
from ..mechanisms import All, ByAttr, Not

ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz")

Symbols = tuple[str, ...]


@dataclass(frozen=True)
class SeqEntity:
    """One sequence entity: a letter at a position, or a run of them."""

    oid: int
    extent: tuple
    size: int
    _attrs: tuple[tuple[str, Hashable], ...]

    @cached_property
    def attributes(self) -> dict[str, Hashable]:
        return dict(self._attrs)


def _runs(symbols: Symbols) -> list[tuple[int, Symbols, str]]:
    """Maximal runs of equal or (standard-)successor letters: (start, syms, kind)."""
    runs: list[tuple[int, Symbols, str]] = []
    i = 0
    while i < len(symbols):
        j = i + 1
        kind = "single"
        while j < len(symbols):
            if symbols[j] == symbols[j - 1] and kind in ("single", "same"):
                kind = "same"
            elif (
                symbols[j - 1] in ALPHABET
                and ALPHABET.index(symbols[j - 1]) + 1 < len(ALPHABET)
                and symbols[j] == ALPHABET[ALPHABET.index(symbols[j - 1]) + 1]
                and kind in ("single", "successor")
            ):
                kind = "successor"
            else:
                break
            j += 1
        runs.append((i, symbols[i:j], kind))
        i = j
    return runs


@dataclass(frozen=True, eq=False)
class SeqState:
    """A letter string. Equality and hashing go through ``key`` — the symbol
    tuple — regardless of segmentation."""

    symbols: Symbols
    abstraction: str = "letters"

    representation = "sequence"

    @property
    def key(self) -> tuple:
        return ("sequence", self.symbols)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SeqState) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    @cached_property
    def objects(self) -> tuple[SeqEntity, ...]:
        if self.abstraction == "runs":
            return tuple(
                SeqEntity(
                    oid=start,
                    extent=(start, syms),
                    size=len(syms),
                    _attrs=(
                        ("letter", syms[0]),
                        ("position", start),
                        ("length", len(syms)),
                        ("kind", kind),
                    ),
                )
                for start, syms, kind in _runs(self.symbols)
            )
        return tuple(
            SeqEntity(
                oid=i,
                extent=(i, sym),
                size=1,
                _attrs=(("letter", sym), ("position", i)),
            )
            for i, sym in enumerate(self.symbols)
        )


# ------------------------------------------------------------------ selectors


@dataclass(frozen=True)
class Rightmost:
    def select(self, objects: tuple[SeqEntity, ...]) -> tuple[SeqEntity, ...]:
        if not objects:
            return ()
        top = max(o.attributes["position"] for o in objects)
        return tuple(o for o in objects if o.attributes["position"] == top)

    def __str__(self) -> str:
        return "the rightmost letter(s)"


@dataclass(frozen=True)
class Leftmost:
    def select(self, objects: tuple[SeqEntity, ...]) -> tuple[SeqEntity, ...]:
        if not objects:
            return ()
        low = min(o.attributes["position"] for o in objects)
        return tuple(o for o in objects if o.attributes["position"] == low)

    def __str__(self) -> str:
        return "the leftmost letter(s)"


# ------------------------------------------------------------------ transforms


@dataclass(frozen=True)
class SuccessorT:
    """Advance a letter along an EXPLICIT alphabet — a permuted alphabet is a
    different mechanism, so 'what if the alphabet were permuted' is an
    ordinary intervention."""

    alphabet: tuple[str, ...] = ALPHABET

    def replacement(self, sym: str) -> str | None:
        if sym not in self.alphabet:
            return None
        i = self.alphabet.index(sym)
        return self.alphabet[i + 1] if i + 1 < len(self.alphabet) else None

    def __str__(self) -> str:
        tag = "" if self.alphabet == ALPHABET else " of a permuted alphabet"
        return f"advance %s to its successor{tag}"


@dataclass(frozen=True)
class SetLetterTo:
    value: str

    def replacement(self, sym: str) -> str:
        return self.value

    def __str__(self) -> str:
        return f"set %s to {self.value!r}"


@dataclass(frozen=True)
class DeleteT:
    def __str__(self) -> str:
        return "delete %s"


SeqTransform = SuccessorT | SetLetterTo | DeleteT


def _positions_of(entity: SeqEntity) -> tuple[tuple[int, str], ...]:
    start, payload = entity.extent
    if isinstance(payload, tuple):  # a run: (start, symbols)
        return tuple((start + k, sym) for k, sym in enumerate(payload))
    return ((start, payload),)


@dataclass(frozen=True)
class SeqRule:
    """Apply ``transform`` to every letter picked by ``selector`` — the
    sequence analogue of the grid ObjectRule."""

    selector: object
    transform: SeqTransform

    @property
    def exact_preimage(self) -> bool:
        return isinstance(self.transform, SuccessorT) and isinstance(
            self.selector, (All, Rightmost, Leftmost)
        )

    def apply(self, s: SeqState) -> SeqState | None:
        selected = self.selector.select(s.objects)
        if not selected:
            return None
        positions = {pos for entity in selected for pos, _ in _positions_of(entity)}
        if isinstance(self.transform, DeleteT):
            new = tuple(sym for i, sym in enumerate(s.symbols) if i not in positions)
        else:
            out: list[str] = []
            for i, sym in enumerate(s.symbols):
                if i in positions:
                    replacement = self.transform.replacement(sym)
                    if replacement is None:
                        return None  # e.g. the successor of the alphabet's last letter
                    out.append(replacement)
                else:
                    out.append(sym)
            new = tuple(out)
        result = SeqState(new, s.abstraction)
        if result == s:
            return None  # no-op applications are inapplicable, not silent identities
        return result

    def preimage(self, s: SeqState, budget=None) -> Iterator[SeqState]:
        # budget accepted for protocol uniformity; every catalogue below is tiny
        if isinstance(self.transform, SuccessorT):
            inverse = tuple(reversed(self.transform.alphabet))
            pre = SeqRule(self.selector, SuccessorT(inverse)).apply(s)
            if pre is not None and self.apply(pre) == s:
                yield pre
        elif isinstance(self.transform, SetLetterTo):
            value = self.transform.value
            targets = [i for i, sym in enumerate(s.symbols) if sym == value]
            for pos in targets:  # bounded: single-position flips, verified
                for original in ALPHABET:
                    if original == value:
                        continue
                    candidate = SeqState(
                        tuple(
                            original if i == pos else sym for i, sym in enumerate(s.symbols)
                        ),
                        s.abstraction,
                    )
                    if self.apply(candidate) == s:
                        yield candidate
        else:  # DeleteT: insertion catalogue, verified by re-application
            seen: set[Symbols] = set()
            for pos in range(len(s.symbols) + 1):
                for letter in ALPHABET:
                    candidate = SeqState(
                        s.symbols[:pos] + (letter,) + s.symbols[pos:], s.abstraction
                    )
                    if candidate.symbols in seen:
                        continue  # neighbouring insertions of the same letter coincide
                    seen.add(candidate.symbols)
                    if self.apply(candidate) == s:
                        yield candidate

    def __str__(self) -> str:
        return str(self.transform) % str(self.selector)


# ----------------------------------------------------------- rule induction


@dataclass(frozen=True)
class SeqDelta:
    """What one input letter became, per the structure mapping."""

    obj: SeqEntity
    to_letter: str | None
    deleted: bool


class SuccessorFamily:
    """Emit SuccessorT when every selected letter advanced by one."""

    def __init__(self, alphabet: tuple[str, ...] = ALPHABET):
        self.alphabet = alphabet

    def emit(self, deltas: Sequence[SeqDelta]) -> Iterator[SeqTransform]:
        succ = SuccessorT(self.alphabet)
        if all(
            not d.deleted and d.to_letter == succ.replacement(d.obj.attributes["letter"])
            for d in deltas
        ):
            yield succ


class SetLetterFamily:
    """Emit SetLetterTo when every selected letter became one common letter."""

    def emit(self, deltas: Sequence[SeqDelta]) -> Iterator[SeqTransform]:
        targets = {d.to_letter for d in deltas}
        if len(targets) == 1 and None not in targets and not any(d.deleted for d in deltas):
            (target,) = targets
            if any(d.obj.attributes["letter"] != target for d in deltas):
                yield SetLetterTo(target)


class SeqDeleteFamily:
    """Emit DeleteT when every selected letter disappeared."""

    def emit(self, deltas: Sequence[SeqDelta]) -> Iterator[SeqTransform]:
        if all(d.deleted for d in deltas):
            yield DeleteT()


# ------------------------------------------------------------------ backend


class _Scheme:
    def __init__(self, name: str):
        self.name = name


class SequenceRepresentation:
    name = "sequence"
    default_abstractions = ("letters", "runs")
    transform_families = (SuccessorFamily(ALPHABET), SetLetterFamily(), SeqDeleteFamily())
    abstractions: Mapping[str, object] = {
        "letters": _Scheme("letters"),
        "runs": _Scheme("runs"),
    }
    default_concepts = ConceptNet(
        attributes=(("letter", 2.0), ("position", 4.0)),
        relations=(("precedes", 3.0), ("same_letter", 3.0), ("successor_of", 3.0)),
        source="sequence hand-coded",
    )
    placebo_attr = "letter"

    def parse(
        self, raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> SeqState:
        if abstraction is not None and abstraction not in self.abstractions:
            raise KeyError(abstraction)
        return SeqState(
            tuple(str(sym) for sym in raw), abstraction or self.default_abstractions[0]
        )

    def canon(self, raw) -> tuple:
        return self.parse(raw).key

    def raw_of(self, state: SeqState) -> Symbols:
        return state.symbols

    def frame(self, state: SeqState) -> None:
        return None  # insertions and deletions are legitimate backtracks

    def rebuild(self, template: SeqState, entities: Sequence[SeqEntity]) -> SeqState | None:
        placed: dict[int, str] = {}
        for entity in entities:
            for pos, sym in _positions_of(entity):
                if pos in placed:
                    return None
                placed[pos] = sym
        if sorted(placed) != list(range(len(placed))):
            return None  # a gap: letters must stay contiguous
        return SeqState(tuple(placed[i] for i in range(len(placed))), template.abstraction)

    def candidate_primitives(self, task) -> list[SeqRule]:
        letters = sorted(self.task_values(task))
        selectors = [All(), Rightmost(), Leftmost(), *[ByAttr("letter", letter) for letter in letters]]
        transforms: list[SeqTransform] = [
            SuccessorT(ALPHABET),
            *[SetLetterTo(letter) for letter in letters],
            DeleteT(),
        ]
        return [SeqRule(sel, t) for sel in selectors for t in transforms]

    def task_values(self, task) -> frozenset[str]:
        return frozenset(
            str(sym)
            for pairs in (task.train, task.test)
            for pair in pairs
            for raw in pair
            for sym in raw
        )

    def attr_domain(self, attr: str) -> tuple[str, ...] | None:
        return ALPHABET if attr == "letter" else None

    def fresh_value(self, attr: str, used: Collection) -> str | None:
        if attr != "letter":
            return None
        return next((sym for sym in reversed(ALPHABET) if sym not in used), None)

    def relations(self, state: SeqState) -> set[tuple[str, int, int]]:
        rels: set[tuple[str, int, int]] = set()
        entities = state.objects
        by_position = sorted(entities, key=lambda o: o.attributes["position"])
        for a, b in zip(by_position, by_position[1:]):
            rels.add(("precedes", a.oid, b.oid))
        for a in entities:
            for b in entities:
                if a.oid < b.oid and a.attributes["letter"] == b.attributes["letter"]:
                    rels.add(("same_letter", a.oid, b.oid))
                if a.oid != b.oid:
                    la, lb = a.attributes["letter"], b.attributes["letter"]
                    if la in ALPHABET and ALPHABET.index(la) + 1 < len(ALPHABET):
                        if lb == ALPHABET[ALPHABET.index(la) + 1]:
                            rels.add(("successor_of", a.oid, b.oid))
        return rels

    def overlap(self, a: SeqEntity, b: SeqEntity) -> float:
        return 1.0 if a.extent == b.extent else 0.0

    def make_rule(self, selector, transform) -> SeqRule:
        return SeqRule(selector, transform)

    def candidate_selectors(self, inputs: Sequence[SeqState]) -> list:
        shared = set.intersection(
            *({o.attributes["letter"] for o in s.objects} for s in inputs)
        )
        positive = [All(), *[ByAttr("letter", letter) for letter in sorted(shared)], Rightmost(), Leftmost()]
        negated = [Not(Rightmost()), Not(Leftmost())]
        return positive + negated

    def pair_delta(self, x: SeqEntity, y: SeqEntity) -> SeqDelta:
        return SeqDelta(x, y.attributes["letter"], False)

    def deletion_delta(self, x: SeqEntity) -> SeqDelta:
        return SeqDelta(x, None, True)

    # ------------------------------------------------ optional capabilities

    def probe_perturbations(self, state: SeqState, used: Collection) -> Iterator[Symbols]:
        """Per letter, in position order: delete it, replace it with an unused
        letter, swap it with its right neighbour."""
        fresh = self.fresh_value("letter", used)
        symbols = state.symbols
        for i in range(min(len(symbols), 6)):
            yield symbols[:i] + symbols[i + 1 :]
            if fresh is not None:
                yield symbols[:i] + (fresh,) + symbols[i + 1 :]
            if i + 1 < len(symbols):
                swapped = list(symbols)
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                yield tuple(swapped)

    def addition_values(self, state: SeqState, task) -> list[str]:
        fresh = self.fresh_value("letter", set(state.symbols) | set(self.task_values(task)))
        return [fresh] if fresh is not None else []

    def addition_catalogue(
        self, state: SeqState, max_size: int, separated: bool, values: Sequence[str]
    ) -> Iterator[Addition]:
        """Pertinent-negative additions: one letter appended at the end — the
        only addition leaving every original letter's position intact.
        ``max_size``/``separated`` have no sequence analogue."""
        del max_size, separated
        for value in values:
            yield Addition(
                raw=state.symbols + (value,),
                phrase=f"a letter {value!r} stood at the end",
                size=1,
                group="end",
            )

    def plausible(self, state: SeqState) -> bool:
        """Constraint-consistency certificate: every symbol is a letter of the
        standard alphabet."""
        return all(sym in ALPHABET for sym in state.symbols)

    def render_raw(self, raw, caption: str | None = None, diff_against=None) -> str:
        from ..viz import _esc

        cells = "".join(
            f'<td style="background:#E6E6E6"><code>{_esc(sym)}</code></td>' for sym in raw
        )
        table = f'<table class="g" style="--s:22px"><tr>{cells}</tr></table>'
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return f'<figure class="grid">{table}{cap}</figure>'

    def render_html(self, state: SeqState, caption: str | None = None) -> str:
        if caption is None:
            caption = f"{state.abstraction} — {len(state.objects)} entit(y/ies)"
        return self.render_raw(state.symbols, caption)

    def render_key(self, key, caption: str | None = None) -> str:
        return self.render_raw(key[1], caption)


SEQUENCE = register(SequenceRepresentation())


def letters_task(
    train: Sequence[tuple[str, str]],
    test: Sequence[tuple[str, str]],
    task_id: str = "letters",
):
    """A letter-string analogy task: string pairs on the sequence backend."""
    from ..engine import Task

    def pair(a: str, b: str) -> tuple[Symbols, Symbols]:
        return tuple(a), tuple(b)

    return Task(
        train=tuple(pair(a, b) for a, b in train),
        test=tuple(pair(a, b) for a, b in test),
        task_id=task_id,
        representation="sequence",
    )
