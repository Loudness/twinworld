"""The staged API: model → identify → compute → refute.

The procedure mirrors DoWhy's four verbs. Assumptions (abstraction scheme,
primitive vocabulary, induced program) are explicit, inspectable artifacts on
the :class:`CausalRepresentation`; identification is a purely structural check
that fails fast with an explanatory error; refutation is a battery, documented
as necessary-not-sufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Hashable, Sequence

if TYPE_CHECKING:
    from .discriminate import DiscriminationReport

from . import metrics as _metrics
from .analogy import induce_rules
from .backend import Addition, representation_of
from .concepts import ConceptNet
from .engine import (
    Counterfactual,
    Solution,
    Task,
    Trace,
    UnsolvedTaskError,
    backtrack,
    intervene,
    minimal_edits,
    solve,
)
from .mechanisms import Mechanism
from .refute import RefutationReport, refutation_battery
from .representation import Grid

DEFAULT_ABSTRACTIONS = ("cc4", "cc8", "mcc")


# ---------------------------------------------------------------- 1. model()


@dataclass
class CausalRepresentation:
    """Everything the later stages consume, as explicit data.

    ``abstraction`` is the *chosen* segmentation; ``solutions`` holds every
    abstraction that also solved and ``failures`` those that could not — the
    raw material for counterfactual re-segmentation queries.
    """

    task: Task
    abstraction: str
    primitives: tuple[Mechanism, ...]
    solution: Solution
    solutions: dict[str, Solution] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    max_depth: int = 2
    induction: str = "auto"
    analogy_depth: int = 3
    concepts: ConceptNet | None = None
    mapper: str = "sme"


def _fit_abstraction(
    task: Task,
    abstraction: str,
    blind: Sequence[Mechanism],
    max_depth: int,
    induction: str,
    analogy_depth: int,
    concepts: ConceptNet | None = None,
    mapper: str = "sme",
) -> Solution:
    """Fit one abstraction: analogy-proposed rules first, blind enumeration after.

    Analogy narrows the candidate set enough that deeper search (analogy_depth)
    stays cheap — the Chollet-efficiency argument in code.
    """
    if induction == "asp":
        if task.representation != "grid":
            raise UnsolvedTaskError(
                f"the ASP induction backend is grid-only; task uses {task.representation!r}"
            )
        from .asp_solver import asp_solve, solution_from_asp

        result = asp_solve(task, abstraction=abstraction, max_depth=analogy_depth)
        return solution_from_asp(task, abstraction, result)
    analogy_error: str | None = None
    if induction in ("auto", "always"):
        candidates = induce_rules(task, abstraction, concepts=concepts, mapper=mapper)
        if candidates:
            try:
                return solve(
                    task, candidates, max_depth=analogy_depth,
                    abstraction=abstraction, strategy="analogy",
                )
            except UnsolvedTaskError as err:
                analogy_error = str(err)
        else:
            analogy_error = "analogy induced no candidate rules"
        if induction == "always":
            raise UnsolvedTaskError(analogy_error)
    return solve(task, blind, max_depth=max_depth, abstraction=abstraction)


def model(
    task: Task,
    abstractions: Sequence[str] | None = None,
    primitives: Sequence[Mechanism] | None = None,
    max_depth: int = 2,
    induction: str = "auto",
    analogy_depth: int = 3,
    concepts: ConceptNet | None = None,
    mapper: str = "sme",
) -> CausalRepresentation:
    """Parse the task into object graphs, induce the program, record the DAG.

    ``fit`` here is program induction from 2–5 deterministic demonstrations,
    not statistical estimation — the deliberate inversion of DoWhy's setting.
    Per abstraction, analogy-induced object rules are tried first (``induction``:
    "auto" falls back to blind enumeration, "always" never falls back, "never"
    skips analogy); the shortest program across abstractions wins (ties break
    to abstraction order), and both the abstraction choice and the strategy
    that produced the program are recorded as revisable modelling decisions.
    """
    backend = representation_of(task)
    if abstractions is None:
        abstractions = backend.default_abstractions
    if primitives is None:
        primitives = backend.candidate_primitives(task)
    solutions: dict[str, Solution] = {}
    failures: dict[str, str] = {}
    for abstraction in abstractions:
        try:
            solutions[abstraction] = _fit_abstraction(
                task, abstraction, primitives, max_depth, induction, analogy_depth,
                concepts, mapper,
            )
        except UnsolvedTaskError as err:
            failures[abstraction] = str(err)
    if not solutions:
        raise UnsolvedTaskError(
            f"task {task.task_id}: no program exists under any abstraction "
            f"in {tuple(abstractions)} (max_depth={max_depth}, analogy_depth={analogy_depth})"
        )
    chosen = min(
        solutions, key=lambda a: (len(solutions[a].program), list(abstractions).index(a))
    )
    return CausalRepresentation(
        task, chosen, tuple(primitives), solutions[chosen], solutions, failures,
        max_depth, induction, analogy_depth, concepts, mapper,
    )


# ------------------------------------------------------------- 2. identify()


@dataclass(frozen=True)
class Interventional:
    """What if the solver had applied ``alternative`` at ``step``?"""

    step: int
    alternative: Mechanism


@dataclass(frozen=True)
class Backtracking:
    """What if the input grid had been ``edited_input`` (same laws)?"""

    edited_input: Grid


@dataclass(frozen=True)
class Representational:
    """What if the solver had segmented the world under ``abstraction``?

    Counterfactual re-segmentation: representation choice is a modelling
    decision like any other, so it can be intervened on.
    """

    abstraction: str


@dataclass(frozen=True)
class Contrastive:
    """Why is the outcome of ``on`` what it is, rather than ``target_output``?

    Answered by the smallest program-edit set reaching the target (certified
    minimal), or by a certificate that the target is unreachable within
    ``k_max`` edits — the outcome is robust against this foil.
    """

    target_output: Grid
    on: str = "train[0]"
    k_max: int = 2


@dataclass(frozen=True)
class PertinentNegative:
    """What must be ABSENT from the input for the outcome to be what it is?

    CEM's pertinent negatives (Dhurandhar et al. 2018), transplanted from
    classifiers to solving traces: new, separated objects are added to the
    input in increasing footprint; an addition is a witness when some
    ORIGINAL object's outcome image changes or the program stops applying —
    the added object's own image is allowed to appear or move inertly.
    """

    on: str = "train[0]"
    max_cells: int = 3
    max_witnesses: int = 6
    # separated=True keeps additions halo-distant from existing objects — the
    # right default for ARC, where a same-colour neighbour silently fuses into
    # an object. In domains where adjacency is meaningful and the palette is
    # distinct (e.g. blocks world: "what if a block sat ON TOP of this one?"),
    # pass separated=False to probe contact.
    separated: bool = True
    # colours=None probes the task palette plus one fresh colour; pass an
    # explicit tuple to restrict (e.g. only a fresh colour, when palette
    # duplicates would trip uniqueness preconditions everywhere).
    colours: tuple[Hashable, ...] | None = None  # backends read these as generic values


Query = Interventional | Backtracking | Representational | Contrastive | PertinentNegative


class IdentificationError(Exception):
    """The query is not well-posed for this representation — and why."""


def _resolve_trace(rep: CausalRepresentation, on: str) -> Trace:
    m = re.fullmatch(r"(train|test)\[(\d+)\]", on)
    if not m:
        raise IdentificationError(
            f"unknown trace reference {on!r}; use 'train[i]' or 'test[i]'"
        )
    kind, idx = m.group(1), int(m.group(2))
    traces = rep.solution.train_traces if kind == "train" else rep.solution.test_traces
    if idx >= len(traces):
        raise IdentificationError(
            f"{on} does not exist: the solution has {len(traces)} {kind} trace(s)"
        )
    return traces[idx]


@dataclass(frozen=True)
class IdentifiedQuery:
    rep: CausalRepresentation
    query: Query
    mode: str


def identify(rep: CausalRepresentation, query: Query) -> IdentifiedQuery:
    """Structural well-posedness check; no counterfactual is computed here."""
    program = rep.solution.program
    backend = representation_of(rep.task)
    if isinstance(query, Interventional):
        if not 0 <= query.step < len(program):
            raise IdentificationError(
                f"step {query.step} does not exist: the induced program has "
                f"{len(program)} step(s) ({' ; '.join(map(str, program))})"
            )
        if query.alternative == program[query.step]:
            raise IdentificationError(
                f"the alternative at step {query.step} equals the factual mechanism "
                f"({program[query.step]}); the counterfactual would be the factual world"
            )
        return IdentifiedQuery(rep, query, "interventional")
    if isinstance(query, Backtracking):
        edited = backend.parse(query.edited_input, rep.abstraction)
        factual_start = rep.solution.train_traces[0].states[0]
        if backend.frame(edited) != backend.frame(factual_start):
            raise IdentificationError(
                "edited input has different dimensions than the factual input; "
                "backtracking counterfactuals hold the world's frame fixed"
            )
        return IdentifiedQuery(rep, query, "backtracking")
    if isinstance(query, Representational):
        if query.abstraction not in backend.abstractions:
            raise IdentificationError(
                f"unknown abstraction {query.abstraction!r}; "
                f"registered: {sorted(backend.abstractions)}"
            )
        if query.abstraction == rep.abstraction:
            raise IdentificationError(
                f"the alternative abstraction equals the factual one "
                f"({rep.abstraction}); the counterfactual would be the factual world"
            )
        return IdentifiedQuery(rep, query, "representational")
    if isinstance(query, Contrastive):
        trace = _resolve_trace(rep, query.on)
        if query.k_max < 1:
            raise IdentificationError("k_max must be at least 1")
        if backend.canon(query.target_output) == trace.outcome.key:
            raise IdentificationError(
                f"the target equals the factual outcome of {query.on}; "
                f"the counterfactual would be the factual world"
            )
        return IdentifiedQuery(rep, query, "contrastive")
    if isinstance(query, PertinentNegative):
        _resolve_trace(rep, query.on)
        if query.max_cells < 1:
            raise IdentificationError("max_cells must be at least 1")
        if not hasattr(backend, "addition_catalogue"):
            raise IdentificationError(
                f"pertinent negatives need an addition catalogue; the "
                f"{backend.name} representation does not define one"
            )
        return IdentifiedQuery(rep, query, "pertinent_negative")
    raise IdentificationError(f"unknown query type {type(query).__name__}")


# -------------------------------------------------------------- 3. compute()


@dataclass(frozen=True)
class CounterfactualItem:
    counterfactual: Counterfactual
    metrics: _metrics.MetricVector
    narrative: str


@dataclass(frozen=True)
class CounterfactualSet:
    identified: IdentifiedQuery
    items: tuple[CounterfactualItem, ...]
    responsibility: dict[int, float] | None = None  # per-step, contrastive mode only
    diversity: float | None = None  # mean pairwise outcome distance, ≥2 applicable items

    def __str__(self) -> str:
        return "\n".join(item.narrative for item in self.items)

    def alternatives(self):
        """This set as a uniform :class:`~twinworld.alternatives.Alternatives`
        container — the same ranking/Pareto machinery that serves program
        classes and preimage candidates."""
        from .alternatives import Alternatives

        kind = {"contrastive": "edit_sets", "pertinent_negative": "pn_witnesses"}.get(
            self.identified.mode, self.identified.mode
        )
        scores = tuple(item.metrics.as_dict() for item in self.items)
        return Alternatives(kind, self.items, scores, context=self)


def compute(identified: IdentifiedQuery, backend: str = "cf.twinworld") -> CounterfactualSet:
    """Abduction–action–prediction over the stored traces (twin-world forks)."""
    if backend != "cf.twinworld":
        raise ValueError(f"unknown backend {backend!r}; registered: 'cf.twinworld'")
    rep, query = identified.rep, identified.query
    solution = rep.solution
    items = []
    traces = list(solution.train_traces) + list(solution.test_traces)
    labels = [f"train[{i}]" for i in range(len(solution.train_traces))] + [
        f"test[{i}]" for i in range(len(solution.test_traces))
    ]
    if isinstance(query, Interventional):
        for label, trace in zip(labels, traces):
            cf = intervene(solution, trace, query.step, query.alternative)
            m = _metrics.evaluate(solution, cf)
            items.append(CounterfactualItem(cf, m, _narrate(label, cf, m)))
    elif isinstance(query, Backtracking):
        trace = solution.train_traces[0]
        cf = backtrack(solution, trace, query.edited_input)
        m = _metrics.evaluate(solution, cf)
        items.append(CounterfactualItem(cf, m, _narrate("train[0]", cf, m)))
    elif isinstance(query, Contrastive):
        return _contrast(identified)
    elif isinstance(query, PertinentNegative):
        return _pertinent(identified)
    else:
        items.append(_resegment(rep, query.abstraction))
    return CounterfactualSet(identified, tuple(items))


def _pertinent(identified: IdentifiedQuery) -> CounterfactualSet:
    """Walk the backend's addition catalogue by footprint; report
    certified-minimal witnesses within the declared catalogue, or a bounded
    robustness certificate. One witness per catalogue group (per anchor)."""
    rep, query = identified.rep, identified.query
    backend = representation_of(rep.task)
    trace = _resolve_trace(rep, query.on)
    state = trace.states[0]
    if query.colours is not None:
        values = list(query.colours)
    else:
        values = backend.addition_values(state, rep.task)

    witnesses: list[tuple[Addition, Counterfactual, str]] = []
    current_size: int | None = None
    hit_groups: set = set()
    for add in backend.addition_catalogue(state, query.max_cells, query.separated, values):
        if add.size != current_size:
            if witnesses:
                break  # a smaller footprint already yielded witnesses: minimal
            current_size, hit_groups = add.size, set()
        if add.group in hit_groups:
            continue  # one witness per anchor is plenty
        if len(witnesses) >= query.max_witnesses:
            break
        cf_run = backtrack(rep.solution, trace, add.raw)
        pertinent, detail = _absence_matters(trace, cf_run)
        if pertinent:
            hit_groups.add(add.group)
            cf = Counterfactual(
                "pertinent_negative", trace, cf_run.counterfactual, 0, trace.mechanisms
            )
            witnesses.append((add, cf, detail))
    if witnesses:
        items = []
        for add, cf, detail in witnesses:
            m = _metrics.evaluate(rep.solution, cf)
            items.append(
                CounterfactualItem(
                    cf,
                    m,
                    f"{query.on}: had {add.phrase}, "
                    f"{detail} — its absence is load-bearing ({add.size}-cell addition; "
                    f"minimal within the catalogue, certified).",
                )
            )
        return CounterfactualSet(
            identified, tuple(items), diversity=_diversity_of([cf for _, cf, _ in witnesses])
        )
    cf = Counterfactual("pertinent_negative", trace, None, 0, trace.mechanisms)
    m = _metrics.evaluate(rep.solution, cf)
    narrative = (
        f"{query.on}: no added separated object of up to {query.max_cells} cell(s) in "
        f"{len(values)} value(s) changes any original object's outcome — within this "
        f"catalogue, the outcome depends on no absence (bounded certificate)."
    )
    return CounterfactualSet(identified, (CounterfactualItem(cf, m, narrative),))


def _diversity_of(cfs: list[Counterfactual]) -> float | None:
    """Mean pairwise outcome distance of a returned set; None below two
    applicable members (a singleton has no diversity to report)."""
    if sum(1 for cf in cfs if cf.applicable) < 2:
        return None
    return _metrics.diversity(cfs)


def _absence_matters(trace: Trace, cf_run: Counterfactual) -> tuple[bool, str]:
    if not cf_run.applicable:
        return True, "the program would no longer apply"
    factual_images = {o.extent for o in trace.outcome.objects}
    cf_images = {o.extent for o in cf_run.counterfactual.outcome.objects}
    if factual_images - cf_images:
        return True, "an original object's outcome would change"
    return False, ""


def _contrast(identified: IdentifiedQuery) -> CounterfactualSet:
    """Why X rather than Y: certified minimal program edits reaching the foil."""
    rep, query = identified.rep, identified.query
    trace = _resolve_trace(rep, query.on)
    pool = list(
        dict.fromkeys(
            [
                *induce_rules(rep.task, rep.abstraction, rep.concepts, rep.mapper),
                *rep.primitives,
            ]
        )
    )
    k, cfs = minimal_edits(
        rep.solution, trace, query.target_output, pool, k_max=query.k_max
    )
    if not cfs:
        cf = Counterfactual("contrastive", trace, None, 0, trace.mechanisms)
        m = _metrics.evaluate(rep.solution, cf)
        narrative = (
            f"{query.on}: no program within {query.k_max} edit(s) over {len(pool)} "
            f"mechanisms produces the target — the factual outcome is robust "
            f"against this foil (certified)."
        )
        return CounterfactualSet(identified, (CounterfactualItem(cf, m, narrative),))
    items = []
    for cf in cfs:
        m = _metrics.evaluate(rep.solution, cf)
        edits = sorted(_metrics.edited_steps(cf))
        edit_desc = "; ".join(f"step {t} -> [{cf.program[t]}]" for t in edits)
        items.append(
            CounterfactualItem(
                cf,
                m,
                f"{query.on}: the outcome is the factual one rather than the target "
                f"because of step(s) {edits}; the smallest change producing the "
                f"target is {edit_desc} (k={k}, certified minimal).",
            )
        )
    return CounterfactualSet(
        identified, tuple(items), _metrics.responsibility_profile(cfs), _diversity_of(cfs)
    )


def _resegment(rep: CausalRepresentation, alt_abstraction: str) -> CounterfactualItem:
    """Counterfactual re-segmentation: refit the world under another abstraction.

    Abstractions not attempted during model() are solved lazily and cached on
    the representation.
    """
    alt = rep.solutions.get(alt_abstraction)
    if alt is None and alt_abstraction not in rep.failures:
        try:
            alt = _fit_abstraction(
                rep.task, alt_abstraction, rep.primitives,
                rep.max_depth, rep.induction, rep.analogy_depth,
                rep.concepts, rep.mapper,
            )
            rep.solutions[alt_abstraction] = alt
        except UnsolvedTaskError as err:
            rep.failures[alt_abstraction] = str(err)
    factual_trace = rep.solution.train_traces[0]
    cf = Counterfactual(
        "representational",
        factual_trace,
        alt.train_traces[0] if alt else None,
        0,
        alt.program if alt else (),
    )
    m = _metrics.evaluate(rep.solution, cf)
    if alt is not None:
        same = alt.program == rep.solution.program
        outcome = (
            "the very same program is induced"
            if same
            else f"a program still exists: [{' ; '.join(map(str, alt.program))}]"
        )
        narrative = (
            f"representation: the solver segmented the world under [{rep.abstraction}] "
            f"rather than [{alt_abstraction}]; had it seen [{alt_abstraction}]-objects, "
            f"{outcome} — the explanation is robust to re-segmentation."
        )
    else:
        narrative = (
            f"representation: under [{alt_abstraction}] no program within depth "
            f"{rep.max_depth} exists — the [{rep.abstraction}] segmentation is "
            f"load-bearing for this explanation."
        )
    return CounterfactualItem(cf, m, narrative)


def _narrate(label: str, cf: Counterfactual, m: _metrics.MetricVector) -> str:
    """One contrastive sentence per counterfactual — the human-facing artifact."""
    factual = cf.factual.mechanisms[cf.divergence_step]
    if cf.mode == "interventional":
        alt = cf.program[cf.divergence_step]
        if not cf.applicable:
            return (
                f"{label}: at step {cf.divergence_step} the solver applied [{factual}]; "
                f"[{alt}] would not even apply in that state — the factual choice is "
                f"necessary at this point in the trace."
            )
        verdict = (
            "the task would still be solved — this step is not uniquely necessary"
            if m.validity
            else "the task would NO LONGER be solved"
        )
        return (
            f"{label}: at step {cf.divergence_step} the solver applied [{factual}] "
            f"rather than [{alt}]; had it chosen [{alt}], {verdict} "
            f"(sparsity {m.sparsity} edit, outcome proximity {m.proximity:g})."
        )
    changed = "does not change the outcome" if m.proximity == 0 else (
        f"changes the outcome (proximity {m.proximity:g})"
    )
    return (
        f"{label}: had the input differed as specified, rerunning the same program "
        f"{changed}."
    )


# --------------------------------------------------------------- 4. refute()


def refute(rep: CausalRepresentation) -> RefutationReport:
    """Attack the induced explanation; passing is necessary, not sufficient."""
    return refutation_battery(rep.solution)


# ------------------------------------------------------ 5. assess / predict


@dataclass(frozen=True)
class ConfidenceReport:
    """The solver's own epistemic state, derived counterfactually.

    Hypotheses = every program over the solver's induced candidate language
    (plus the found program's mechanisms) that fits the demonstrations,
    grouped into behavioural classes by counterfactual probes. Confidence is
    HIGH when one class exists, or when all classes agree on the test
    input(s) — unanimity despite ambiguity; LOW otherwise, carrying the
    per-class predictions and the probe on which the hypotheses part ways.
    """

    fits: tuple
    classes: int
    underdetermined: bool
    unanimous_on_test: bool
    confidence: str  # "high" | "low"
    predictions: tuple  # one tuple of output grids (or None) per class
    probe: Grid | None
    discrimination: DiscriminationReport | None = None  # the full class structure

    def __str__(self) -> str:
        return (
            f"confidence {self.confidence.upper()}: {len(self.fits)} fitting program(s) "
            f"in {self.classes} behavioural class(es); "
            + (
                "classes agree on the test input(s)"
                if self.unanimous_on_test
                else "classes DISAGREE on the test input(s)"
                if self.underdetermined
                else "the demonstrations determine the behaviour"
            )
        )


def assess(rep: CausalRepresentation) -> ConfidenceReport:
    """diagnose() as a confidence gate — the solver knows when it doesn't know."""
    from .discriminate import diagnose
    from .engine import ApplyCache, solve_all

    pool = list(
        dict.fromkeys(
            [
                *induce_rules(rep.task, rep.abstraction, rep.concepts, rep.mapper),
                *rep.solution.program,
            ]
        )
    )
    depth = max(1, len(rep.solution.program))
    fits = solve_all(rep.task, pool, max_depth=depth, abstraction=rep.abstraction)
    if tuple(rep.solution.program) not in fits:
        fits = [tuple(rep.solution.program), *fits]
    report = diagnose(rep.task, fits, rep.abstraction)
    cache = ApplyCache()
    backend = representation_of(rep.task)
    test_inputs = [backend.parse(i, rep.abstraction) for i, _ in rep.task.test]
    predictions = []
    for cls in report.classes:
        outs = tuple(
            trace.outcome.key if (trace := cache.run(s, cls[0])) is not None else None
            for s in test_inputs
        )
        predictions.append(outs)
    unanimous = len(set(predictions)) == 1
    confidence = "high" if (not report.underdetermined or unanimous) else "low"
    return ConfidenceReport(
        tuple(fits), len(report.classes), report.underdetermined,
        unanimous, confidence, tuple(predictions), report.probe, report,
    )


def predict(rep: CausalRepresentation) -> tuple[tuple[Grid, ...] | None, ConfidenceReport]:
    """Gated prediction: the test output(s) when confident, None (abstention)
    with the full report — including every class's alternative — when not."""
    report = assess(rep)
    if report.confidence == "high":
        return report.predictions[0], report
    return None, report
