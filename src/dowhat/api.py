"""The staged API: model → identify → compute → refute.

The procedure mirrors DoWhy's four verbs. Assumptions (abstraction scheme,
primitive vocabulary, induced program) are explicit, inspectable artifacts on
the :class:`CausalRepresentation`; identification is a purely structural check
that fails fast with an explanatory error; refutation is a battery, documented
as necessary-not-sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import metrics as _metrics
from .analogy import induce_rules
from .engine import Counterfactual, Solution, Task, UnsolvedTaskError, backtrack, intervene, solve
from .mechanisms import Mechanism, candidate_primitives
from .refute import RefutationReport, refutation_battery
from .representation import ABSTRACTIONS, Grid, as_grid, infer_background, parse_grid

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


def _fit_abstraction(
    task: Task,
    abstraction: str,
    blind: Sequence[Mechanism],
    max_depth: int,
    induction: str,
    analogy_depth: int,
) -> Solution:
    """Fit one abstraction: analogy-proposed rules first, blind enumeration after.

    Analogy narrows the candidate set enough that deeper search (analogy_depth)
    stays cheap — the Chollet-efficiency argument in code.
    """
    analogy_error: str | None = None
    if induction in ("auto", "always"):
        candidates = induce_rules(task, abstraction)
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
    abstractions: Sequence[str] = DEFAULT_ABSTRACTIONS,
    primitives: Sequence[Mechanism] | None = None,
    max_depth: int = 2,
    induction: str = "auto",
    analogy_depth: int = 3,
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
    if primitives is None:
        primitives = candidate_primitives(
            task.colours(), background=infer_background(as_grid(task.train[0][0]))
        )
    solutions: dict[str, Solution] = {}
    failures: dict[str, str] = {}
    for abstraction in abstractions:
        try:
            solutions[abstraction] = _fit_abstraction(
                task, abstraction, primitives, max_depth, induction, analogy_depth
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
        max_depth, induction, analogy_depth,
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


Query = Interventional | Backtracking | Representational


class IdentificationError(Exception):
    """The query is not well-posed for this representation — and why."""


@dataclass(frozen=True)
class IdentifiedQuery:
    rep: CausalRepresentation
    query: Query
    mode: str


def identify(rep: CausalRepresentation, query: Query) -> IdentifiedQuery:
    """Structural well-posedness check; no counterfactual is computed here."""
    program = rep.solution.program
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
        edited = parse_grid(query.edited_input, rep.abstraction)
        factual_start = rep.solution.train_traces[0].states[0]
        if (edited.height, edited.width) != (factual_start.height, factual_start.width):
            raise IdentificationError(
                "edited input has different dimensions than the factual input; "
                "backtracking counterfactuals hold the world's frame fixed"
            )
        return IdentifiedQuery(rep, query, "backtracking")
    if isinstance(query, Representational):
        if query.abstraction not in ABSTRACTIONS:
            raise IdentificationError(
                f"unknown abstraction {query.abstraction!r}; "
                f"registered: {sorted(ABSTRACTIONS)}"
            )
        if query.abstraction == rep.abstraction:
            raise IdentificationError(
                f"the alternative abstraction equals the factual one "
                f"({rep.abstraction}); the counterfactual would be the factual world"
            )
        return IdentifiedQuery(rep, query, "representational")
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

    def __str__(self) -> str:
        return "\n".join(item.narrative for item in self.items)


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
    else:
        items.append(_resegment(rep, query.abstraction))
    return CounterfactualSet(identified, tuple(items))


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
