"""Selection among underdetermined candidate programs (thesis Experiment 3).

Failure class 3 (Bober-Irizar & Banerjee, arXiv:2402.03507) is the case where
several programs fit the demonstrations and disagree on the test input.
:func:`dowhat.predict` answers it with calibrated ABSTENTION; this module is
the experimental instrument that asks whether one can do better than abstain:
passive POLICIES choose a behavioural class without new information, and
:func:`resolve_with_probe` spends oracle queries on the diagnosing probe —
counterfactual discrimination made actionable. On a symmetric collision no
passive policy can exceed the latent prior by construction; the measurable
claims live in skewed-prior worlds and in the active arm
(examples/discrimination_report.py). The two counterfactual policies
operationalize earlier findings: probe stability (a class that keeps applying
under perturbation) and pertinent-negative fragility (Dhurandhar et al. 2018;
size-based readings carry absence dependencies that colour-based ones lack).
"""

from __future__ import annotations

import random
from typing import Callable

from .api import (
    ConfidenceReport,
    CausalRepresentation,
    PertinentNegative,
    assess,
    compute,
    identify,
    model,
)
from .discriminate import diagnose, probes
from .engine import ApplyCache, Program, Task
from .representation import Grid, parse_grid


def _classes(report: ConfidenceReport) -> tuple[tuple[Program, ...], ...]:
    assert report.discrimination is not None
    return report.discrimination.classes


def _first(rep, report, rng) -> int:
    return 0


def _random(rep, report, rng) -> int:
    return rng.randrange(len(_classes(report)))


def _shortest(rep, report, rng) -> int:
    lengths = [min(len(p) for p in cls) for cls in _classes(report)]
    return lengths.index(min(lengths))


def _largest_class(rep, report, rng) -> int:
    sizes = [len(cls) for cls in _classes(report)]
    return sizes.index(max(sizes))


def _probe_stability(rep, report, rng) -> int:
    """Prefer the class that keeps APPLYING under counterfactual perturbation
    (fewest None entries in its probe signature)."""
    sigs = report.discrimination.signatures
    counts = [sum(1 for out in sig if out is not None) for sig in sigs]
    return counts.index(max(counts))


def _fewest_absences(rep, report, rng) -> int:
    """Prefer the class whose representative has the fewest pertinent-negative
    witnesses: hypotheses with absence dependencies are fragile (the M5
    finding, used as a selection criterion)."""
    counts = []
    for cls in _classes(report):
        program = cls[0]
        refit = model(
            rep.task,
            abstractions=(rep.abstraction,),
            primitives=list(program),
            induction="never",
            max_depth=max(1, len(program)),
        )
        pn = compute(identify(refit, PertinentNegative(on="train[0]", max_cells=3)))
        counts.append(sum("load-bearing" in item.narrative for item in pn.items))
    return counts.index(min(counts))


POLICIES: dict[str, Callable] = {
    "first": _first,
    "random": _random,
    "shortest": _shortest,
    "largest_class": _largest_class,
    "probe_stability": _probe_stability,
    "fewest_absences": _fewest_absences,
}


def select(
    rep: CausalRepresentation,
    policy: str = "probe_stability",
    rng: random.Random | None = None,
) -> tuple[tuple[Grid | None, ...], ConfidenceReport]:
    """Like :func:`dowhat.predict`, but on LOW confidence a policy picks a
    behavioural class instead of abstaining. The gate's calibrated abstention
    stays the library default; this is the experiment."""
    report = assess(rep)
    if report.confidence == "high" or not report.underdetermined:
        return report.predictions[0], report
    index = POLICIES[policy](rep, report, rng if rng is not None else random.Random(0))
    return report.predictions[index], report


def resolve_with_probe(
    task: Task,
    programs: list[Program],
    oracle: Program,
    abstraction: str = "cc4",
    max_queries: int = 2,
) -> tuple[Program, int]:
    """Active counterfactual discrimination: answer the diagnosing probe with
    the oracle, keep the classes consistent with the answer, repeat. Returns
    (a representative of the surviving first class, queries actually spent)."""
    cache = ApplyCache()
    pool = [tuple(p) for p in programs]
    probe_grids = probes(task, abstraction)
    queries = 0
    while queries < max_queries:
        report = diagnose(task, pool, abstraction)
        if not report.underdetermined or report.probe is None:
            break
        index = probe_grids.index(report.probe)
        trace = cache.run(parse_grid(report.probe, abstraction), tuple(oracle))
        answer = trace.outcome.key if trace is not None else None
        queries += 1
        survivors = [
            cls
            for cls, sig in zip(report.classes, report.signatures)
            if sig[index] == answer
        ]
        if not survivors:
            break  # the oracle lies outside every fitted class; keep the pool
        pool = [p for cls in survivors for p in cls]
    return diagnose(task, pool, abstraction).classes[0][0], queries
