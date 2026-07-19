"""Underdetermination diagnosis via counterfactual probes (thesis Experiment 3).

Programs that all fit the demonstrations are indistinguishable on the training
data *by construction*; they can only be told apart counterfactually. Probes
are deterministic object-level perturbations of the train inputs (delete an
object, recolour it to a task-unused colour, nudge it); two programs are
behaviourally equivalent iff they agree on every probe. More than one
equivalence class means the task is underdetermined in the current mechanism
language — Bober-Irizar & Banerjee's failure class 3 — and the first
disagreeing probe is exactly the input on which the competing hypotheses part
ways.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Raw, get_representation, representation_of
from .engine import ApplyCache, Program, Task
from .representation import Grid


def probes(task: Task, abstraction: str | None = None) -> list[Raw]:
    """Deterministic probe inputs: the train inputs plus the backend's
    object-level perturbations of them (grid: delete / recolour-fresh / nudge)."""
    rep = representation_of(task)
    abstraction = abstraction or rep.default_abstractions[0]
    out: list[Raw] = []
    seen: set[Raw] = set()
    used = rep.task_values(task)
    for raw_in, _ in task.train:
        state = rep.parse(raw_in, abstraction)
        _add(out, seen, rep.raw_of(state))
        for perturbed in rep.probe_perturbations(state, used):
            _add(out, seen, perturbed)
    return out


def _add(out: list[Raw], seen: set[Raw], raw: Raw | None) -> None:
    if raw is not None and raw not in seen:
        seen.add(raw)
        out.append(raw)


def signature(
    program: Program, probe_grids: list[Grid], abstraction: str, cache: ApplyCache, rep=None
) -> tuple:
    """The program's behaviour fingerprint: its output (or None) on every probe."""
    rep = rep if rep is not None else get_representation("grid")
    sig = []
    for grid in probe_grids:
        trace = cache.run(rep.parse(grid, abstraction), program)
        sig.append(trace.outcome.key if trace is not None else None)
    return tuple(sig)


@dataclass(frozen=True)
class DiscriminationReport:
    classes: tuple[tuple[Program, ...], ...]  # behavioural equivalence classes
    probe: Grid | None  # first probe separating the first two classes
    outputs: tuple[Grid | None, ...]  # each class's output on that probe
    signatures: tuple[tuple, ...] = ()  # full per-class probe fingerprints

    @property
    def underdetermined(self) -> bool:
        return len(self.classes) > 1

    def __str__(self) -> str:
        if not self.underdetermined:
            return (
                f"1 behavioural class among {sum(len(c) for c in self.classes)} fitting "
                f"program(s): the demonstrations determine the behaviour (within probes)"
            )
        return (
            f"{len(self.classes)} behavioural classes — UNDERDETERMINED: the "
            f"demonstrations do not fix the behaviour; the classes part ways on the "
            f"reported probe"
        )


def diagnose(
    task: Task, programs: list[Program], abstraction: str | None = None
) -> DiscriminationReport:
    """Group train-fitting programs into probe-equivalence classes."""
    rep = representation_of(task)
    abstraction = abstraction or rep.default_abstractions[0]
    probe_grids = probes(task, abstraction)
    cache = ApplyCache()
    groups: dict[tuple, list[Program]] = {}
    for program in programs:
        groups.setdefault(
            signature(tuple(program), probe_grids, abstraction, cache, rep), []
        ).append(tuple(program))
    signatures = list(groups)
    classes = tuple(tuple(groups[s]) for s in signatures)
    if len(classes) < 2:
        return DiscriminationReport(classes, None, (), tuple(signatures))
    first, second = signatures[0], signatures[1]
    idx = next(i for i in range(len(probe_grids)) if first[i] != second[i])
    return DiscriminationReport(
        classes, probe_grids[idx], tuple(s[idx] for s in signatures), tuple(signatures)
    )
