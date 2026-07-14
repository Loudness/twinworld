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

from .engine import ApplyCache, Program, Task
from .representation import Grid, Obj, as_grid, parse_grid

MAX_PROBE_OBJECTS = 6  # per train input; keeps the probe set small and readable


def probes(task: Task, abstraction: str = "cc4") -> list[Grid]:
    """Deterministic probe inputs: the train inputs plus object-level variants."""
    out: list[Grid] = []
    seen: set[Grid] = set()
    used = task.colours()
    fresh = next((c for c in range(9, -1, -1) if c not in used), 9)
    for grid_in, _ in task.train:
        state = parse_grid(grid_in, abstraction)
        base = state.grid
        _add(out, seen, base)
        for o in state.objects[:MAX_PROBE_OBJECTS]:
            _add(out, seen, _paint(base, o, state.background))  # delete it
            _add(out, seen, _paint(base, o, fresh))  # recolour to unused colour
            for dr, dc in ((0, 1), (1, 0)):  # nudge right / down
                _add(out, seen, _nudge(base, o, dr, dc, state.background))
    return out


def _add(out: list[Grid], seen: set[Grid], grid: Grid | None) -> None:
    if grid is not None and grid not in seen:
        seen.add(grid)
        out.append(grid)


def _paint(grid: Grid, obj: Obj, colour: int) -> Grid:
    rows = [list(r) for r in grid]
    for r, c in obj.cells:
        rows[r][c] = colour
    return as_grid(rows)


def _nudge(grid: Grid, obj: Obj, dr: int, dc: int, background: int) -> Grid | None:
    h, w = len(grid), len(grid[0])
    targets = {(r + dr, c + dc) for r, c in obj.cells}
    if not all(0 <= r < h and 0 <= c < w for r, c in targets):
        return None
    if any((r, c) not in obj.cells and grid[r][c] != background for r, c in targets):
        return None
    rows = [list(r) for r in grid]
    for r, c in obj.cells:
        rows[r][c] = background
    for r, c, colour in obj.pixels:
        rows[r + dr][c + dc] = colour
    return as_grid(rows)


def signature(
    program: Program, probe_grids: list[Grid], abstraction: str, cache: ApplyCache
) -> tuple:
    """The program's behaviour fingerprint: its output (or None) on every probe."""
    sig = []
    for grid in probe_grids:
        trace = cache.run(parse_grid(grid, abstraction), program)
        sig.append(trace.outcome.key if trace is not None else None)
    return tuple(sig)


@dataclass(frozen=True)
class DiscriminationReport:
    classes: tuple[tuple[Program, ...], ...]  # behavioural equivalence classes
    probe: Grid | None  # first probe separating the first two classes
    outputs: tuple[Grid | None, ...]  # each class's output on that probe

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


def diagnose(task: Task, programs: list[Program], abstraction: str = "cc4") -> DiscriminationReport:
    """Group train-fitting programs into probe-equivalence classes."""
    probe_grids = probes(task, abstraction)
    cache = ApplyCache()
    groups: dict[tuple, list[Program]] = {}
    for program in programs:
        groups.setdefault(signature(tuple(program), probe_grids, abstraction, cache), []).append(
            tuple(program)
        )
    signatures = list(groups)
    classes = tuple(tuple(groups[s]) for s in signatures)
    if len(classes) < 2:
        return DiscriminationReport(classes, None, ())
    first, second = signatures[0], signatures[1]
    idx = next(i for i in range(len(probe_grids)) if first[i] != second[i])
    return DiscriminationReport(classes, probe_grids[idx], tuple(s[idx] for s in signatures))
