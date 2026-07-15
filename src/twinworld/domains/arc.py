"""ARC domain adapter: the first (and reference) domain plugin.

A domain plugin owes the core exactly two things — a perception adapter
(here: arckit grids → :class:`twinworld.Task`) and a primitive vocabulary
(the core's task-scoped ``candidate_primitives`` for the slice).
Requires the ``arc`` extra: ``pip install twinworld[arc]``.
"""

from __future__ import annotations

from typing import Iterator

from ..api import model
from ..engine import Task, UnsolvedTaskError
from ..representation import as_grid


def _to_task(arckit_task) -> Task:
    return Task(
        train=tuple((as_grid(i), as_grid(o)) for i, o in arckit_task.train),
        test=tuple((as_grid(i), as_grid(o)) for i, o in arckit_task.test),
        task_id=arckit_task.id,
    )


def load_task(task_id: str) -> Task:
    import arckit

    train_set, eval_set = arckit.load_data()
    for ds in (train_set, eval_set):
        try:
            return _to_task(ds[task_id])
        except (KeyError, IndexError):
            continue
    raise KeyError(f"task {task_id!r} not found in ARC training or evaluation set")


def iter_tasks(dataset: str = "train") -> Iterator[Task]:
    import arckit

    train_set, eval_set = arckit.load_data()
    ds = train_set if dataset == "train" else eval_set
    for t in ds:
        yield _to_task(t)


def find_solvable(
    limit: int | None = None, max_depth: int = 2, dataset: str = "train"
) -> Iterator[tuple[Task, object]]:
    """Scan ARC tasks and yield (task, representation) for those the current
    primitive vocabulary can solve under any registered abstraction. The
    vocabulary is small, so expect few hits — each one is an end-to-end
    demonstration candidate."""
    for n, task in enumerate(iter_tasks(dataset)):
        if limit is not None and n >= limit:
            return
        if len(task.colours()) > 6:  # cheap pre-filter: colour-heavy tasks exceed the DSL
            continue
        try:
            yield task, model(task, max_depth=max_depth)
        except UnsolvedTaskError:
            continue
