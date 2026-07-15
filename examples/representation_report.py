"""Representation-level evaluation over the ARC training corpus (Milestone 2).

Correspondence coverage: a train pair is *covered* by an abstraction when every
input object matches an output object and vice versa (no appearances or
disappearances) — i.e. the pair is explainable as pure object transformations
under that segmentation. This measures representation quality independently of
the (still tiny) mechanism vocabulary, and is the precondition for
analogy-driven rule induction in Experiment 2.

Usage:
    python examples/representation_report.py [N_TASKS]          # coverage only
    python examples/representation_report.py [N_TASKS] --solve  # + solve scan
"""

import sys

from twinworld import DEFAULT_ABSTRACTIONS, UnsolvedTaskError, match_objects, parse_grid
from twinworld.api import model
from twinworld.domains.arc import iter_tasks

MAX_OBJECTS = 80  # beyond this the segmentation is too fragmented to be meaningful

args = [a for a in sys.argv[1:] if not a.startswith("--")]
limit = int(args[0]) if args else None
do_solve = "--solve" in sys.argv


def pair_covered(grid_in, grid_out, abstraction) -> bool:
    si, so = parse_grid(grid_in, abstraction), parse_grid(grid_out, abstraction)
    if not (0 < len(si.objects) <= MAX_OBJECTS and 0 < len(so.objects) <= MAX_OBJECTS):
        return False
    return all(x is not None and y is not None for x, y in match_objects(si, so))


covered = {a: 0 for a in DEFAULT_ABSTRACTIONS}
union = 0
total = 0
for task in iter_tasks():
    if limit is not None and total >= limit:
        break
    total += 1
    any_scheme = False
    for a in DEFAULT_ABSTRACTIONS:
        if all(pair_covered(gi, go, a) for gi, go in task.train):
            covered[a] += 1
            any_scheme = True
    union += any_scheme

print(f"\ncorrespondence coverage over {total} ARC training tasks")
print("(all train pairs explainable as object transformations — no object")
print(" appears or disappears under greedy matching)\n")
for a in DEFAULT_ABSTRACTIONS:
    print(f"  {a:>4}: {covered[a]:4d} tasks  ({covered[a] / total:5.1%})")
print(f"  best-of-any: {union:4d} tasks  ({union / total:5.1%})")

if do_solve:
    print("\nsolve scan (multi-abstraction model, depth <= 2) ...", flush=True)
    solved = []
    n = 0
    for task in iter_tasks():
        if limit is not None and n >= limit:
            break
        n += 1
        if len(task.colours()) > 6:
            continue
        try:
            rep = model(task, max_depth=2)
        except UnsolvedTaskError:
            continue
        also = sorted(set(rep.solutions) - {rep.abstraction})
        print(
            f"  SOLVED {task.task_id} under [{rep.abstraction}] "
            f"(also: {also or '-'}): {' ; '.join(map(str, rep.solution.program))}",
            flush=True,
        )
        solved.append(task.task_id)
    print(f"\nsolved {len(solved)}/{n} tasks: {solved}")
