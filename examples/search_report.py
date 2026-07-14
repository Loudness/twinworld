"""Experiment 2 search report over the ARC training corpus.

For every task, the analogy path (structure map -> induced object rules ->
shallow verified search) is attempted under each abstraction; blind
enumeration is used as fallback only where its vocabulary stays tractable
(<= 6 colours). Reports solve rate AND search efficiency — programs tried and
wall-clock — since Chollet's framing scores the efficiency of skill
acquisition, not the skill itself.

Usage: python examples/search_report.py [N_TASKS] [--asp]
       --asp uses the clingo program-induction solver (depth <= 2) instead of
       the analogy-first strategy — completeness within the fragment vs speed.
"""

import statistics
import sys
import time

from dowhat import UnsolvedTaskError, as_grid
from dowhat.api import assess, model
from dowhat.domains.arc import iter_tasks

args = [a for a in sys.argv[1:] if not a.startswith("--")]
limit = int(args[0]) if args else None
use_asp = "--asp" in sys.argv


def test_transfers(task, solution) -> bool:
    """Train fit is not skill: the program must also map the held-out test
    input(s) correctly — failures here are underdetermination made visible."""
    if len(solution.test_traces) != len(task.test):
        return False
    return all(
        trace.outcome.key == as_grid(expected)
        for trace, (_, expected) in zip(solution.test_traces, task.test)
    )


rows = []
n = 0
for task in iter_tasks():
    if limit is not None and n >= limit:
        break
    n += 1
    if use_asp:
        induction = "asp"
    else:
        induction = "auto" if len(task.colours()) <= 6 else "always"
    t0 = time.perf_counter()
    try:
        rep = model(task, induction=induction, analogy_depth=2 if use_asp else 3)
    except UnsolvedTaskError:
        continue
    ms = (time.perf_counter() - t0) * 1000
    sol = rep.solution
    transfers = test_transfers(task, sol)
    gate = assess(rep).confidence
    rows.append((task.task_id, sol.strategy, rep.abstraction, len(sol.program),
                 sol.programs_tried, ms, transfers, gate))
    print(f"  {'SOLVED  ' if transfers else 'TRAIN-FIT ONLY'} {task.task_id} "
          f"[{sol.strategy}/{rep.abstraction}] depth={len(sol.program)} "
          f"tried={sol.programs_tried} {ms:.0f}ms gate={gate}: "
          f"{' ; '.join(map(str, sol.program))}", flush=True)

full = [r for r in rows if r[6]]
print(f"\ntest-verified solved: {len(full)}/{n} tasks "
      f"(+{len(rows) - len(full)} train-fit only — underdetermined)")
gate_table = {
    (gate, ok): sum(1 for r in rows if r[7] == gate and r[6] == ok)
    for gate in ("high", "low")
    for ok in (True, False)
}
print("confidence gate     : "
      f"high&correct {gate_table[('high', True)]}, high&wrong {gate_table[('high', False)]}, "
      f"low&correct {gate_table[('low', True)]}, low&wrong {gate_table[('low', False)]} "
      f"(the gate never sees the test output)")
for strategy in ("analogy", "enumerate"):
    sub = [r for r in rows if r[1] == strategy]
    if sub:
        tried = [r[4] for r in sub]
        ms = [r[5] for r in sub]
        print(f"  {strategy:>9}: {len(sub):3d} fits | median programs tried "
              f"{statistics.median(tried):.0f} | median wall {statistics.median(ms):.0f}ms")
depths = [r[3] for r in rows]
if depths:
    print(f"  program depth: {dict(sorted((d, depths.count(d)) for d in set(depths)))}")