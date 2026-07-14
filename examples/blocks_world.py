"""Blocks world end-to-end: the same core, a different world.

The domain plugin supplies only a perception adapter and the MoveBlock
primitive; representation, engine, counterfactual queries, metrics, refuters
and diagnosis are the unchanged ARC machinery.

Run:  python examples/blocks_world.py
"""

import dowhat
from dowhat import Contrastive, IdentificationError, Interventional, PertinentNegative
from dowhat.discriminate import diagnose
from dowhat.domains.blocks import build_grid, candidate_moves, task_from_towers, towers_of
from dowhat.engine import solve_all


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def show_towers(grid, indent="    "):
    print(indent + " | ".join(str(t) if t else "[]" for t in towers_of(grid)))


task = task_from_towers(
    train=[
        ([[1, 2], [], []], [[], [2], [1]]),
        ([[1, 2], [3], []], [[], [3, 2], [1]]),
    ],
    test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
)

rule("1. model()  —  plan induction over MoveBlock primitives")
rep = dowhat.model(task, primitives=candidate_moves(task), induction="never", max_depth=2)
sol = rep.solution
print(f"\ninduced plan   : {' ; '.join(map(str, sol.program))}")
print(f"programs tried : {sol.programs_tried}")
print("\ntest instance (columns, bottom->top):")
show_towers(task.test[0][0])
print("plan result:")
show_towers(sol.test_traces[0].outcome.grid)

rule("2. contrastive  —  why is block 1 in column 2 and not on top of block 2?")
foil = build_grid([[], [5, 2, 1], []])
cfs = dowhat.compute(dowhat.identify(rep, Contrastive(foil, on="test[0]")))
print()
for item in cfs.items:
    print(item.narrative)
print(f"\nresponsibility profile: {cfs.responsibility}")

rule("3. pertinent negatives  —  what does the plan presuppose is absent?")
pn = dowhat.compute(
    dowhat.identify(
        rep, PertinentNegative(max_cells=1, separated=False, colours=(9,), max_witnesses=8)
    )
)
print()
for item in pn.items[:4]:
    print(item.narrative)
if len(pn.items) > 4:
    print(f"(... {len(pn.items) - 4} more witnesses)")

rule("4. necessity  —  does any alternative move preserve the plan?")
for step in range(len(sol.program)):
    valid, tested = [], 0
    for move in rep.primitives:
        try:
            identified = dowhat.identify(rep, Interventional(step=step, alternative=move))
        except IdentificationError:
            continue
        tested += 1
        if dowhat.compute(identified).items[0].metrics.validity:
            valid.append(move)
    verdict = "NECESSARY" if not valid else f"replaceable by: {', '.join(map(str, valid))}"
    print(f"\nstep {step} [{sol.program[step]}]: tested {tested} alternatives -> {verdict}")

rule("5. diagnose  —  is the plan behaviourally determined?")
fits = solve_all(task, candidate_moves(task), max_depth=2)
report = diagnose(task, fits)
print(f"\n{len(fits)} fitting plan(s); {report}")

rule("done — same pipeline, different world")
