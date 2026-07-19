"""Getting started 5/7 — the relational backend: my problem is stacks and plans.

task_from_towers() turns before/after block configurations into a Task;
model() induces a PLAN (a sequence of MoveBlock steps) that explains every
demonstration; Interventional asks what a different move would have done,
and refute() attacks the plan with a native placebo.

Assumes 01_pipeline.py (the four verbs).

Run:  python examples/getting_started/05_relational.py
"""

import twinworld
from twinworld import Interventional
from twinworld.domains.blocks import MoveBlock, task_from_towers


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def show_towers(towers, indent="    "):
    print(indent + " | ".join(str(list(t)) if t else "[]" for t in towers))


# Each pair is (before, after); a configuration is a list of columns, each
# listing its blocks bottom -> top. Here: unstack — 2 goes to column 1,
# then 1 goes to column 2. Block 3 (and 5 in the test) is a spectator.
task = task_from_towers(
    train=[
        ([[1, 2], [], []], [[], [2], [1]]),
        ([[1, 2], [3], []], [[], [3, 2], [1]]),
    ],
    test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
)

# ------------------------------------------------------------------ model
rule("model()  —  plan induction over MoveBlock primitives")
# The backend supplies candidate moves automatically; induction='never'
# skips the analogy stage, which has nothing to propose for plans.
rep = twinworld.model(task, max_depth=2, induction="never")
sol = rep.solution
print(f"\ninduced plan   : {' ; '.join(map(str, sol.program))}")
print(f"programs tried : {sol.programs_tried}")
print("\ntest instance (columns, bottom->top):")
show_towers(task.test[0][0])
print("plan result:")
show_towers(sol.test_traces[0].outcome.towers)

# ------------------------------------------------------------ Interventional
rule("Interventional  —  what if the first move had gone elsewhere?")
alt = MoveBlock(2, 2)  # move block 2 to column 2 instead of column 1
cfs = twinworld.compute(twinworld.identify(rep, Interventional(step=0, alternative=alt)))
print()
for item in cfs.items:
    print(item.narrative)
cf = cfs.items[0].counterfactual
if cf.applicable:
    print("\ncounterfactual end state of train[0]:")
    show_towers(cf.counterfactual.outcome.towers)

# ------------------------------------------------------------------ refute
rule("refute()  —  the placebo works natively on blocks too")
# The placebo perturbs a spectator entity (block 3, never moved by the plan)
# and expects the rest of the outcome to be unchanged.
print(f"\n{twinworld.refute(rep)}")

rule("done — next: 06_sequence.py, letter-string analogies")
