"""Getting started 4/7 — the tabular backend: my problem is rows and labels.

rows_task() turns labelled feature rows into a Task; model() induces a
deterministic decision list; Backtracking answers the recourse question
("what change to THIS row flips the label?"). Also shown: which queries the
tabular backend does NOT support, and how that surfaces at identify().

Assumes 01_pipeline.py (the four verbs).

Run:  python examples/getting_started/04_tabular.py
"""

import twinworld
from twinworld import Backtracking, IdentificationError, Interventional, PertinentNegative
from twinworld.backends.tabular import LABEL, SetLabelIf, rows_task


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# Each demonstration is (feature_row, label); rows_task builds the Task.
task = rows_task(
    train=[
        ({"income": 60}, "approve"),
        ({"income": 30}, "deny"),
        ({"income": 55}, "approve"),
    ],
    test=[({"income": 40}, "deny")],
    task_id="loan-mini",
)

# ------------------------------------------------------------------ model
rule("model()  —  induce a decision list from the labelled rows")
rep = twinworld.model(task, max_depth=2)
sol = rep.solution
print(f"\ninduced list   : {' ; '.join(map(str, sol.program))}")
print(f"programs tried : {sol.programs_tried}")
print(f"held-out test  : income=40 -> {sol.test_traces[0].outcome.row()[LABEL]!r}")

# ------------------------------------------------------------- Backtracking
rule("Backtracking  —  recourse: what income flips the decision?")
# The edited row is a raw payload with the SAME schema, label unset:
for income in (35, 65):
    edited = {"income": income, LABEL: None}
    cfs = twinworld.compute(twinworld.identify(rep, Backtracking(edited)))
    outcome = cfs.items[0].counterfactual.counterfactual.outcome.row()[LABEL]
    print(f"\n  income={income}  ->  {outcome!r}")
    print(f"  {cfs.items[0].narrative}")

# A row with a new feature is a different frame — rejected at identify():
try:
    twinworld.identify(rep, Backtracking({"income": 65, "age": 40, LABEL: None}))
except IdentificationError as err:
    print(f"\na row with a new feature is rejected:\n    {err}")

# ------------------------------------------------------------ Interventional
rule("Interventional  —  what if a RULE had been different?")
alt = SetLabelIf("income", ">=", 60, "approve")
cfs = twinworld.compute(twinworld.identify(rep, Interventional(step=0, alternative=alt)))
print()
for item in cfs.items:
    print(item.narrative)

# -------------------------------------------------------- capability limits
rule("capability boundaries  —  what tabular does NOT support")
# Not every query type exists in every backend; unsupported queries fail
# loudly at identify() instead of returning something meaningless.
try:
    twinworld.identify(rep, PertinentNegative())
except IdentificationError as err:
    print(f"\nPertinentNegative: {err}")
print("\n(Representational is likewise unavailable: the tabular backend has a"
      "\n single abstraction, so there is no alternative segmentation to try)")

rule("done — next: 05_relational.py, block stacking as plan induction")
