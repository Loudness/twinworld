"""Tabular recourse end-to-end: a decision list as a twinworld program.

Rows are labelled by a LATENT ≤3-rule list; twinworld INDUCES a rule list
from the labelled rows (the decision-tree-as-program story), then the usual
counterfactual suite applies with certificates: recourse what-ifs via
Backtracking, certified-minimal contrastive rule edits with an exact Pareto
front, and Rashomon-set diagnosis (several rule lists fitting the same rows,
told apart by probes).

Deferred by design (docs/beyond-grids.md §5.3): class-REGION contrastive
targets ("any approving row") and minimal input-edit search — Contrastive
today targets one exact outcome row.

Run:  python examples/tabular_recourse.py
"""

import random
import time

import twinworld
from twinworld import Backtracking, Contrastive, pareto_front
from twinworld.backends.tabular import LABEL, SetLabelIf, rows_task
from twinworld.discriminate import diagnose
from twinworld.engine import solve_all


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


TAB = twinworld.get_representation("tabular")

LATENT = (
    SetLabelIf("debt", ">=", 40, "deny"),
    SetLabelIf("income", ">=", 50, "approve"),
    SetLabelIf("income", ">=", 0, "deny"),  # the default rule
)


def label_of(row):
    state = TAB.parse({**row, LABEL: None})
    for step in LATENT:
        state = step.apply(state)
    return state.row()[LABEL]


rng = random.Random(7)
rows = [
    {"income": rng.choice((20, 30, 45, 50, 60, 75)), "debt": rng.choice((5, 20, 45, 60))}
    for _ in range(5)
]
rows.append({"income": 75, "debt": 45})  # pins the order: high income yet DENIED (debt first)
train = [(row, label_of(row)) for row in rows]
test_rows = [{"income": 60, "debt": 45}, {"income": 50, "debt": 60}]
test = [(row, label_of(row)) for row in test_rows]
task = rows_task(train=train, test=test, task_id="loan-toy")

rule("1. model()  —  induce the decision list from labelled rows")
t0 = time.perf_counter()
rep = twinworld.model(task, max_depth=3)
ms = (time.perf_counter() - t0) * 1000
sol = rep.solution
print(f"\nlatent list  : {' ; '.join(map(str, LATENT))}")
print(f"induced list : {' ; '.join(map(str, sol.program))}")
print(f"search       : {sol.programs_tried} rule list(s) tried, {ms:.1f} ms")
correct = sum(
    sol.cache.run(TAB.parse(i), sol.program).outcome.key == TAB.canon(o) for i, o in task.test
)
print(f"held-out     : {correct}/{len(task.test)} test row(s) reproduced (validity certificate)")

rule("2. recourse  —  what change of THIS applicant's row flips the label?")
denied = next((i for i, o in task.test if dict(o)[LABEL] == "deny"), task.test[0][0])
base = dict(denied)
print(f"\napplicant: { {k: v for k, v in base.items() if k != LABEL} } -> {label_of(base)}")
for edit in ({"income": 55}, {"debt": 30, "income": 55}, {"income": 45}):
    candidate = {**{k: v for k, v in base.items() if k != LABEL}, **edit, LABEL: None}
    cfs = twinworld.compute(twinworld.identify(rep, Backtracking(candidate)))
    outcome = cfs.items[0].counterfactual.counterfactual.outcome.row()[LABEL]
    print(f"  edit {str(edit):34s} -> {outcome}")

rule("3. contrastive  —  why deny and not approve? certified rule edits + Pareto")
factual = rep.solution.train_traces[0]
target_row = {**dict(task.train[0][0]), LABEL: "approve"}
try:
    cfs = twinworld.compute(twinworld.identify(rep, Contrastive(tuple(sorted(target_row.items())))))
    front = pareto_front(list(cfs.items))
    print(f"\n{len(cfs.items)} certified minimal edit(s); exact Pareto front keeps {len(front)}:")
    for item in front[:3]:
        print(f"  {item.narrative}")
except twinworld.IdentificationError as err:
    print(f"\n(foil coincides with the factual outcome here: {err})")

rule("4. Rashomon  —  when the rows underdetermine the list, a probe says where")
collision = rows_task(  # income >= 50 and savings >= 10 coincide on every demonstration
    train=[
        ({"income": 60, "savings": 12}, "approve"),
        ({"income": 30, "savings": 3}, "deny"),
    ],
    test=[({"income": 55, "savings": 11}, "approve")],
    task_id="loan-collision",
)
fits = solve_all(collision, TAB.candidate_primitives(collision), max_depth=2, limit=16)
report = diagnose(collision, fits)
print(f"\n{len(fits)} fitting rule list(s) in {len(report.classes)} behavioural class(es)")
print(report)
if report.probe is not None:
    print(f"separating probe row: {dict(report.probe)}")

rule("done — certificates, not estimates, on the recourse crowd's home turf")
