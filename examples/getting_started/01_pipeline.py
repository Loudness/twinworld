"""Getting started 1/7 — the four verbs: model, identify, compute, refute.

A five-line grid task ("every colour-3 object becomes colour 4") is enough to
run the whole pipeline: induce a program from demonstrations, pose an
interventional and a backtracking counterfactual, and attack the explanation
with the refutation battery. Later tutorials build on the vocabulary
introduced here.

Run:  python examples/getting_started/01_pipeline.py
"""

import twinworld
from twinworld import (
    Backtracking,
    IdentificationError,
    Interventional,
    Recolor,
    Task,
    as_grid,
)


def g(*rows):
    """Compact grid literal: '00300' -> row of ints."""
    return as_grid([[int(ch) for ch in row] for row in rows])


def line(row):
    return "".join("·" if c == 0 else str(c) for c in row)


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# A Task is just train/test pairs of raw inputs and outputs. Here: colour-3
# objects become colour 4; the colour-5 object is a spectator, never touched.
task = Task(
    train=(
        (
            g("00000", "03300", "03000", "00050", "00000"),
            g("00000", "04400", "04000", "00050", "00000"),
        ),
        (
            g("33300", "00000", "05000", "00003", "00000"),
            g("44400", "00000", "05000", "00004", "00000"),
        ),
    ),
    test=(
        (
            g("00000", "00300", "00305", "00000", "00000"),
            g("00000", "00400", "00405", "00000", "00000"),
        ),
    ),
    task_id="hello-recolor",
)

# ----------------------------------------------------------- 1. model(task)
rule("1. model()  —  induce a program that explains the demonstrations")
rep = twinworld.model(task, max_depth=1)
sol = rep.solution

print(f"\ninduced program : {' ; '.join(map(str, sol.program))}")
print(f"strategy        : {sol.strategy}")
print(f"abstraction     : {rep.abstraction}  (how the grid was carved into objects)")
print(f"programs tried  : {sol.programs_tried}")
print("\ntest input:                 predicted output:")
for a, b in zip(task.test[0][0], sol.test_traces[0].outcome.grid):
    print(f"    {line(a)}                   {line(b)}")

# ---------------------------------------------------------- 2. identify(rep, query)
rule("2. identify()  —  is the question well-posed? (no computation yet)")
query = Interventional(step=0, alternative=Recolor(3, 9))
identified = twinworld.identify(rep, query)
print(f"\nquery: what if step 0 had been [{query.alternative}] "
      f"instead of [{sol.program[0]}]?  ->  identified, mode {identified.mode!r}")

# An ill-posed query fails HERE, before any counterfactual is computed:
try:
    twinworld.identify(rep, Interventional(step=0, alternative=sol.program[0]))
except IdentificationError as err:
    print(f"asking for the factual mechanism itself is rejected:\n    {err}")

# ----------------------------------------------------------- 3. compute(identified)
rule("3. compute()  —  fork a twin world per trace and rerun")
cfs = twinworld.compute(identified)
print()
for item in cfs.items:  # one CounterfactualItem per train + test trace
    print(item.narrative)
m = cfs.items[0].metrics
print(f"\nmetrics of the first item: validity={m.validity} "
      f"(is the task still solved?), sparsity={m.sparsity} (program edits)")

# ------------------------------------------------- backtracking counterfactual
rule("3b. Backtracking  —  what if the INPUT had differed, same laws?")
edited = g("00000", "03300", "03000", "00000", "00050")  # spectator moved down
cfs = twinworld.compute(twinworld.identify(rep, Backtracking(edited)))
print()
print(cfs.items[0].narrative)
print("\nfactual outcome:            counterfactual outcome:")
factual = sol.train_traces[0].outcome.grid
counterfactual = cfs.items[0].counterfactual.counterfactual.outcome.grid
for a, b in zip(factual, counterfactual):
    print(f"    {line(a)}                   {line(b)}")

# The edited input must live in the same frame (here: the same 5x5 grid):
try:
    twinworld.identify(rep, Backtracking(g("33", "00")))
except IdentificationError as err:
    print(f"\na differently-shaped input is rejected at identify():\n    {err}")

# ------------------------------------------------------------- 4. refute(rep)
rule("4. refute()  —  attack the explanation (passing is necessary, not sufficient)")
report = twinworld.refute(rep)
print(f"\n{report}")
print("\n(the placebo refuter perturbs the spectator and expects no change;"
      "\n SKIPped rows — e.g. the ASP cross-check without clingo — count as"
      "\n not-failed, so report.passed treats them as passing)")

rule("done — next: 02_queries.py for contrastive, pertinent-negative and"
     "\nrepresentational queries")
