"""End-to-end vertical slice on a real ARC task.

Pipeline: model -> identify -> compute -> refute on ARC task a79310a0
("the azure shape moves down one row and becomes red"), plus a necessity
analysis over every step of the induced program, a backtracking
counterfactual, and the placebo refuter demonstrated on a spectator task.

Run:  python examples/vertical_slice.py
"""

import twinworld
from twinworld import (
    Backtracking,
    IdentificationError,
    Interventional,
    Representational,
    Task,
    as_grid,
)
from twinworld.domains.arc import load_task


def line(row):
    return "".join("·" if c == 0 else str(c) for c in row)


def show(grid, indent="    "):
    for row in grid:
        print(indent + line(row))


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# ----------------------------------------------------------- 1. model(task)
rule("1. model()  —  ARC task a79310a0")
task = load_task("a79310a0")
rep = twinworld.model(task, max_depth=2)
sol = rep.solution

print(f"\ninduced program : {' ; '.join(map(str, sol.program))}")
print(f"strategy        : {sol.strategy}")
print(f"programs tried  : {sol.programs_tried}")
print(f"trajectory DAG  : {sol.dag.number_of_nodes()} states, "
      f"{sol.dag.number_of_edges()} transitions recorded")

test_in, test_out = task.test[0]
solved = sol.test_traces and sol.test_traces[0].outcome.key == as_grid(test_out)
print(f"held-out test   : {'SOLVED' if solved else 'FAILED'}")
print("\ntest input:")
show(test_in)
print("predicted output:")
show(sol.test_traces[0].outcome.grid)

# --------------------------------------- 2+3. identify() and compute() a CF
rule("2+3. identify() + compute()  —  interventional counterfactual")
alt = twinworld.Recolor(8, 3)
print(f"\nquery: what if step 0 had been [{alt}] instead of [{sol.program[0]}]?\n")
cfs = twinworld.compute(twinworld.identify(rep, Interventional(step=0, alternative=alt)))
for item in cfs.items:
    print(item.narrative)
print("\ncounterfactual test output under the intervention:")
cf_test = cfs.items[-1]
if cf_test.counterfactual.applicable:
    show(cf_test.counterfactual.counterfactual.outcome.grid)

# --------------------------------------------------- necessity of each step
rule("necessity analysis  —  does ANY alternative preserve success?")
for step in range(len(sol.program)):
    valid_alts = []
    tested = 0
    for prim in rep.primitives:
        try:
            identified = twinworld.identify(rep, Interventional(step=step, alternative=prim))
        except IdentificationError:
            continue  # the factual mechanism itself
        tested += 1
        result = twinworld.compute(identified)
        if result.items[0].metrics.validity:
            valid_alts.append(prim)
    verdict = (
        "NECESSARY — no alternative mechanism preserves success"
        if not valid_alts
        else f"replaceable by: {', '.join(map(str, valid_alts))}"
    )
    print(f"\nstep {step} [{sol.program[step]}]: tested "
          f"{tested} alternatives -> {verdict}")

# --------------------------------------------- backtracking counterfactual
rule("backtracking  —  what if the input itself had differed?")
factual_trace = sol.train_traces[0]
shifted = twinworld.Translate(0, 1, colour=None).apply(factual_trace.states[0])
if shifted is not None:
    cfs = twinworld.compute(twinworld.identify(rep, Backtracking(shifted.grid)))
    print("\nedit: the input shape starts one column to the right")
    print(cfs.items[0].narrative)
    print("\nfactual outcome:            counterfactual outcome:")
    fo, co = factual_trace.outcome.grid, cfs.items[0].counterfactual.counterfactual.outcome.grid
    for a, b in zip(fo, co):
        print(f"    {line(a)}            {line(b)}")

# --------------------------------------------- contrastive: why not that?
rule("contrastive  —  why this outcome and not that one?")
factual_out = sol.test_traces[0].outcome
azure = twinworld.Recolor(2, 8).apply(factual_out)  # foil: moved but still azure
print("\nfoil A: the shape moves down but KEEPS its azure colour (8)\n")
cfs = twinworld.compute(twinworld.identify(rep, twinworld.Contrastive(azure.grid, on="test[0]")))
for item in cfs.items[:3]:
    print(item.narrative)
if len(cfs.items) > 3:
    print(f"(... {len(cfs.items) - 3} more minimal counterfactuals)")
print(f"\nresponsibility profile (Chockler-Halpern): {cfs.responsibility}")

extra = [list(r) for r in factual_out.grid]
extra[0][0] = 2  # foil: same outcome plus one impossible extra pixel
print("\nfoil B: the same outcome plus one extra red pixel at (0,0)\n")
cfs_b = twinworld.compute(twinworld.identify(rep, twinworld.Contrastive(as_grid(extra), on="test[0]")))
print(cfs_b.items[0].narrative)

# ----------------------------------------- pertinent negatives (Experiment 4)
rule("pertinent negatives  —  what ABSENCE is load-bearing?")
pn = twinworld.compute(twinworld.identify(rep, twinworld.PertinentNegative(on="test[0]", max_cells=1)))
print()
for item in pn.items[:3]:
    print(item.narrative)
if len(pn.items) > 3:
    print(f"(... {len(pn.items) - 3} more witnesses)")

# --------------------------------------- counterfactual re-segmentation
rule("re-segmentation  —  what if the objects had been carved differently?")
print(f"\nchosen abstraction: [{rep.abstraction}]; "
      f"also solved under: {sorted(set(rep.solutions) - {rep.abstraction}) or '-'}")
for alt_abstraction in twinworld.DEFAULT_ABSTRACTIONS:
    if alt_abstraction == rep.abstraction:
        continue
    resegmented = twinworld.compute(twinworld.identify(rep, Representational(alt_abstraction)))
    print(resegmented.items[0].narrative)

# ------------------------------------------------------------- 4. refute()
rule("4. refute()  —  refutation battery")
print(f"\n[ARC a79310a0]\n{twinworld.refute(rep)}")

spectator_task = Task(
    train=(
        (as_grid([[0, 3, 0], [0, 0, 0], [5, 0, 0]]), as_grid([[0, 4, 0], [0, 0, 0], [5, 0, 0]])),
        (as_grid([[3, 0, 3], [0, 5, 0], [0, 0, 0]]), as_grid([[4, 0, 4], [0, 5, 0], [0, 0, 0]])),
    ),
    test=((as_grid([[0, 0, 3], [5, 0, 0], [0, 0, 0]]), as_grid([[0, 0, 4], [5, 0, 0], [0, 0, 0]])),),
    task_id="spectator-demo",
)
print(f"\n[spectator task — placebo has a target]\n{twinworld.refute(twinworld.model(spectator_task, max_depth=1))}")

rule("done — the full pipeline ran: model -> identify -> compute -> refute")
