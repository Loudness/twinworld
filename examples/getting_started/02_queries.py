"""Getting started 2/7 — the other three query types.

Contrastive ("why this outcome and not that one?"), PertinentNegative ("what
absence is load-bearing?") and Representational ("what if the world had been
segmented differently?"), each on a tiny grid task. The recurring theme: when
no counterfactual exists, the answer is a CERTIFICATE, not silence.

Assumes 01_pipeline.py (model/identify/compute and the recolor task).

Run:  python examples/getting_started/02_queries.py
"""

import twinworld
from twinworld import Contrastive, PertinentNegative, Representational, Task, as_grid


def g(*rows):
    return as_grid([[int(ch) for ch in row] for row in rows])


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# The recolor task from tutorial 01: colour-3 objects become colour 4.
recolor = Task(
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
rep = twinworld.model(recolor, max_depth=1)

# ------------------------------------------------------------- Contrastive
rule("Contrastive  —  why colour 4, and not colour 5?")
# The target is a raw output grid; the answer is the smallest set of program
# edits reaching it — certified minimal, because the search per edit-count k
# is exhaustive over the candidate pool.
foil = g("00000", "05500", "05000", "00050", "00000")  # train[0] recoloured to 5
cfs = twinworld.compute(twinworld.identify(rep, Contrastive(foil, on="train[0]", k_max=1)))
print()
for item in cfs.items:
    print(item.narrative)
print(f"\nresponsibility profile (Chockler-Halpern): {cfs.responsibility}")

rule("Contrastive with an UNREACHABLE foil  —  a robustness certificate")
# One extra pixel makes the target expressible by no program in the pool:
unreachable = [list(r) for r in recolor.train[0][1]]
unreachable[4][4] = 7
cfs = twinworld.compute(
    twinworld.identify(rep, Contrastive(as_grid(unreachable), on="train[0]", k_max=1))
)
print(f"\n{cfs.items[0].narrative}")

# ------------------------------------------------------- PertinentNegative
rule("PertinentNegative  —  what ABSENCE is load-bearing?")
# A two-step task: colour-2 objects move right by one AND become colour 6.
# Movement makes room matter, so an added object can break the program.
move_recolor = Task(
    train=(
        (g("0000", "2200", "0000", "0007"), g("0000", "0660", "0000", "0007")),
        (g("2000", "0000", "0700", "0000"), g("0600", "0000", "0700", "0000")),
    ),
    test=((g("0000", "0020", "0700", "0000"), g("0000", "0006", "0700", "0000")),),
    task_id="move-recolor",
)
mv = twinworld.model(move_recolor)
print(f"\ninduced program : {' ; '.join(map(str, mv.solution.program))}\n")
pn = twinworld.compute(twinworld.identify(mv, PertinentNegative(on="train[0]", max_cells=1)))
for item in pn.items:
    print(item.narrative)

rule("PertinentNegative with NO witness  —  a bounded robustness certificate")
# Pure recolouring is indifferent to extra objects, and the query says so:
pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(on="train[0]", max_cells=1)))
print(f"\n{pn.items[0].narrative}")

# ------------------------------------------------------- Representational
rule("Representational  —  what if the objects had been carved differently?")
# Segmentation is a recorded, revisable decision, so it is intervenable too.
# cc4 (4-connected) vs cc8 (8-connected) components:
cfs = twinworld.compute(twinworld.identify(rep, Representational("cc8")))
print(f"\nfactual abstraction [{rep.abstraction}] vs counterfactual [cc8]:")
print(cfs.items[0].narrative)

rule("done — next: 03_confidence.py for the assess/predict gate")
