"""Getting started 3/7 — assess() and predict(): the solver knows when it
doesn't know.

Several different programs can fit the same demonstrations (a Rashomon set).
assess() groups every fitting program into behavioural classes via
counterfactual probes; predict() returns the test answer only when the
classes agree — and honestly ABSTAINS (returns None) when they don't.

Assumes 01_pipeline.py (model and the Task constructor).

Run:  python examples/getting_started/03_confidence.py
"""

import twinworld
from twinworld import Task, as_grid


def g(*rows):
    return as_grid([[int(ch) for ch in row] for row in rows])


def line(row):
    return "".join("·" if c == 0 else str(c) for c in row)


def show(grid, indent="    "):
    for row in grid:
        print(indent + line(row))


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# --------------------------------------------------------- confidence HIGH
rule("HIGH confidence  —  the demonstrations pin the behaviour down")
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
report = twinworld.assess(rep)
print(f"\n{report}")
prediction, _ = twinworld.predict(rep)
print("\nprediction for test[0]:")
show(prediction[0])

# ---------------------------------------------------------- confidence LOW
rule("LOW confidence  —  the test input separates rival readings: ABSTAIN")
# In every train pair the largest object IS the colour-2 object, so
# "recolour the largest to 5" and "recolour colour-2 to 5" both fit. The
# TEST input finally separates them: there the largest object is colour 3.
treacherous = Task(
    train=(
        (g("220000", "000000", "000300"), g("550000", "000000", "000300")),
        (g("000000", "002200", "300000"), g("000000", "005500", "300000")),
    ),
    test=(
        (g("333000", "000000", "000020"), g("333000", "000000", "000050")),
    ),
    task_id="treacherous",
)
rep = twinworld.model(treacherous)
prediction, report = twinworld.predict(rep)
print(f"\n{report}")
print(f"\nprediction: {prediction}   <- abstention: no answer is justified")

print("\nthe hypotheses part ways on this probe input:")
show(report.probe)
print("\nper-class answers to the test input (not unanimous, so no gate pass):")
for i, outs in enumerate(report.predictions):
    print(f"\n  class {i}:")
    if outs[0] is None:
        print("      (program inapplicable on the test input)")
    else:
        show(outs[0], indent="      ")

rule("done — next: 04_tabular.py, the same pipeline on rows and labels")
