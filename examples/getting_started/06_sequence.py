"""Getting started 6/7 — the sequence backend: my problem is strings.

letters_task() turns before/after strings into a Task; model() induces a
Copycat-style rule ("abc -> abd; ijk -> ?"); Contrastive asks why THIS
transformation rather than another, and PertinentNegative finds the letter
whose absence the rule depends on.

Assumes 01_pipeline.py (the four verbs) and 02_queries.py (the query types).

Run:  python examples/getting_started/06_sequence.py
"""

import twinworld
from twinworld import Contrastive, PertinentNegative
from twinworld.backends.sequence import letters_task


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def word(state):
    return "".join(state.symbols)


# The classic letter-string analogy: abc -> abd, ijk -> ijl, so mno -> ?
task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])

# ------------------------------------------------------------------ model
rule("model()  —  induce the transformation rule")
rep = twinworld.model(task)
sol = rep.solution
print(f"\ninduced rule   : {' ; '.join(map(str, sol.program))}")
print(f"programs tried : {sol.programs_tried}")
print(f"test answer    : mno -> {word(sol.test_traces[0].outcome)}")

# ------------------------------------------------------------- Contrastive
rule("Contrastive  —  why 'abd' and not 'bbc'?")
# The target is the raw payload: a tuple of symbols, not a str.
cfs = twinworld.compute(
    twinworld.identify(rep, Contrastive(tuple("bbc"), on="train[0]", k_max=1))
)
print()
for item in cfs.items:
    print(item.narrative)

# ------------------------------------------------------- PertinentNegative
rule("PertinentNegative  —  which absent letter is load-bearing?")
# 'z' has no successor, so a trailing 'z' would make the rule inapplicable:
pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(on="train[0]")))
print()
for item in pn.items:
    print(item.narrative)

rule("also available")
print("\nRepresentational('runs') — re-segment letters into Copycat runs;"
      "\nsee examples/letter_strings.py for counterfactual alphabets and"
      "\nambiguity diagnosis on this backend")

rule("done — next: 07_graph.py, labelled graphs")
