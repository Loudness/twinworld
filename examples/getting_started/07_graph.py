"""Getting started 7/7 — the graph backend: my problem is a labelled graph.

graph_task() turns before/after labelled graphs into a Task; model() induces
a motif rule ("label every triangle node 1"); Backtracking deletes an edge
and reruns the same rule — CF-GNNExplainer's question, answered with
certificates instead of gradient estimates.

Assumes 01_pipeline.py (the four verbs).

Run:  python examples/getting_started/07_graph.py
"""

import twinworld
from twinworld import Backtracking
from twinworld.backends.graph import graph_task


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


# A raw graph is (nodes, edges): nodes as (id, label) pairs, edges as
# normalized (min, max) pairs. Two triangles share node 3; every node on a
# triangle gets label 1.
NODES = ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0))
EDGES = ((1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5))
OUT = (((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)), EDGES)

task = graph_task(
    train=[((NODES, EDGES), OUT)],
    test=[((NODES, EDGES), OUT)],
    task_id="two-triangles",
)

# ------------------------------------------------------------------ model
rule("model()  —  induce a motif rule")
rep = twinworld.model(task, max_depth=1)
sol = rep.solution
print(f"\ninduced program : {' ; '.join(map(str, sol.program))}")
print(f"programs tried  : {sol.programs_tried}")
print(f"factual labels  : {sol.train_traces[0].outcome.labels()}")

# ------------------------------------------------------------- Backtracking
rule("Backtracking  —  which edge made this node's label flip?")
# Delete edge (1,2): the triangle 1-2-3 is gone, but 3-4-5 survives.
edited = (NODES, tuple(e for e in EDGES if e != (1, 2)))
cfs = twinworld.compute(twinworld.identify(rep, Backtracking(edited)))
cf = cfs.items[0].counterfactual
print(f"\nedit: delete edge (1, 2)\n{cfs.items[0].narrative}")
print(f"counterfactual labels: {cf.counterfactual.outcome.labels()}")
print("-> nodes 1 and 2 flip to 0 (their triangle is gone); 3, 4, 5 keep label 1")

rule("Backtracking that breaks EVERY triangle  —  a different, honest answer")
# Keep only the two triangle-closing edges away: no triangle remains, so the
# rule has nothing to apply to. That is reported, not papered over.
edited = (NODES, ((1, 2), (2, 3), (3, 4), (4, 5)))  # a path: no triangles
cfs = twinworld.compute(twinworld.identify(rep, Backtracking(edited)))
cf = cfs.items[0].counterfactual
print(f"\nedit: keep only the path 1-2-3-4-5\napplicable: {cf.applicable}")
print(cfs.items[0].narrative)

rule("done — that was the tour: one core, five backends."
     "\nTo bring your OWN domain, see 'Using twinworld in your own domain'"
     "\nin the README")
