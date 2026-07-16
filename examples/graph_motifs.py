"""Graph motifs end-to-end: a deterministic triangle-closure classifier as a
twinworld program, and CF-GNNExplainer's question — "which edge made this
node's label flip?" — answered by EXHAUSTIVE one-edge sweeps with
certificates instead of gradient estimates.

Run:  python examples/graph_motifs.py
"""

import time

import twinworld
from twinworld import Backtracking, PertinentNegative
from twinworld.backends.graph import graph_task


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


GRAPH = twinworld.get_representation("graph")

NODES = ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0))
EDGES = ((1, 2), (2, 3), (1, 3), (3, 4), (4, 5))  # triangle 1-2-3, tail 4-5
LABELLED = (((1, 1), (2, 1), (3, 1), (4, 0), (5, 0)), EDGES)

OTHER = (((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), ((2, 4), (4, 5), (2, 5), (1, 2)))
OTHER_OUT = (((1, 0), (2, 1), (3, 0), (4, 1), (5, 1)), OTHER[1])

task = graph_task(
    train=[((NODES, EDGES), LABELLED), (OTHER, OTHER_OUT)],
    test=[((NODES, EDGES), LABELLED)],
    task_id="triangle-closure",
)

rule("1. model()  —  one motif rule fits BOTH graphs (context-dependent)")
t0 = time.perf_counter()
rep = twinworld.model(task, max_depth=1)
ms = (time.perf_counter() - t0) * 1000
sol = rep.solution
print(f"\ninduced program : {' ; '.join(map(str, sol.program))}")
print(f"search          : {sol.programs_tried} candidate(s) tried, {ms:.1f} ms")
print(f"test labelling  : {sol.test_traces[0].outcome.labels()}")

rule("2. which SINGLE edge edit flips a label?  (exhaustive sweep, certified)")
state = GRAPH.parse(task.test[0][0])
factual = sol.test_traces[0].outcome.labels()
nids = [nid for nid, _ in state.nodes]
flips = []
t0 = time.perf_counter()
candidates = [("delete", edge, state.edges - {edge}) for edge in sorted(state.edges)]
candidates += [
    ("add", (u, v), state.edges | {(u, v)})
    for i, u in enumerate(nids)
    for v in nids[i + 1 :]
    if (u, v) not in state.edges
]
for verb, edge, edges in candidates:
    raw = (state.nodes, tuple(sorted(edges)))
    cfs = twinworld.compute(twinworld.identify(rep, Backtracking(raw)))
    cf = cfs.items[0].counterfactual
    outcome = cf.counterfactual.outcome.labels() if cf.applicable else None
    if outcome != factual:
        changed = (
            sorted(n for n in factual if outcome.get(n) != factual[n])
            if outcome is not None
            else "program inapplicable (no triangle left)"
        )
        flips.append((verb, edge, changed))
ms = (time.perf_counter() - t0) * 1000
print(f"\n{len(candidates)} one-edge worlds swept in {ms:.1f} ms; {len(flips)} flip the labelling:")
for verb, edge, changed in flips:
    print(f"  {verb:6s} {edge}  ->  affected node(s): {changed}")

rule("3. pertinent negatives  —  absent edges, at state-identity level")
pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(max_witnesses=2)))
print()
for item in pn.items:
    print(item.narrative)
print(
    "\n(note: an added edge persists into the outcome, so EVERY absent edge\n"
    " witnesses at this level — the label-level question is section 2's sweep)"
)

rule("done — exhaustive edge-edit certificates on the GNN explainers' question")
