"""The graph backend: identity laws, motif mechanism, edge-edit certificates."""

import twinworld
from twinworld import PertinentNegative, conformance_battery
from twinworld.backends.graph import (
    AddEdge,
    DelEdge,
    GraphState,
    LabelTriangles,
    SetLabel,
    graph_task,
)

GRAPH = twinworld.get_representation("graph")

TRIANGLE = (((1, 0), (2, 0), (3, 0), (4, 0)), ((1, 2), (2, 3), (1, 3), (3, 4)))
PATH = (((1, 0), (2, 0), (3, 0)), ((1, 2), (2, 3)))

SAMPLES = (TRIANGLE, PATH)

MECHANISMS = (
    AddEdge(1, 3),
    DelEdge(1, 2),
    SetLabel(2, 1),
    LabelTriangles(1),
)


def test_parse_normalizes_and_keys():
    state = GRAPH.parse((((2, 0), (1, 0)), ((2, 1),)))
    assert state.nodes == ((1, 0), (2, 0))  # nodes sorted
    assert state.edges == frozenset({(1, 2)})  # edges normalized (min, max)
    assert GRAPH.parse((((1, 0), (2, 0)), ((1, 2),))) == state


def test_entities_label_degree_and_components_scheme():
    state = GRAPH.parse(TRIANGLE)
    hub = next(o for o in state.objects if o.oid == 3)
    assert hub.attributes == {"label": 0, "degree": 3}
    assert hub.size == 4  # degree + 1: Largest selects the hub
    two_components = GRAPH.parse((((1, 0), (2, 0), (3, 0)), ((1, 2),)), "components")
    assert [(o.oid, o.size) for o in two_components.objects] == [(1, 2), (3, 1)]


def test_key_abstraction_invariance():
    for raw in SAMPLES:
        assert len({GRAPH.parse(raw, name).key for name in GRAPH.abstractions}) == 1


def test_addedge_deledge_exact_preimages():
    state = GRAPH.parse(PATH)
    grown = AddEdge(1, 3).apply(state)
    assert list(AddEdge(1, 3).preimage(grown)) == [state]
    shrunk = DelEdge(1, 2).apply(state)
    assert list(DelEdge(1, 2).preimage(shrunk)) == [state]
    assert AddEdge(1, 3).exact_preimage and DelEdge(1, 2).exact_preimage


def test_label_triangles_is_context_dependent():
    motif = LabelTriangles(1)
    labelled = motif.apply(GRAPH.parse(TRIANGLE))
    assert labelled.labels() == {1: 1, 2: 1, 3: 1, 4: 0}  # node 4 hangs off, unlabelled
    assert motif.apply(GRAPH.parse(PATH)) is None  # no triangle: inapplicable


def test_setlabel_preimage_bounded_honest():
    state = GRAPH.parse((((1, 0), (2, 1)), ()))
    pres = list(SetLabel(2, 1).preimage(state))
    assert GRAPH.parse((((1, 0), (2, 0)), ())) in pres  # flip back to an observed label
    assert SetLabel(2, 1).exact_preimage is False  # the original may be unobserved


def test_conformance_battery_graph_passes():
    report = conformance_battery(GRAPH, SAMPLES, mechanisms=MECHANISMS)
    assert report.passed, str(report)
    by_name = {row.name: row for row in report.rows}
    assert by_name["L6_exact_preimage_spot"].passed is True


def test_solve_triangle_closure_motif():
    other = (((1, 0), (2, 0), (3, 0), (4, 0)), ((1, 4), (2, 4), (1, 2)))  # triangle 1-2-4
    task = graph_task(
        train=[
            (TRIANGLE, (((1, 1), (2, 1), (3, 1), (4, 0)), TRIANGLE[1])),
            (other, (((1, 1), (2, 1), (3, 0), (4, 1)), other[1])),
        ],
        test=[(TRIANGLE, (((1, 1), (2, 1), (3, 1), (4, 0)), TRIANGLE[1]))],
    )
    rep = twinworld.model(task, max_depth=1)
    assert rep.solution.program == (LabelTriangles(1),)  # one motif rule fits BOTH graphs


def test_pn_absent_edge_witness_phrase():
    task = graph_task(
        train=[(TRIANGLE, (((1, 1), (2, 1), (3, 1), (4, 0)), TRIANGLE[1]))],
        test=[(TRIANGLE, (((1, 1), (2, 1), (3, 1), (4, 0)), TRIANGLE[1]))],
    )
    rep = twinworld.model(task, primitives=[LabelTriangles(1)], induction="never", max_depth=1)
    pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(max_witnesses=8)))
    witnesses = [item.narrative for item in pn.items]
    # adding edge (1,4) or (2,4) closes a NEW triangle through node 4: load-bearing absence
    assert any("an edge (" in text and "outcome would change" in text for text in witnesses)


def test_probes_cover_delete_relabel_rewire():
    state = GRAPH.parse(PATH)
    probes = list(GRAPH.probe_perturbations(state, used=GRAPH.task_values
    (graph_task(train=[(PATH, PATH)], test=[]))))
    assert (state.nodes, ((2, 3),)) in probes  # edge (1,2) deleted
    assert any(dict(nodes).get(1) == 1 for nodes, _ in probes)  # node 1 relabelled fresh


def test_plausible_rejects_self_loop():
    assert GRAPH.plausible(GRAPH.parse(TRIANGLE)) is True
    assert GRAPH.plausible(GraphState(((1, 0),), frozenset({(1, 1)}))) is False
