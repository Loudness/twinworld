"""The graph representation backend — labelled simple graphs, registered as
``"graph"``.

States are node/edge sets under SEMANTIC node ids (graph isomorphism is
deliberately NOT quotiented — two isomorphic graphs with different ids are
different states; sorted certificates keep keys canonical on the small graphs
this targets). The motif vocabulary is where the backend earns its keep:
:class:`LabelTriangles` labels every node that closes a triangle — a
context-dependent mechanism (the same program labels different nodes in
different graphs), so certified minimal EDGE edits answer the
CF-GNNExplainer-style question "which edge made this node's label flip?"
with exhaustive certainty instead of a gradient estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Collection, Hashable, Iterator, Mapping, Sequence

from ..backend import Addition, register

Nodes = tuple[tuple[int, Hashable], ...]  # sorted (nid, label)
Edge = tuple[int, int]  # normalized: (min, max)


def _norm(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class GraphEntity:
    """One node (``"nodes"`` scheme) or one connected component."""

    oid: int
    extent: tuple
    size: int
    _attrs: tuple[tuple[str, Hashable], ...]

    @cached_property
    def attributes(self) -> dict[str, Hashable]:
        return dict(self._attrs)


@dataclass(frozen=True, eq=False)
class GraphState:
    """A labelled simple graph. Equality and hashing go through ``key`` —
    the sorted node/edge certificate."""

    nodes: Nodes
    edges: frozenset[Edge]
    abstraction: str = "nodes"

    representation = "graph"

    @property
    def key(self) -> tuple:
        return ("graph", self.nodes, tuple(sorted(self.edges)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphState) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def labels(self) -> dict[int, Hashable]:
        return dict(self.nodes)

    def neighbours(self, nid: int) -> frozenset[int]:
        return frozenset(v if u == nid else u for u, v in self.edges if nid in (u, v))

    @cached_property
    def objects(self) -> tuple[GraphEntity, ...]:
        if self.abstraction == "components":
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(nid for nid, _ in self.nodes)
            g.add_edges_from(self.edges)
            labels = self.labels()
            entities = []
            for component in nx.connected_components(g):
                members = tuple(sorted(component))
                entities.append(
                    GraphEntity(
                        oid=members[0],
                        extent=(members[0], members),
                        size=len(members),
                        _attrs=(
                            ("size", len(members)),
                            ("labels", tuple(sorted(str(labels[m]) for m in members))),
                        ),
                    )
                )
            return tuple(sorted(entities, key=lambda e: e.oid))
        return tuple(
            GraphEntity(
                oid=nid,
                extent=(nid, frozenset(e for e in self.edges if nid in e)),
                size=len(self.neighbours(nid)) + 1,  # degree + 1: Largest = the hub
                _attrs=(("label", label), ("degree", len(self.neighbours(nid)))),
            )
            for nid, label in self.nodes
        )


# ------------------------------------------------------------------ mechanisms


@dataclass(frozen=True)
class AddEdge:
    u: int
    v: int

    exact_preimage = True  # the unique predecessor lacks exactly this edge

    def apply(self, s: GraphState) -> GraphState | None:
        edge = _norm(self.u, self.v)
        known = {nid for nid, _ in s.nodes}
        if self.u == self.v or edge[0] not in known or edge[1] not in known:
            return None
        if edge in s.edges:
            return None  # no-op applications are inapplicable
        return GraphState(s.nodes, s.edges | {edge}, s.abstraction)

    def preimage(self, s: GraphState, budget=None) -> Iterator[GraphState]:
        edge = _norm(self.u, self.v)
        if edge in s.edges:
            candidate = GraphState(s.nodes, s.edges - {edge}, s.abstraction)
            if self.apply(candidate) == s:
                yield candidate

    def __str__(self) -> str:
        return f"add edge ({min(self.u, self.v)}, {max(self.u, self.v)})"


@dataclass(frozen=True)
class DelEdge:
    u: int
    v: int

    exact_preimage = True

    def apply(self, s: GraphState) -> GraphState | None:
        edge = _norm(self.u, self.v)
        if edge not in s.edges:
            return None
        return GraphState(s.nodes, s.edges - {edge}, s.abstraction)

    def preimage(self, s: GraphState, budget=None) -> Iterator[GraphState]:
        edge = _norm(self.u, self.v)
        known = {nid for nid, _ in s.nodes}
        if edge[0] in known and edge[1] in known and edge not in s.edges:
            candidate = GraphState(s.nodes, s.edges | {edge}, s.abstraction)
            if self.apply(candidate) == s:
                yield candidate

    def __str__(self) -> str:
        return f"delete edge ({min(self.u, self.v)}, {max(self.u, self.v)})"


@dataclass(frozen=True)
class SetLabel:
    nid: int
    label: Hashable

    # the predecessor's label is enumerated from the state, not known exactly
    exact_preimage = False

    def apply(self, s: GraphState) -> GraphState | None:
        labels = s.labels()
        if self.nid not in labels or labels[self.nid] == self.label:
            return None
        labels[self.nid] = self.label
        return GraphState(tuple(sorted(labels.items())), s.edges, s.abstraction)

    def preimage(self, s: GraphState, budget=None) -> Iterator[GraphState]:
        labels = s.labels()
        if labels.get(self.nid) != self.label:
            return
        observed = {label for _, label in s.nodes}
        for original in sorted(observed - {self.label}, key=repr):
            candidate_labels = dict(labels)
            candidate_labels[self.nid] = original
            candidate = GraphState(
                tuple(sorted(candidate_labels.items())), s.edges, s.abstraction
            )
            if self.apply(candidate) == s:
                yield candidate

    def __str__(self) -> str:
        return f"set label of node {self.nid} to {self.label!r}"


@dataclass(frozen=True)
class LabelTriangles:
    """Label every node that closes a triangle — the deterministic motif
    classifier. Context-dependent like MoveBlock: the same program labels
    different nodes in different graphs, so edge edits genuinely change the
    outcome and minimal edge-edit certificates mean something."""

    label: Hashable

    exact_preimage = False  # pre-labels of triangle nodes are enumerated, not known

    def _in_triangle(self, s: GraphState, nid: int) -> bool:
        nbrs = sorted(s.neighbours(nid))
        return any(
            _norm(a, b) in s.edges for i, a in enumerate(nbrs) for b in nbrs[i + 1 :]
        )

    def apply(self, s: GraphState) -> GraphState | None:
        labels = s.labels()
        changed = False
        for nid in labels:
            if self._in_triangle(s, nid) and labels[nid] != self.label:
                labels[nid] = self.label
                changed = True
        if not changed:
            return None  # no triangle to label (or all already labelled)
        return GraphState(tuple(sorted(labels.items())), s.edges, s.abstraction)

    def preimage(self, s: GraphState, budget=None) -> Iterator[GraphState]:
        labels = s.labels()
        triangle_nodes = [nid for nid in labels if self._in_triangle(s, nid)]
        if any(labels[nid] != self.label for nid in triangle_nodes):
            return  # s is not a possible result of this mechanism
        observed = sorted({label for _, label in s.nodes} - {self.label}, key=repr)
        # bounded: flip SINGLE triangle nodes back to an observed label, verified
        for nid in triangle_nodes:
            for original in observed:
                candidate_labels = dict(labels)
                candidate_labels[nid] = original
                candidate = GraphState(
                    tuple(sorted(candidate_labels.items())), s.edges, s.abstraction
                )
                if self.apply(candidate) == s:
                    yield candidate

    def __str__(self) -> str:
        return f"label every triangle node {self.label!r}"


class _Scheme:
    def __init__(self, name: str):
        self.name = name


class GraphRepresentation:
    name = "graph"
    default_abstractions = ("nodes",)
    transform_families: tuple = ()  # motif vocabulary is enumerated, not induced
    abstractions: Mapping[str, object] = {
        "nodes": _Scheme("nodes"),
        "components": _Scheme("components"),
    }

    def parse(
        self, raw, abstraction: str | None = None, context: Mapping | None = None
    ) -> GraphState:
        if abstraction is not None and abstraction not in self.abstractions:
            raise KeyError(abstraction)
        raw_nodes, raw_edges = raw
        pairs = raw_nodes.items() if isinstance(raw_nodes, Mapping) else raw_nodes
        nodes = tuple(sorted((int(nid), label) for nid, label in pairs))
        edges = frozenset(_norm(int(u), int(v)) for u, v in raw_edges)
        return GraphState(nodes, edges, abstraction or self.default_abstractions[0])

    def canon(self, raw) -> tuple:
        return self.parse(raw).key

    def raw_of(self, state: GraphState) -> tuple:
        return (state.nodes, tuple(sorted(state.edges)))

    def frame(self, state: GraphState) -> tuple[int, ...]:
        return tuple(nid for nid, _ in state.nodes)  # the node universe is the frame

    def rebuild(
        self, template: GraphState, entities: Sequence[GraphEntity]
    ) -> GraphState | None:
        labels = template.labels()
        seen: set[int] = set()
        edges: set[Edge] = set()
        for entity in entities:
            nid, incident = entity.extent
            if not isinstance(incident, frozenset) or nid in seen:
                return None
            seen.add(nid)
            edges |= incident
        if seen != set(labels):
            return None
        return GraphState(template.nodes, frozenset(edges), template.abstraction)

    def candidate_primitives(self, task) -> list:
        state = self.parse(task.train[0][0])
        nids = [nid for nid, _ in state.nodes]
        labels = sorted(
            {label for _, raw_out in task.train for _, label in self.parse(raw_out).nodes},
            key=repr,
        )
        rules: list = [LabelTriangles(label) for label in labels]
        rules += [AddEdge(u, v) for i, u in enumerate(nids) for v in nids[i + 1 :]]
        rules += [DelEdge(u, v) for i, u in enumerate(nids) for v in nids[i + 1 :]]
        rules += [SetLabel(nid, label) for nid in nids for label in labels]
        return rules

    def task_values(self, task) -> frozenset:
        return frozenset(
            label
            for pairs in (task.train, task.test)
            for pair in pairs
            for raw in pair
            for _, label in self.parse(raw).nodes
        )

    def attr_domain(self, attr: str) -> None:
        return None

    def fresh_value(self, attr: str, used: Collection) -> int:
        ints = [v for v in used if isinstance(v, int)]
        return max(ints, default=0) + 1

    def relations(self, state: GraphState) -> set[tuple[str, int, int]]:
        rels: set[tuple[str, int, int]] = {("edge", u, v) for u, v in state.edges}
        labels = state.labels()
        for u in labels:
            for v in labels:
                if u < v and labels[u] == labels[v]:
                    rels.add(("same_label", u, v))
        return rels

    def overlap(self, a: GraphEntity, b: GraphEntity) -> float:
        if a.extent == b.extent:
            return 1.0
        ea, eb = a.extent[1], b.extent[1]
        if not isinstance(ea, frozenset) or not isinstance(eb, frozenset) or not (ea | eb):
            return 0.0
        return len(ea & eb) / len(ea | eb)

    # ------------------------------------------------ optional capabilities

    def probe_perturbations(self, state: GraphState, used: Collection) -> Iterator[tuple]:
        """Delete each edge; relabel each node to a fresh label; rewire one
        endpoint of the first few edges."""
        fresh = self.fresh_value("label", used)
        for edge in sorted(state.edges):
            yield (state.nodes, tuple(sorted(state.edges - {edge})))
        for nid, label in state.nodes:
            if label == fresh:
                continue
            relabelled = tuple(
                (n, fresh if n == nid else lab) for n, lab in state.nodes
            )
            yield (relabelled, tuple(sorted(state.edges)))
        nids = [nid for nid, _ in state.nodes]
        for u, v in sorted(state.edges)[:3]:
            for target in nids:
                if target in (u, v) or _norm(u, target) in state.edges:
                    continue
                rewired = (state.edges - {(u, v) if (u, v) in state.edges else _norm(u, v)}) | {
                    _norm(u, target)
                }
                yield (state.nodes, tuple(sorted(rewired)))
                break

    def addition_values(self, state: GraphState, task) -> list:
        return [None]  # edges carry no value; the catalogue enumerates absent pairs

    def addition_catalogue(
        self, state: GraphState, max_size: int, separated: bool, values: Sequence
    ) -> Iterator[Addition]:
        """Pertinent-negative additions: one absent edge at a time.
        ``values``/``max_size``/``separated`` have no graph analogue.

        Bluntness, documented: an added edge persists into the outcome, so it
        changes its endpoints' extents and EVERY absent edge witnesses at the
        state-identity level. The sharp label-level question ("which edge
        flips a label?") is the exhaustive Backtracking sweep
        (examples/graph_motifs.py §2)."""
        del max_size, separated, values
        nids = [nid for nid, _ in state.nodes]
        for i, u in enumerate(nids):
            for v in nids[i + 1 :]:
                if _norm(u, v) in state.edges:
                    continue
                yield Addition(
                    raw=(state.nodes, tuple(sorted(state.edges | {_norm(u, v)}))),
                    phrase=f"an edge ({u}, {v}) existed",
                    size=1,
                    group=(u, v),
                )

    def plausible(self, state: GraphState) -> bool:
        """Simple-graph invariants: sorted unique nids, normalized edges with
        known endpoints, no self-loops."""
        nids = [nid for nid, _ in state.nodes]
        if nids != sorted(nids) or len(nids) != len(set(nids)):
            return False
        known = set(nids)
        return all(u < v and u in known and v in known for u, v in state.edges)

    def render_raw(self, raw, caption: str | None = None, diff_against=None) -> str:
        """Adjacency-matrix HTML — documented as the weakest renderer."""
        from ..viz import _esc

        state = raw if isinstance(raw, GraphState) else self.parse(raw)
        nids = [nid for nid, _ in state.nodes]
        labels = state.labels()
        header = "<tr><th></th>" + "".join(f"<th>{n}</th>" for n in nids) + "<th>label</th></tr>"
        rows = []
        for u in nids:
            cells = "".join(
                f"<td>{'●' if _norm(u, v) in state.edges else ''}</td>" for v in nids
            )
            rows.append(f"<tr><th>{u}</th>{cells}<td>{_esc(labels[u])}</td></tr>")
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return f'<figure class="grid"><table class="t">{header}{"".join(rows)}</table>{cap}</figure>'

    def render_html(self, state: GraphState, caption: str | None = None) -> str:
        return self.render_raw(state, caption)

    def render_key(self, key, caption: str | None = None) -> str:
        return self.render_raw((key[1], key[2]), caption)


GRAPH = register(GraphRepresentation())


def graph_task(
    train: Sequence[tuple[tuple, tuple]],
    test: Sequence[tuple[tuple, tuple]],
    task_id: str = "graph",
):
    """A graph task from ((nodes, edges) -> (nodes, edges)) raw pairs."""
    from ..engine import Task

    rep = GRAPH

    def canon_raw(raw) -> tuple:
        state = rep.parse(raw)
        return rep.raw_of(state)

    return Task(
        train=tuple((canon_raw(a), canon_raw(b)) for a, b in train),
        test=tuple((canon_raw(a), canon_raw(b)) for a, b in test),
        task_id=task_id,
        representation="graph",
    )
