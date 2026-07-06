"""Taxonomy layer: category super-parents over the evidence graph.

Collins & Quillian (1969) hierarchical semantic memory, grafted onto our
Collins & Loftus (1975) spreading activation: NAMES / PLACES / EVENTS / ...
category nodes sit above instances, with instance-level parents (Tanzania
under PLACES; people and events linked beneath it) built from located_in /
is_in / part_of edges that extraction emits.

THE HUB RULE (critical): category nodes are promiscuous hubs — every place
connects to PLACES. Activation must NEVER flow through them, or one active
place lights up all places and intersection scoring dies. Taxonomy edges are
for CLIMBING (fallback) and SCOPING (counts, subtree queries), not association.
Enforced in activation.spread via is_category().
"""
from __future__ import annotations

from .graph import Edge, EvidenceGraph, Node, NodeType

CAT_PREFIX = "cat_"

# instance-level containment relations extraction may emit
CONTAINMENT_RELS = {"located_in", "is_in", "part_of", "in_country",
                    "region_of", "in_episode"}


def is_category(node_id: str) -> bool:
    return node_id.startswith(CAT_PREFIX)


def cat_id(node_type: NodeType) -> str:
    return f"{CAT_PREFIX}{node_type.value}"


def ensure_categories(graph: EvidenceGraph):
    """Create one category super-parent per node type and link every instance
    with an `is_a` edge. Idempotent; call after ingestion."""
    for ntype in NodeType:
        cid = cat_id(ntype)
        if cid not in graph.nodes:
            graph.add_node(Node(cid, ntype, ntype.value.upper() + "S",
                                source="taxonomy"))
    linked = {
        (e.src, e.dst) for _, _, d in graph.g.edges(data=True)
        for e in [d["obj"]] if e.rel == "is_a"
    }
    for node in list(graph.nodes.values()):
        if is_category(node.id):
            continue
        cid = cat_id(node.type)
        if (node.id, cid) not in linked:
            graph.add_edge(Edge(node.id, cid, "is_a", 0.9, source="taxonomy"))


def subtree(graph: EvidenceGraph, root_id: str, max_depth: int = 4) -> set[str]:
    """All instance nodes under a category or containment parent — the
    membership set for scoped counts and generalization queries.

    Children arrive via `is_a` (instance -> category) and CONTAINMENT_RELS
    (Arusha -> located_in -> Tanzania), followed INWARD from the root.
    """
    out: set[str] = set()
    frontier = {root_id}
    for _ in range(max_depth):
        nxt = set()
        for nid in frontier:
            for src, _, data in graph.g.in_edges(nid, data=True):
                e = data["obj"]
                if e.rel == "is_a" or e.rel in CONTAINMENT_RELS:
                    if src not in out and not is_category(src):
                        out.add(src)
                        nxt.add(src)
        frontier = nxt
        if not frontier:
            break
    return out


def parents_of(graph: EvidenceGraph, node_id: str) -> list[str]:
    """Containment parents of an instance (Tanzania for Arusha), for the
    fallback ladder's extra rungs."""
    out = []
    for _, dst, data in graph.g.out_edges(node_id, data=True):
        e = data["obj"]
        if e.rel in CONTAINMENT_RELS or e.rel == "is_a":
            out.append(dst)
    return out
