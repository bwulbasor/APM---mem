"""Entity resolution module (v1): type-gated fuzzy matching with alias memory.

Why this module exists: the contradiction / knowledge-update / counting machinery
only works if the same real-world entity lands on the SAME node every session.
Resolution is deterministic string math (reproducible), never an LLM call —
the LLM proposes labels; this module decides identity.

Every node accumulates `aliases` (every surface form ever matched to it), so
resolution gets stronger as the graph grows.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .graph import EvidenceGraph, Node, NodeType

# generic words that carry no identity ("my new car" == "the car")
_STOP = {"the", "a", "an", "my", "your", "his", "her", "our", "their", "this",
         "that", "new", "old", "own", "some"}

# score needed to merge; person names are held to a stricter standard because
# "Mark"/"Marta" style near-misses are catastrophic merges
THRESHOLD = 87.0
PERSON_THRESHOLD = 92.0


def normalize(label: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", label.lower()).split()
    kept = [w for w in words if w not in _STOP]
    return " ".join(kept) if kept else " ".join(words)


def _numbers(label: str) -> set[str]:
    return set(re.findall(r"\d[\d,.:]*", label))


def similarity(a: str, b: str) -> float:
    """Order-insensitive token similarity on stop-stripped labels.

    NUMBERS ARE IDENTITY (from held-out invention 852ce960: '$350,000
    pre-approval' merged with 'Pre-approval of $400,000', erasing the update
    and resurrecting the stale value as CONFIRMED). If both labels carry
    numbers and the number sets are disjoint, they are competing VALUES of
    the same metric — never the same entity."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    num_a, num_b = _numbers(a), _numbers(b)
    if num_a and num_b and not (num_a & num_b):
        return 0.0
    return fuzz.token_set_ratio(na, nb)


def resolve(graph: EvidenceGraph, label: str,
            node_type: NodeType | None = None) -> str | None:
    """Find the existing node this label refers to, or None.

    Type-gated: a person can never resolve to a place. When node_type is None
    (query-side, type unknown) all types compete but the threshold holds.
    """
    best_id, best_score = None, 0.0
    for n in graph.nodes.values():
        if node_type is not None and n.type != node_type:
            continue
        candidates = [n.label] + list(n.attrs.get("aliases", []))
        score = max(similarity(label, c) for c in candidates)
        if score > best_score:
            best_id, best_score = n.id, score
    if best_id is None:
        return None
    threshold = (PERSON_THRESHOLD
                 if graph.nodes[best_id].type == NodeType.PERSON
                 else THRESHOLD)
    return best_id if best_score >= threshold else None


def remember_alias(graph: EvidenceGraph, node_id: str, surface: str):
    """Record a surface form on the node so future resolution catches it."""
    node = graph.nodes[node_id]
    aliases = node.attrs.setdefault("aliases", [])
    if surface != node.label and surface not in aliases:
        aliases.append(surface)


def merge_nodes(graph: EvidenceGraph, keep_id: str, drop_id: str):
    """Fold drop into keep: rewire edges, union aliases/attrs. Provenance of the
    merge is recorded as an alias — merges must be explainable, like everything."""
    keep, drop = graph.nodes[keep_id], graph.nodes[drop_id]
    remember_alias(graph, keep_id, drop.label)
    for alias in drop.attrs.get("aliases", []):
        remember_alias(graph, keep_id, alias)
    for k, v in drop.attrs.items():
        if k != "aliases":
            keep.attrs.setdefault(k, v)
    for _, dst, data in list(graph.g.out_edges(drop_id, data=True)):
        e = data["obj"]
        if dst != keep_id:
            e.src = keep_id
            graph.g.add_edge(keep_id, dst, rel=e.rel, obj=e)
    for src, _, data in list(graph.g.in_edges(drop_id, data=True)):
        e = data["obj"]
        if src != keep_id:
            e.dst = keep_id
            graph.g.add_edge(src, keep_id, rel=e.rel, obj=e)
    graph.g.remove_node(drop_id)
    del graph.nodes[drop_id]
