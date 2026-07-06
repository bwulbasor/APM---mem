"""Episode clustering: discover the natural chunks of memory.

Humans don't store experience as a flat soup — event segmentation carves it
into episodes ("the Tanzania trip", "the mortgage saga"). We recover those
with Louvain community detection over the association graph: dense subgraphs
= episodes. Each becomes a parent node (`ep_*`) with `in_episode` edges.

Episode nodes obey THE HUB RULE (like `cat_*`): they never carry activation —
they exist for scoping ("under" queries), summarization, and future
consolidation. Labels are deterministic: the 3 highest-degree member labels.
"""
from __future__ import annotations

import networkx as nx

from .graph import Edge, EvidenceGraph, Node, NodeType

MIN_EPISODE = 5   # smaller clusters aren't episodes, just neighborhoods


def is_episode(node_id: str) -> bool:
    return node_id.startswith("ep_")


def build_episodes(graph: EvidenceGraph) -> int:
    """Idempotent: clears previous ep_* nodes, re-clusters, relinks."""
    for nid in [n for n in graph.nodes if is_episode(n)]:
        graph.remove_node(nid)

    # cluster over the ASSOCIATION structure only — taxonomy would glue
    # everything of one type into a fake mega-episode
    sub = nx.Graph()
    for src, dst, data in graph.g.edges(data=True):
        e = data["obj"]
        if src.startswith("cat_") or dst.startswith("cat_"):
            continue
        if e.rel == "is_a":
            continue
        w = max(e.weight, sub.get_edge_data(src, dst, {}).get("weight", 0))
        sub.add_edge(src, dst, weight=w)
    if sub.number_of_nodes() < MIN_EPISODE:
        return 0

    communities = nx.algorithms.community.louvain_communities(
        sub, weight="weight", seed=7)   # fixed seed: deterministic episodes
    made = 0
    for members in communities:
        members = [m for m in members if m in graph.nodes]
        if len(members) < MIN_EPISODE:
            continue
        # deterministic label: the 3 best-connected members name the episode
        top = sorted(members, key=lambda m: -sub.degree(m))[:3]
        label = "Episode: " + " · ".join(graph.node(m).label[:24] for m in top)
        # deterministic id from sorted membership
        eid = f"ep_{abs(hash(tuple(sorted(members)))) % 10**10}"
        ts = max((graph.node(m).timestamp or "" for m in members), default=None)
        graph.add_node(Node(eid, NodeType.TOPIC, label,
                            attrs={"members": sorted(members)},
                            timestamp=ts or None, source="episodes"))
        for m in members:
            graph.add_edge(Edge(m, eid, "in_episode", 0.9, source="episodes"))
        made += 1
    return made
