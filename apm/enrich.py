"""Containment enrichment: build the middle taxonomy tier (Arusha -> Tanzania
-> Africa) that extraction only produces by luck.

One cached LLM call per unenriched place; the cache makes it deterministic and
free on re-runs. Enrichment can only ADD containment edges (taxonomy relations,
which never carry activation per the hub rule) — it cannot touch facts, so a
wrong enrichment costs scoping recall, never answer correctness.
"""
from __future__ import annotations

import json

from .entities import resolve
from .graph import Edge, EvidenceGraph, Node, NodeType
from .ingest import _ask_json, _slug
from .taxonomy import CONTAINMENT_RELS

ENRICH_SYSTEM = """Given a place name, output ONLY JSON naming its containers:
{"city": "<city or null>", "country": "<country or null>", "region": "<continent/region or null>"}
Use null when the place IS that level or you are not sure. Never guess wildly."""


def enrich_places(graph: EvidenceGraph, max_calls: int = 15) -> int:
    """Give every parentless PLACE node a containment parent chain."""
    added = 0
    calls = 0
    for node in list(graph.nodes.values()):
        if node.type != NodeType.PLACE or node.id.startswith("cat_"):
            continue
        has_parent = any(e.rel in CONTAINMENT_RELS
                         for _, e in graph.neighbors(node.id)
                         if e.src == node.id)
        if has_parent or calls >= max_calls:
            continue
        try:
            data = _ask_json(ENRICH_SYSTEM, node.label, role="enrich")
        except Exception:
            continue
        calls += 1
        prev = node.id
        for level in ("city", "country", "region"):
            name = data.get(level)
            if not name or str(name).lower() in ("null", node.label.lower()):
                continue
            parent_id = resolve(graph, str(name), NodeType.PLACE)
            if parent_id is None:
                parent_id = f"pl_{_slug(str(name))}"
                if parent_id not in graph.nodes:
                    graph.add_node(Node(parent_id, NodeType.PLACE, str(name),
                                        source="enrich"))
            if parent_id != prev:
                graph.add_edge(Edge(prev, parent_id, "located_in", 0.85,
                                    source="enrich"))
                added += 1
                prev = parent_id
    return added
