"""Spreading activation with per-anchor path tracking (SPEC.md §3.1)."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime

from .graph import Edge, EvidenceGraph, NodeType, Status, parse_ts

DECAY = 0.6
ACTIVATION_THRESHOLD = 0.05
MAX_HOPS = 4

# temporal proximity boost (SPEC.md §6): edges into near-future events matter more
TEMPORAL_BETA = 1.0
TEMPORAL_TAU_HOURS = 48.0
WEIGHT_CAP = 0.99


@dataclass
class Path:
    """Best path from one anchor to one node."""
    anchor: str
    node: str
    score: float
    edges: list[Edge] = field(default_factory=list)

    @property
    def intermediates(self) -> set[str]:
        """Nodes strictly between anchor and target — used for independence checks."""
        seen = set()
        for e in self.edges:
            seen.add(e.src)
            seen.add(e.dst)
        return seen - {self.anchor, self.node}


def temporal_boost(graph: EvidenceGraph, dst_id: str, now: datetime | None) -> float:
    """Boost association strength toward events close in the future (S3 mechanic)."""
    if now is None:
        return 1.0
    node = graph.node(dst_id)
    if node.type != NodeType.EVENT or not node.timestamp:
        return 1.0
    delta_h = (parse_ts(node.timestamp) - now).total_seconds() / 3600.0
    if delta_h < 0:
        return 1.0
    import math
    return 1.0 + TEMPORAL_BETA * math.exp(-delta_h / TEMPORAL_TAU_HOURS)


MAX_PARALLEL_PATHS = 4   # route-distinct paths kept per (anchor, node)


def spread(graph: EvidenceGraph, anchors: list[str],
           now: datetime | None = None) -> dict[str, list[Path]]:
    """Run activation from each anchor independently; return Paths per node.

    Dijkstra-style on multiplicative scores: activation(next) = activation(cur)
    * edge.weight * DECAY * temporal_boost, cut at ACTIVATION_THRESHOLD / MAX_HOPS.

    CONVERGENCE (v1.5, SPEC §3.1 'a(target) +='): a node reached from the SAME
    anchor via several route-distinct paths keeps up to MAX_PARALLEL_PATHS of
    them, not just the best — four independent assertions all pointing at one
    store ARE stronger evidence than one (the S2 mechanic, within-anchor).
    Only the best path propagates onward, so there is no combinatorial blowup;
    scoring's independence filter (shared-intermediate rule) guards double
    counting downstream."""
    reached: dict[str, list[Path]] = {}
    for anchor in anchors:
        best: dict[str, list[Path]] = {anchor: [Path(anchor, anchor, 1.0)]}
        heap = [(-1.0, 0, anchor)]
        while heap:
            neg, hops, cur = heapq.heappop(heap)
            score = -neg
            if score < best[cur][0].score or hops >= MAX_HOPS:
                continue
            for other, edge in graph.neighbors(cur):
                if graph.node(other).status == Status.RETRACTED:
                    continue
                # THE HUB RULE: category super-parents (PLACES, NAMES...) are
                # promiscuous hubs — activation flowing through one would light
                # up every sibling and destroy intersection scoring. Taxonomy
                # is for climbing/scoping (fallback, counts), not association.
                if other.startswith("cat_") or other.startswith("ep_"):
                    continue
                w = min(edge.weight * temporal_boost(graph, other, now), WEIGHT_CAP)
                nxt = score * w * DECAY
                if nxt < ACTIVATION_THRESHOLD:
                    continue
                # no revisits: a route that bounces off a neighbor back into
                # a node it already visited is a cycle, not fresh evidence
                seen = {anchor} | {n for e in best[cur][0].edges
                                   for n in (e.src, e.dst)}
                if other in seen:
                    continue
                path = Path(anchor, other, nxt, best[cur][0].edges + [edge])
                lst = best.setdefault(other, [])
                if not lst or nxt > lst[0].score:
                    lst.insert(0, path)
                    del lst[MAX_PARALLEL_PATHS:]
                    heapq.heappush(heap, (-nxt, hops + 1, other))
                elif len(lst) < MAX_PARALLEL_PATHS:
                    # parallel route: counts as convergent evidence if it
                    # arrives via genuinely different intermediates; weaker
                    # routes are recorded but never propagated onward
                    inter = path.intermediates
                    if all(inter != p.intermediates for p in lst):
                        lst.append(path)
        for node_id, paths in best.items():
            if node_id != anchor:
                reached.setdefault(node_id, []).extend(paths)
    return reached
