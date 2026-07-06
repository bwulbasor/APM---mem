"""The fallback ladder: parent climb -> constraint projection -> defaults (SPEC.md §3.3)."""
from __future__ import annotations

from dataclasses import dataclass

from .activation import Path
from .graph import EvidenceGraph, Node, parse_ts
from .scoring import match_constraints, noisy_or

PARENT_MIN_SCORE = 0.3  # a parent hub must be this well-supported before we project from it


@dataclass
class Candidate:
    node: Node
    score: float
    paths: list[Path]          # anchor paths supporting this candidate (or its parent)
    match: float = 1.0
    via_parent: Node | None = None
    projection_note: str | None = None


def project_from_parents(graph: EvidenceGraph, activated: dict[str, list[Path]],
                         placeholder_type, constraints: dict,
                         required: frozenset = frozenset()) -> list[Candidate]:
    """Step [6a]+[6b]: find well-supported parent hubs among activated nodes, then
    project the placeholder's constraints onto each parent's adjacent knowledge."""
    out: list[Candidate] = []
    for node_id, paths in activated.items():
        parent = graph.node(node_id)
        parent_score = noisy_or(paths)
        if parent_score < PARENT_MIN_SCORE:
            continue
        # a hub is only a "parent" if some anchor reached it via a parent relation
        if not any(e.rel in ("member_of", "part_of") for p in paths for e in p.edges):
            continue
        for other_id, edge in graph.neighbors(node_id):
            cand = graph.node(other_id)
            if cand.type != placeholder_type:
                continue
            m = match_constraints(cand, constraints, strict=False, required=required)
            if m == 0.0:
                continue
            out.append(Candidate(
                node=cand, score=parent_score * m, paths=paths, match=m,
                via_parent=parent,
                projection_note=(f"{parent.label} --{edge.rel}--> {cand.label} | "
                                 f"constraint match {m:.2f}"),
            ))
    return out


def recency_default(cands: list[Candidate]) -> Candidate | None:
    """Step [6c]: among undiscriminated candidates, prefer the most recent (SPEC.md §6)."""
    dated = [c for c in cands if c.node.timestamp]
    if not dated:
        return None
    return max(dated, key=lambda c: parse_ts(c.node.timestamp))
