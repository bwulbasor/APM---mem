"""Structural enrichment — recover the TYPED structure a proof chain needs from
what the extractor ALREADY produced. Sits between extraction and normalization:

    raw extraction -> [structural enrichment] -> canon/contradict/supersede -> engine

FAITHFULNESS INVARIANT: enrichment may only make EXPLICIT what is already in a
node's own label, its existing edges, or its session timestamp. It never
imports an outside claim, so a CONFIRMED built on hoisted structure is still
backed by the source. Fields are written in the CANONICAL schema (apm.contract)
so parser constraints and node fields align. ANTI-NOISE: never hoist a field
that applies uniformly to all candidates (the 'User attends everything' bug).

Enable with APM_ENRICH=1.
"""
from __future__ import annotations

import re
from collections import Counter

from .contract import resolve_date_phrase
from .graph import Edge, EvidenceGraph, NodeType, parse_ts

# event-type keywords -> canonical `kind` value (first hit wins)
_ETYPE = [
    (("meeting", "assembly", "sync", "board", "standup", "session", "hearing",
      "review", "1:1", "one-on-one"), "meeting"),
    (("call", "phone", "zoom", "dial"), "call"),
    (("trip", "travel", "flight", "vacation", "visited", "traveled"), "travel"),
    (("deadline", "due", "filing", "submission", "cutoff"), "deadline"),
    (("party", "concert", "dinner", "celebration", "social", "gathering"), "social"),
    (("deposition", "trial", "mediation", "arbitration"), "legal_proceeding"),
]
_XVOTE = re.compile(r"\b([a-z][a-z0-9 ]{2,25}?)\s+(?:vote|agenda|item)\b", re.I)
_ABOUT = re.compile(r"\b(?:about|regarding|re:|on the topic of)\s+(?:the\s+)?"
                    r"([a-z][a-z0-9 ]{2,30})", re.I)


def normalize_dates(g: EvidenceGraph) -> int:
    """Resolve a relative/partial date in a node's label to a canonical
    `date` / `date_range` attr, anchored to the node's OWN session timestamp."""
    n = 0
    for node in g.nodes.values():
        if node.attrs.get("date") or node.attrs.get("date_range"):
            continue
        anchor = parse_ts(node.timestamp) if node.timestamp else None
        r = resolve_date_phrase(node.label, anchor)
        if r:
            node.attrs[r[0]] = r[1]
            node.attrs["date_basis"] = f"session={anchor.date().isoformat() if anchor else '?'}"
            n += 1
    return n


def type_events(g: EvidenceGraph) -> int:
    """Hoist canonical `kind` from keywords already in an EVENT node's label."""
    n = 0
    for node in g.nodes.values():
        if node.type != NodeType.EVENT or node.attrs.get("kind"):
            continue
        low = node.label.lower()
        for keys, typ in _ETYPE:
            if any(k in low for k in keys):
                node.attrs["kind"] = typ
                node.attrs["kind_basis"] = "keyword-in-label"
                n += 1
                break
    return n


def hoist_topics(g: EvidenceGraph) -> int:
    """Surface a canonical `topic` token from an EVENT/ASSERTION label."""
    n = 0
    for node in g.nodes.values():
        if node.type not in (NodeType.EVENT, NodeType.ASSERTION):
            continue
        if node.attrs.get("topic"):
            continue
        m = _XVOTE.search(node.label) or _ABOUT.search(node.label)
        if m:
            topic = m.group(1).strip().lower()
            if topic and topic not in ("the", "a", "an"):
                node.attrs["topic"] = topic
                node.attrs["topic_basis"] = "hoisted-from-label"
                n += 1
    return n


def link_participants(g: EvidenceGraph) -> int:
    """Surface `participants` from EXISTING edges — only DISCRIMINATIVE people.
    A person linked to (nearly) every event carries no signal; hoisting them
    uniformly collapses ranking (measured: 'User' on every event -> tie ->
    lost CONFIRMED). Exclude ubiquitous participants."""
    events = [x for x in g.nodes.values() if x.type == NodeType.EVENT]
    if not events:
        return 0
    freq, per_event = Counter(), {}
    for ev in events:
        ppl = [o for o, e in g.neighbors(ev.id)
               if o in g.nodes and g.node(o).type == NodeType.PERSON]
        per_event[ev.id] = ppl
        for p in set(ppl):
            freq[p] += 1
    ubiquitous = {p for p, c in freq.items() if c >= max(2, 0.6 * len(events))}
    n = 0
    for ev in events:
        people = [p for p in per_event[ev.id] if p not in ubiquitous]
        if not people:
            continue
        if not ev.attrs.get("participants"):
            ev.attrs["participants"] = sorted(set(people))
            n += 1
        orgs = {o for p in people for o, e in g.neighbors(p)
                if o in g.nodes and g.node(o).type == NodeType.ORG
                and e.rel in ("member_of", "part_of", "works_at")}
        existing = {(e.src, e.dst) for _, _, d in g.g.edges(data=True)
                    for e in [d["obj"]]}
        for org in orgs:
            if (ev.id, org) not in existing:
                g.add_edge(Edge(ev.id, org, "involves_org", 0.7,
                                source="enrich:participant-cluster"))
                n += 1
    return n


def enrich_structure(g: EvidenceGraph) -> dict:
    return {"dates": normalize_dates(g), "kinds": type_events(g),
            "topics": hoist_topics(g), "participants": link_participants(g)}
