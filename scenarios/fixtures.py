"""The four founding scenarios (SPEC.md §4) as graph fixtures + structured queries."""
from __future__ import annotations

from datetime import datetime

from apm import Edge, EvidenceGraph, Node, NodeType, Placeholder, Query

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def s1_iaeste(with_agenda: bool = True, with_calendar: bool = False):
    """S1 — 'Don't forget Friday the 26th!' in a chat with Anna, Marco, Priya."""
    g = EvidenceGraph()
    for pid, name in [("p_anna", "Anna"), ("p_marco", "Marco"), ("p_priya", "Priya")]:
        g.add_node(Node(pid, NodeType.PERSON, name))
    g.add_node(Node("org_iaeste", NodeType.ORG, "IAESTE"))
    for pid in ["p_anna", "p_marco", "p_priya"]:
        g.add_edge(Edge(pid, "org_iaeste", "member_of", 0.9))

    # distractor: Anna is also in a book club, which plans nothing in June
    g.add_node(Node("org_bookclub", NodeType.ORG, "Book club"))
    g.add_edge(Edge("p_anna", "org_bookclub", "member_of", 0.7))

    if with_agenda:
        g.add_node(Node("ev_ga", NodeType.EVENT, "IAESTE General Assembly",
                        attrs={"date_window": ("2026-06-20", "2026-06-30")},
                        timestamp="2026-06-26T18:00:00", source="iaeste_agenda_2026"))
        g.add_edge(Edge("org_iaeste", "ev_ga", "plans", 0.85))
    if with_calendar:
        g.add_node(Node("ev_cal", NodeType.EVENT, "GA (calendar entry)",
                        attrs={"date": "2026-06-26",
                               "participants": ["p_anna", "p_marco", "p_priya"]},
                        timestamp="2026-06-26T18:00:00", source="calendar"))
        g.add_edge(Edge("org_iaeste", "ev_cal", "plans", 0.9))

    q = Query(anchors=["p_anna", "p_marco", "p_priya"],
              placeholder=Placeholder(NodeType.EVENT,
                                      {"date": "2026-06-26",
                                       "participants": ["p_anna", "p_marco", "p_priya"]}))
    return g, q


def s2_movie(all_cues: bool = True):
    """S2 — 'that movie with that actor... the synth soundtrack... set in Tokyo'."""
    g = EvidenceGraph()
    g.add_node(Node("p_actor", NodeType.PERSON, "The actor"))
    g.add_node(Node("t_music", NodeType.TOPIC, "Synth soundtrack"))
    g.add_node(Node("pl_tokyo", NodeType.PLACE, "Tokyo"))
    movies = [
        ("m_target", "The movie he's recalling", "2019-01-01"),
        ("m_other1", "Earlier movie A", "2011-01-01"),
        ("m_other2", "Recent movie B", "2023-01-01"),
    ]
    for mid, label, ts in movies:
        g.add_node(Node(mid, NodeType.TOPIC, label, attrs={"kind": "movie"}, timestamp=ts))
        g.add_edge(Edge("p_actor", mid, "acted_in", 0.85))
    g.add_edge(Edge("m_target", "t_music", "has_music", 0.8))
    g.add_edge(Edge("m_target", "pl_tokyo", "set_in", 0.7))

    anchors = ["p_actor", "t_music", "pl_tokyo"] if all_cues else ["p_actor"]
    q = Query(anchors=anchors,
              placeholder=Placeholder(NodeType.TOPIC, {"kind": "movie"}))
    return g, q


def s3_call():
    """S3 — 'You have a call from Person X' (event-triggered, predictive)."""
    g = EvidenceGraph()
    g.add_node(Node("p_x", NodeType.PERSON, "Person X"))
    g.add_node(Node("org_y", NodeType.ORG, "Organization Y"))
    g.add_edge(Edge("p_x", "org_y", "member_of", 0.9))

    # the meeting 12 hours from NOW -> temporal boost
    g.add_node(Node("ev_meet", NodeType.EVENT, "Meeting X",
                    timestamp="2026-06-22T22:00:00"))
    g.add_edge(Edge("org_y", "ev_meet", "hosts", 0.85))

    # the live assertion: docs missing, needed for that meeting
    g.add_node(Node("a_docs", NodeType.ASSERTION, "Documents missing from Org Y",
                    timestamp="2026-06-20T09:00:00"))
    g.add_edge(Edge("a_docs", "ev_meet", "needed_for", 0.9))
    g.add_edge(Edge("a_docs", "org_y", "about", 0.8))

    # distractor assertion: stale, tied to a past event
    g.add_node(Node("ev_old", NodeType.EVENT, "Org Y summer party",
                    timestamp="2026-05-20T19:00:00"))
    g.add_edge(Edge("org_y", "ev_old", "hosts", 0.85))
    g.add_node(Node("a_photos", NodeType.ASSERTION, "Party photos not yet shared",
                    timestamp="2026-05-21T09:00:00"))
    g.add_edge(Edge("a_photos", "ev_old", "relates_to", 0.9))

    # unreachable distractor: assertion about an org Person X is NOT in
    g.add_node(Node("org_z", NodeType.ORG, "Organization Z"))
    g.add_node(Node("a_invoice", NodeType.ASSERTION, "Invoice unpaid at Org Z"))
    g.add_edge(Edge("a_invoice", "org_z", "about", 0.9))

    q = Query(anchors=["p_x"],
              placeholder=Placeholder(NodeType.ASSERTION, {}),
              predictive=True)
    return g, q


def s4_travel():
    """S4 — 'he goes to ^£*#>' (corrupted destination; only history exists)."""
    g = EvidenceGraph()
    g.add_node(Node("p_x", NodeType.PERSON, "Person X"))
    trips = [
        ("pl_greece", "Greece", "2025-08-10"),
        ("pl_japan", "Japan", "2024-11-02"),
        ("pl_china", "China", "2026-03-15"),  # most recent
    ]
    for pid, label, ts in trips:
        g.add_node(Node(pid, NodeType.PLACE, label, timestamp=ts))
        g.add_edge(Edge("p_x", pid, "traveled_to", 0.9, timestamp=ts))

    q = Query(anchors=["p_x"],
              placeholder=Placeholder(NodeType.PLACE, {}),
              signature="s4_where_does_x_go")
    return g, q
