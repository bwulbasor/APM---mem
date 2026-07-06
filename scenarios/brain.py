"""Brain-inspired scenarios B1-B6 — each models a documented human-memory
phenomenon and defines what CORRECT machine behavior looks like.

B1 interference/updating   — "Bob moved" (retroactive interference; newer fact wins)
B2 episodic sequence       — "what did I do after the gym?" (episodic chaining)
B3 semantic priming        — "the bank" (context disambiguates an ambiguous cue)
B4 forgetting curve        — unrehearsed memories fade to honest UNKNOWN
B5 source confusion guard  — two similar episodes; brain confabulates, APM tags ASSUMED
B6 salience / flashbulb    — emotionally weighted memories dominate recall
"""
from __future__ import annotations

from datetime import datetime

from apm import Edge, EvidenceGraph, Node, NodeType, Placeholder, Query

NOW = datetime.fromisoformat("2026-07-01T12:00:00")


def b1_interference():
    """Bob lived in Paris (2023); a newer fact says he moved to Berlin (2026).
    Human memory: retroactive interference — the new fact should win.
    Failure mode we must avoid: confidently answering Paris."""
    g = EvidenceGraph()
    g.add_node(Node("p_bob", NodeType.PERSON, "Bob"))
    g.add_node(Node("a_paris", NodeType.ASSERTION, "Bob lives in Paris",
                    attrs={"topic": "residence"}, timestamp="2023-04-01T00:00:00"))
    g.add_node(Node("a_berlin", NodeType.ASSERTION, "Bob lives in Berlin",
                    attrs={"topic": "residence"}, timestamp="2026-05-10T00:00:00"))
    g.add_edge(Edge("a_paris", "p_bob", "about", 0.9))
    g.add_edge(Edge("a_berlin", "p_bob", "about", 0.9))
    g.add_edge(Edge("a_berlin", "a_paris", "contradicts", 0.9,
                    timestamp="2026-05-10T00:00:00"))
    q = Query(anchors=["p_bob"],
              placeholder=Placeholder(NodeType.ASSERTION, {"topic": "residence"}))
    return g, q


def b2_episodic_sequence():
    """'What did I do after the gym on Monday?' — episodic memory is a chain of
    `precedes` links plus a shared time context that binds the episode together."""
    g = EvidenceGraph()
    g.add_node(Node("t_monday", NodeType.TIME, "Monday morning"))
    events = [
        ("ev_gym", "Gym session", "2026-06-29T07:00:00"),
        ("ev_brunch", "Brunch with Lea", "2026-06-29T09:30:00"),
        ("ev_library", "Library study block", "2026-06-29T11:30:00"),
    ]
    for eid, label, ts in events:
        g.add_node(Node(eid, NodeType.EVENT, label, timestamp=ts))
        g.add_edge(Edge(eid, "t_monday", "part_of", 0.8))
    g.add_edge(Edge("ev_gym", "ev_brunch", "precedes", 0.9))
    g.add_edge(Edge("ev_brunch", "ev_library", "precedes", 0.9))

    q = Query(anchors=["ev_gym", "t_monday"],
              placeholder=Placeholder(NodeType.EVENT, {}))
    return g, q


def b3_semantic_priming():
    """'We sat by the bank' — said while talking about a fishing trip.
    Spreading activation from context nodes disambiguates the cue,
    exactly like semantic priming in humans (river-bank, not the branch)."""
    g = EvidenceGraph()
    g.add_node(Node("t_river", NodeType.PLACE, "The river"))
    g.add_node(Node("p_sam", NodeType.PERSON, "Sam (fishing buddy)"))
    g.add_node(Node("t_fishing", NodeType.TOPIC, "Fishing"))
    g.add_node(Node("t_bank_river", NodeType.TOPIC, "River bank",
                    attrs={"word": "bank"}))
    g.add_node(Node("t_bank_money", NodeType.TOPIC, "Bank branch",
                    attrs={"word": "bank"}))
    g.add_node(Node("t_finance", NodeType.TOPIC, "Finance errands"))

    g.add_edge(Edge("t_river", "t_bank_river", "has_part", 0.85))
    g.add_edge(Edge("p_sam", "t_fishing", "enjoys", 0.8))
    g.add_edge(Edge("t_fishing", "t_river", "done_at", 0.85))
    g.add_edge(Edge("t_finance", "t_bank_money", "involves", 0.85))

    q = Query(anchors=["t_river", "p_sam"],
              placeholder=Placeholder(NodeType.TOPIC, {"word": "bank"},
                                      required=frozenset({"word"})))
    return g, q


def b4_forgetting():
    """A friend from 2015, never contacted since. Before decay the fact is
    recallable; after a decade of decay the honest answer is UNKNOWN —
    the forgetting curve, not a hallucinated guess."""
    g = EvidenceGraph()
    g.add_node(Node("p_me", NodeType.PERSON, "Me"))
    g.add_node(Node("p_old", NodeType.PERSON, "Old friend from 2015"))
    g.add_node(Node("pl_leipzig", NodeType.PLACE, "Leipzig",
                    timestamp="2015-06-01T00:00:00"))
    g.add_edge(Edge("p_me", "p_old", "knows", 0.9, timestamp="2015-06-01T00:00:00"))
    g.add_edge(Edge("p_old", "pl_leipzig", "lived_in", 0.9,
                    timestamp="2015-06-01T00:00:00"))
    q = Query(anchors=["p_me"], placeholder=Placeholder(NodeType.PLACE, {}))
    return g, q


def b5_source_confusion():
    """Two dinners at the same restaurant, months apart. Asked 'that dinner at
    Luigi's where we talked startups' — the topic isn't recorded on either.
    A human confabulates one confidently (false memory). APM must answer
    ASSUMED at best, never CONFIRMED."""
    g = EvidenceGraph()
    g.add_node(Node("pl_luigis", NodeType.PLACE, "Luigi's restaurant"))
    g.add_node(Node("ev_dinner1", NodeType.EVENT, "Dinner with Sam at Luigi's",
                    attrs={"venue": "luigis"}, timestamp="2026-01-15T20:00:00"))
    g.add_node(Node("ev_dinner2", NodeType.EVENT, "Dinner with Alex at Luigi's",
                    attrs={"venue": "luigis"}, timestamp="2026-04-22T20:00:00"))
    g.add_edge(Edge("ev_dinner1", "pl_luigis", "held_at", 0.9))
    g.add_edge(Edge("ev_dinner2", "pl_luigis", "held_at", 0.9))

    q = Query(anchors=["pl_luigis"],
              placeholder=Placeholder(NodeType.EVENT,
                                      {"venue": "luigis", "topic": "startups"}))
    return g, q


def b6_salience():
    """Flashbulb memory: on the same trip, the emotionally charged event
    (near-missed flight) is burned in with a stronger trace than the mundane
    one (hotel breakfast). Recall of 'that trip' surfaces the salient event."""
    g = EvidenceGraph()
    g.add_node(Node("ev_trip", NodeType.EVENT, "Rome trip 2025"))
    g.add_node(Node("ev_flight", NodeType.EVENT, "Nearly missed the flight",
                    timestamp="2025-09-10T06:00:00"))
    g.add_node(Node("ev_breakfast", NodeType.EVENT, "Hotel breakfast",
                    timestamp="2025-09-11T08:00:00"))
    # salience = association strength: adrenaline writes stronger edges
    g.add_edge(Edge("ev_trip", "ev_flight", "includes", 0.95))
    g.add_edge(Edge("ev_trip", "ev_breakfast", "includes", 0.55))

    q = Query(anchors=["ev_trip"], placeholder=Placeholder(NodeType.EVENT, {}))
    return g, q
