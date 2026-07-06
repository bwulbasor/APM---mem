"""Ordering-mode queries: 'which came first/last, A or B?' (Phase 6 gap-fix)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _graph(date_b="2026-03-02T10:00:00"):
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    # stated dates (attrs.date), not bare mention-stamps — CONFIRMED ordering
    # requires them since the v1.5 mention-stamp guard (dev2 invention)
    g.add_node(Node("ev_a", NodeType.EVENT, "Pottery class",
                    timestamp="2026-02-10T18:00:00",
                    attrs={"date": "2026-02-10"}))
    g.add_node(Node("ev_b", NodeType.EVENT, "Wine tasting", timestamp=date_b,
                    attrs={"date": date_b[:10]}))
    g.add_edge(Edge("p_u", "ev_a", "attended", 0.9))
    g.add_edge(Edge("p_u", "ev_b", "attended", 0.9))
    return g


def test_first_with_named_options_is_confirmed():
    g = _graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              mode="first", options=["ev_a", "ev_b"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.id == "ev_a"
    assert "2026-02-10" in v.chain.render()   # both dates in the chain


def test_last_without_options_or_constraints_is_capped_at_assumed():
    """Ordering an UNCONSTRAINED candidate set answers 'what's my newest node',
    not the user's question — it may be right, but it is never knowledge."""
    g = _graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              mode="last")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.id == "ev_b"
    assert "unconstrained" in v.chain.render()


def test_undated_candidate_degrades_to_assumed():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_a", NodeType.EVENT, "Pottery class",
                    timestamp="2026-02-10T18:00:00"))
    g.add_node(Node("ev_b", NodeType.EVENT, "Wine tasting",
                    timestamp="2026-03-02T10:00:00"))
    g.add_node(Node("ev_c", NodeType.EVENT, "Mystery event"))  # no date
    for ev in ("ev_a", "ev_b", "ev_c"):
        g.add_edge(Edge("p_u", ev, "attended", 0.9))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              mode="first", options=["ev_a", "ev_b", "ev_c"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED      # can't rule out the undated one
    assert v.answer.id == "ev_a"
    assert "Mystery event" in v.chain.render()  # exclusion is disclosed


def test_single_dated_candidate_is_unknown():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_a", NodeType.EVENT, "Pottery class",
                    timestamp="2026-02-10T18:00:00"))
    g.add_edge(Edge("p_u", "ev_a", "attended", 0.9))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              mode="first", options=["ev_a", "the missing one"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN
