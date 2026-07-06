"""Phase 5 tests: time-window constraints + the event-trigger API."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _week_graph():
    g = EvidenceGraph()
    g.add_node(Node("p_a", NodeType.PERSON, "A"))
    g.add_node(Node("ev_in", NodeType.EVENT, "Standup",
                    attrs={"date": "2026-06-24"}))
    g.add_node(Node("ev_out", NodeType.EVENT, "Retro",
                    attrs={"date": "2026-07-08"}))
    g.add_edge(Edge("p_a", "ev_in", "attends", 0.9))
    g.add_edge(Edge("p_a", "ev_out", "attends", 0.9))
    return g


def test_date_range_selects_event_inside_window():
    g = _week_graph()
    q = Query(anchors=["p_a"],
              placeholder=Placeholder(NodeType.EVENT,
                                      {"date_range": ("2026-06-22", "2026-06-28")}))
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "ev_in"
    assert v.klass == VerdictClass.CONFIRMED   # unique strict hit inside window


def test_date_range_refuses_when_nothing_inside():
    g = _week_graph()
    q = Query(anchors=["p_a"],
              placeholder=Placeholder(NodeType.EVENT,
                                      {"date_range": ("2026-08-01", "2026-08-07")}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_on_event_api_matches_s3():
    from scenarios.fixtures import s3_call, NOW as FNOW
    g, q = s3_call()
    v = Engine(g).on_event(["p_x"], now=FNOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.id == "a_docs"
