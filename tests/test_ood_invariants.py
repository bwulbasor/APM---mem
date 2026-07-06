"""Out-of-distribution faithfulness invariants (from the 2026-07-06 law-firm
smoke test, where 2 CONFIRMED-tier answers were confidently wrong on a fresh
domain). These lock the GENERAL rules, not the specific scenarios:

 I1  ordering must not CONFIRM when the question names a modification event
     (reschedule/postpone/cancel...) that no candidate node represents
     (lawfirm_0014: compared a deposition to a hearing's SCHEDULED date).
 I2a auto_supersede must order functional edges by the value's own timestamp
     when the edge timestamp is missing (lawfirm_0015: both 'lives_in' edges
     had ts=None -> the NEW address was wrongly superseded).
 I2b resolve must never CONFIRM a value when a strictly more-recent,
     conflicting value exists for the same subject; it answers the newest,
     tagged ASSUMED (lawfirm_0015: CONFIRMED a client's OLD address).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apm import Engine, EvidenceGraph, Node, NodeType, VerdictClass
from apm.engine import Query, Placeholder
from apm.graph import Edge, parse_ts

NOW = parse_ts("2023-07-01T00:00:00")


def _order_graph():
    g = EvidenceGraph()
    g.add_node(Node("u", NodeType.PERSON, "User"))
    g.add_node(Node("e1", NodeType.EVENT, "Deposition of Craig Simmons",
                    timestamp="2023-05-16T00:00:00", attrs={"date": "2023-05-16"}))
    g.add_node(Node("e2", NodeType.EVENT, "Bell v. Simmons custody hearing",
                    timestamp="2023-05-19T00:00:00", attrs={"date": "2023-05-19"}))
    return g


def _order_query(topic):
    return Query(anchors=["u"], placeholder=Placeholder(NodeType.EVENT, {}),
                 mode="first", options=["e1", "e2"], topic_text=topic)


def test_i1_ordering_refuses_confirm_on_unrepresented_modification_event():
    g = _order_graph()
    q = _order_query("which happened first, the deposition or the first "
                     "rescheduling of the hearing?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED          # 'rescheduling' unrepresented
    assert "modification-event" in (v.default_rule or "")


def test_i1_ordering_still_confirms_when_events_are_represented():
    """Positive control: the guard must NOT nuke legitimate dated ordering."""
    g = _order_graph()
    q = _order_query("which happened first, the deposition or the hearing?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED


def test_i2a_supersede_uses_node_timestamp_when_edge_ts_missing():
    g = EvidenceGraph()
    g.add_node(Node("p", NodeType.PERSON, "Rosa"))
    g.add_node(Node("addr_old", NodeType.PLACE, "118 Maple Court",
                    timestamp="2023-03-14T00:00:00"))
    g.add_node(Node("addr_new", NodeType.PLACE, "4471 Birchwood Lane",
                    timestamp="2023-05-26T00:00:00"))
    g.add_edge(Edge("p", "addr_old", "lives_in", 0.9, timestamp=None))
    g.add_edge(Edge("p", "addr_new", "lives_in", 0.9, timestamp=None))
    g.auto_supersede()
    # the OLDER value is superseded, not the newer (the pre-fix bug)
    assert ("p", "lives_in", "addr_old") in g.superseded
    assert ("p", "lives_in", "addr_new") not in g.superseded


def test_i2b_resolve_never_confirms_a_superseded_value():
    g = EvidenceGraph()
    g.add_node(Node("rosa", NodeType.PERSON, "Rosa Delgado"))
    g.add_node(Node("old", NodeType.ASSERTION,
                    "Mailing address for Rosa Delgado is 118 Maple Court",
                    timestamp="2023-03-14T00:00:00"))
    g.add_node(Node("new", NodeType.ASSERTION,
                    "Mailing address updated to 4471 Birchwood Lane",
                    timestamp="2023-05-26T00:00:00"))
    # both assertions are about Rosa; give OLD an extra anchor so it OUTRANKS
    # NEW (the invention setup — the stale value is the strongest candidate)
    g.add_node(Node("eeoc", NodeType.ORG, "EEOC filing"))
    g.add_edge(Edge("rosa", "old", "stated", 0.9))
    g.add_edge(Edge("rosa", "new", "stated", 0.9))
    g.add_edge(Edge("old", "rosa", "about", 0.9))
    g.add_edge(Edge("new", "rosa", "about", 0.9))
    g.add_edge(Edge("old", "eeoc", "about", 0.9))
    g.add_edge(Edge("eeoc", "old", "about", 0.9))
    q = Query(anchors=["rosa"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"attribute": "mailing address"}),
              topic_text="what is the current mailing address for Rosa Delgado")
    v = Engine(g).answer(q, now=NOW, write_back=False)
    assert v.klass != VerdictClass.CONFIRMED                  # stale never confirmed
    assert v.answer is not None and "birchwood" in v.answer.label.lower()  # newest wins
