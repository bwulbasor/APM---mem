"""Count-mode queries: 'how many X...?' (Phase 6 gap-fix, from LongMemEval multi-session)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _errands_graph():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    for i, label in enumerate(["Return jacket to store", "Pick up tailored suit",
                               "Return shoes"]):
        nid = f"t_{i}"
        g.add_node(Node(nid, NodeType.TOPIC, label, attrs={"kind": "errand"}))
        g.add_edge(Edge("p_u", nid, "needs_to_do", 0.9))
    # distractor of the same type but different kind
    g.add_node(Node("t_x", NodeType.TOPIC, "Learn Spanish", attrs={"kind": "goal"}))
    g.add_edge(Edge("p_u", "t_x", "needs_to_do", 0.9))
    return g


def test_count_with_parsed_predicate_is_assumed():
    """v0.3 (from invention 0a995998): membership from PARSED constraints is an
    LLM artifact — the count caps at ASSUMED with the enumeration disclosed."""
    g = _errands_graph()
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TOPIC, {"kind": "errand"}),
              mode="count")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.default_rule == "count over parsed membership predicate"
    assert v.answer.attrs["count"] == 3
    text = v.chain.render()
    assert "Return jacket" in text and "tailored suit" in text  # enumeration shown
    assert "Learn Spanish" not in text                          # distractor excluded


def test_count_with_named_options_is_confirmed():
    """When the ASKER names the set, membership is verified -> CONFIRMED."""
    g = _errands_graph()
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TOPIC, {"kind": "errand"}),
              mode="count", options=["t_0", "t_1", "t_2"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.attrs["count"] == 3


def test_count_zero_supported_is_unknown():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.EVENT, {}), mode="count")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_count_never_counts_own_inferences():
    """An earlier ASSUMED write-back must not inflate a later count."""
    g = _errands_graph()
    eng = Engine(g)
    # a resolve query that write-backs an ASSUMED node of the same type/kind
    q1 = Query(anchors=["p_u"],
               placeholder=Placeholder(NodeType.TOPIC, {"kind": "errand"}))
    eng.answer(q1, now=NOW)   # ties -> recency default may write back an assumed node
    q2 = Query(anchors=["p_u"],
               placeholder=Placeholder(NodeType.TOPIC, {"kind": "errand"}),
               mode="count")
    v = eng.answer(q2, now=NOW)
    assert v.answer.attrs["count"] == 3   # still 3, not 4


def test_count_weak_member_degrades_to_assumed():
    g = _errands_graph()
    g.add_node(Node("t_w", NodeType.TOPIC, "Maybe exchange the hat",
                    attrs={"kind": "errand"}))
    g.add_edge(Edge("p_u", "t_w", "needs_to_do", 0.4))   # flimsy evidence
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TOPIC, {"kind": "errand"}),
              mode="count")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert "incomplete or inflated" in v.chain.render()
