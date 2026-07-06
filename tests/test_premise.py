"""False-premise handling: unresolved question entities are abstention signals."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _work_graph():
    """User has scattered work memories — but nothing about 'Google'."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("a_proj", NodeType.ASSERTION, "Working long hours on a project",
                    timestamp="2026-05-01T00:00:00"))
    g.add_node(Node("a_cert", NodeType.ASSERTION, "Got a cloud certification",
                    timestamp="2026-03-01T00:00:00"))
    g.add_edge(Edge("a_proj", "p_u", "about", 0.8))
    g.add_edge(Edge("a_cert", "p_u", "about", 0.8))
    return g


def test_unknown_reference_plus_weak_evidence_refuses():
    """'How long did I work before my job at Google?' — Google is not in memory.
    Weak candidates + an unverified premise must NOT produce a rambling guess."""
    g = _work_graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.ASSERTION, {}),
              unresolved=["Google"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN
    assert "Google" in v.chain.render()          # the unknown premise is named


def test_strong_direct_evidence_still_answers_but_never_confirmed():
    """If the answer itself is well-supported, an unverified side reference
    downgrades CONFIRMED -> ASSUMED instead of refusing."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("t_a", NodeType.TOPIC, "Marathon training",
                    attrs={"kind": "hobby"}))
    g.add_node(Node("t_b", NodeType.TOPIC, "Chess club",
                    attrs={"kind": "hobby"}, timestamp="2026-01-01T00:00:00"))
    g.add_edge(Edge("p_u", "t_a", "does", 0.9))
    g.add_edge(Edge("p_u", "t_b", "does", 0.6))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TOPIC, {"kind": "hobby"}),
              unresolved=["that thing Coach Riley said"])
    v = Engine(g).answer(q, now=NOW)
    # two candidates, not tied (0.54 vs 0.36) -> top answered; premise caveat capped it
    assert v.klass != VerdictClass.CONFIRMED


def test_no_unresolved_behaves_as_before():
    g = _work_graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.ASSERTION, {}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED       # recency default still available


def test_suggest_mode_returns_grounding_bundle():
    """Preference questions: the engine retrieves confirmed facts; it never
    invents the suggestion itself. The bundle IS the verdict."""
    from apm import Edge, Node, NodeType, Placeholder, Query
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("t_cam", NodeType.TOPIC, "Sony A7 IV camera"))
    g.add_node(Node("t_flash", NodeType.TOPIC, "Godox V1 flash"))
    g.add_edge(Edge("p_u", "t_cam", "owns", 0.9))
    g.add_edge(Edge("p_u", "t_flash", "owns", 0.85))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TOPIC, {}),
              mode="suggest")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert "grounded" in v.default_rule
    text = v.chain.render()
    assert "Sony A7 IV" in text and "Godox V1" in text


def test_suggest_mode_with_empty_graph_refuses():
    from apm import Node, NodeType, Placeholder, Query
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TOPIC, {}),
              mode="suggest")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_unverified_update_guard_prevents_stale_confirm():
    """Invention 830ce83f: 'Where did Rachel move to?' with only a stale
    lives_in edge must NOT confirm the old city."""
    from apm import Edge, Node, NodeType, Placeholder, Query
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_chi", NodeType.PLACE, "Chicago"))
    g.add_edge(Edge("p_r", "pl_chi", "lives_in", 0.9))
    q = Query(anchors=["p_r"], placeholder=Placeholder(NodeType.PLACE, {}),
              topic_text="Where did Rachel move to after her recent relocation?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.CONFIRMED     # the anti-invention guarantee
