"""v0.6: span mode (date arithmetic) + auto-contradiction (same-metric updates).

Both were queued from held-out failure CLASSES (date-between questions;
the $350k stale-value invention) and are general rules, not per-question fits.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _dated_graph():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_a", NodeType.EVENT, "Holiday market visit",
                    attrs={"date": "2023-12-03"}, timestamp="2023-12-03T12:00:00"))
    g.add_node(Node("ev_b", NodeType.EVENT, "Bought the iPad",
                    attrs={"date": "2023-12-15"}, timestamp="2023-12-15T18:00:00"))
    g.add_edge(Edge("p_u", "ev_a", "attended", 0.9))
    g.add_edge(Edge("p_u", "ev_b", "did", 0.9))
    return g


# ---- span mode ---------------------------------------------------------------
def test_span_confirmed_with_stated_dates():
    g = _dated_graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
              mode="span", options=["ev_a", "ev_b"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.attrs["days"] == 12
    assert "12 days" in v.answer.label
    assert "2023-12-03" in v.chain.render()      # both dates + the subtraction shown


def test_span_over_mention_stamps_is_assumed():
    g = _dated_graph()
    g.node("ev_b").attrs.pop("date")             # only a session mention-stamp left
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
              mode="span", options=["ev_a", "ev_b"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert "mention time" in v.chain.render()


def test_span_with_unresolved_endpoint_refuses():
    g = _dated_graph()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
              mode="span", options=["ev_a", "the thing that never happened"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN       # abstention traps stay refused


def test_span_with_undated_endpoint_refuses():
    g = _dated_graph()
    g.node("ev_b").attrs.pop("date")
    g.node("ev_b").timestamp = None
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
              mode="span", options=["ev_a", "ev_b"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


# ---- auto-contradiction --------------------------------------------------------
def _mortgage_graph():
    """The 852ce960 invention, reconstructed: pre-approval updated 350k -> 400k."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("org_wf", NodeType.ORG, "Wells Fargo"))
    g.add_node(Node("a_old", NodeType.ASSERTION, "Pre-approved for $350,000",
                    attrs={"metric": "pre_approval_amount", "value": "$350,000"},
                    timestamp="2023-02-01T00:00:00"))
    g.add_node(Node("a_new", NodeType.ASSERTION, "Pre-approved for $400,000",
                    attrs={"metric": "pre_approval_amount", "value": "$400,000"},
                    timestamp="2023-05-01T00:00:00"))
    for a in ("a_old", "a_new"):
        g.add_edge(Edge(a, "p_u", "about", 0.9))
        g.add_edge(Edge(a, "org_wf", "about", 0.9))
    return g


def test_auto_contradict_links_same_metric_different_value():
    g = _mortgage_graph()
    added = g.auto_contradict()
    assert added == 1
    rels = [(e.src, e.dst) for _, _, d in g.g.edges(data=True)
            for e in [d["obj"]] if e.rel == "contradicts"]
    assert ("a_new", "a_old") in rels            # newer attacks older, only


def test_auto_contradict_kills_the_350k_invention():
    """Query the pre-approval: the stale $350k must be suppressed, $400k wins."""
    g = _mortgage_graph()
    g.auto_contradict()
    q = Query(anchors=["p_u", "org_wf"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"metric": "pre_approval_amount"}))
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "a_new"
    assert any(s.kind == "contradiction" for s in v.chain.steps)


def test_auto_contradict_spares_different_subjects():
    """Same metric on DIFFERENT subjects (my 5K time vs your 5K time) never clash."""
    g = EvidenceGraph()
    g.add_node(Node("p_a", NodeType.PERSON, "Anna"))
    g.add_node(Node("p_b", NodeType.PERSON, "Bea"))
    g.add_node(Node("a_1", NodeType.ASSERTION, "Anna 5K: 25:50",
                    attrs={"metric": "5k_time", "value": "25:50"},
                    timestamp="2026-01-01T00:00:00"))
    g.add_node(Node("a_2", NodeType.ASSERTION, "Bea 5K: 22:10",
                    attrs={"metric": "5k_time", "value": "22:10"},
                    timestamp="2026-02-01T00:00:00"))
    g.add_edge(Edge("a_1", "p_a", "about", 0.9))
    g.add_edge(Edge("a_2", "p_b", "about", 0.9))
    assert g.auto_contradict() == 0


def test_auto_contradict_idempotent():
    g = _mortgage_graph()
    assert g.auto_contradict() == 1
    assert g.auto_contradict() == 0


def test_numbers_are_identity_in_resolution():
    """From held-out invention 852ce960: differing numbers = different entities."""
    from apm.entities import similarity
    assert similarity("$350,000 pre-approval", "Pre-approval of $400,000") == 0.0
    assert similarity("$350,000 pre-approval", "pre-approval of $350,000") > 90


def test_auto_contradict_fires_on_real_extraction_shapes():
    """The actual node shapes Gemini produced (amount attr, no 'metric' key)."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("a_old", NodeType.ASSERTION, "$350,000 pre-approval",
                    attrs={"amount": "350000", "lender": "Wells Fargo"},
                    timestamp="2023-02-01T00:00:00"))
    g.add_node(Node("a_new", NodeType.ASSERTION, "Pre-approval of $400,000",
                    attrs={"amount": "400000"},
                    timestamp="2023-08-11T00:00:00"))
    for a in ("a_old", "a_new"):
        g.add_edge(Edge(a, "p_u", "about", 0.9))
    assert g.auto_contradict() == 1
    rels = [(e.src, e.dst) for _, _, d in g.g.edges(data=True)
            for e in [d["obj"]] if e.rel == "contradicts"]
    assert ("a_new", "a_old") in rels


def test_auto_supersede_downweights_stale_functional_relations():
    """Invention 830ce83f: 'Rachel moved' — the old city must stop outranking."""
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_chi", NodeType.PLACE, "Chicago"))
    g.add_node(Node("pl_sub", NodeType.PLACE, "the suburbs"))
    g.add_edge(Edge("p_r", "pl_chi", "lives_in", 0.9,
                    timestamp="2023-01-01T00:00:00"))
    g.add_edge(Edge("p_r", "pl_sub", "lives_in", 0.9,
                    timestamp="2023-09-01T00:00:00"))
    assert g.auto_supersede() == 1
    weights = {(e.src, e.dst): e.weight for _, _, d in g.g.edges(data=True)
               for e in [d["obj"]]}
    assert weights[("p_r", "pl_sub")] == 0.9      # current: untouched
    assert weights[("p_r", "pl_chi")] < 0.4       # stale: suppressed
    assert g.auto_supersede() == 0                 # idempotent


def test_auto_supersede_query_returns_current_location():
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_chi", NodeType.PLACE, "Chicago"))
    g.add_node(Node("pl_sub", NodeType.PLACE, "the suburbs"))
    g.add_edge(Edge("p_r", "pl_chi", "lives_in", 0.9,
                    timestamp="2023-01-01T00:00:00"))
    g.add_edge(Edge("p_r", "pl_sub", "moved_to", 0.9,
                    timestamp="2023-09-01T00:00:00"))
    g.auto_supersede()
    q = Query(anchors=["p_r"], placeholder=Placeholder(NodeType.PLACE, {}),
              topic_text="Where did Rachel move to after her relocation?")
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "pl_sub"


def test_auto_supersede_ignores_nonfunctional_relations():
    """visited/attended legitimately repeat — never suppressed."""
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_a", NodeType.PLACE, "Paris"))
    g.add_node(Node("pl_b", NodeType.PLACE, "Rome"))
    g.add_edge(Edge("p_r", "pl_a", "visited", 0.9, timestamp="2023-01-01T00:00:00"))
    g.add_edge(Edge("p_r", "pl_b", "visited", 0.9, timestamp="2023-09-01T00:00:00"))
    assert g.auto_supersede() == 0
