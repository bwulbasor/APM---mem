"""Phase 2 tests: brain-inspired scenarios + adversarial gate hardening.

The theme: wherever human memory fails soft (interference, confabulation,
forgetting), APM must fail HONEST — suppress, tag, or refuse. Never invent.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Engine, Node, NodeType, Placeholder, Query, VerdictClass
from apm.graph import Edge
from scenarios.brain import (NOW, b1_interference, b2_episodic_sequence,
                             b3_semantic_priming, b4_forgetting,
                             b5_source_confusion, b6_salience)


# ---- B1: retroactive interference --------------------------------------------
def test_b1_newer_fact_wins():
    g, q = b1_interference()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "a_berlin"
    assert v.klass != VerdictClass.UNKNOWN


def test_b1_old_fact_is_suppressed_not_tied():
    """Without contradiction handling these two would tie and recency-default;
    with it, Paris must be actively suppressed (visible in the chain)."""
    g, q = b1_interference()
    v = Engine(g).answer(q, now=NOW)
    assert any(s.kind == "contradiction" for s in v.chain.steps)


def test_b1_no_contradiction_edge_means_tie_default():
    """Control: remove the contradicts edge -> engine falls back to recency
    default and must say ASSUMED (it can't structurally rule Paris out)."""
    g, q = b1_interference()
    # rebuild without the contradicts edge
    from apm import EvidenceGraph
    g2 = EvidenceGraph()
    for n in g.nodes.values():
        g2.add_node(n)
    for _, _, data in g.g.edges(data=True):
        if data["obj"].rel != "contradicts":
            g2.add_edge(data["obj"])
    v = Engine(g2).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.id == "a_berlin"          # recency still picks Berlin
    assert v.default_rule == "recency"


# ---- B2: episodic sequence ----------------------------------------------------
def test_b2_next_event_recalled():
    g, q = b2_episodic_sequence()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "ev_brunch"          # what came after the gym
    assert v.klass != VerdictClass.UNKNOWN


# ---- B3: semantic priming ------------------------------------------------------
def test_b3_context_disambiguates_bank():
    g, q = b3_semantic_priming()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "t_bank_river"
    assert v.klass != VerdictClass.UNKNOWN


def test_b3_money_bank_unreachable_from_context():
    g, q = b3_semantic_priming()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id != "t_bank_money"


# ---- B4: the forgetting curve ---------------------------------------------------
def test_b4_recallable_before_decay():
    g, q = b4_forgetting()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer is not None and v.answer.id == "pl_leipzig"


def test_b4_decayed_memory_is_honest_unknown():
    g, q = b4_forgetting()
    g.apply_decay(now=NOW, half_life_days=730)   # ~11 years at 2y half-life
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN
    assert v.answer is None


def test_b4_rehearsal_protects_from_decay():
    """Hebbian counterpart: recalling the fact re-stamps the edges, so the
    same decay applied afterwards no longer erases it."""
    g, q = b4_forgetting()
    eng = Engine(g)
    v1 = eng.answer(q, now=NOW)                 # recall = rehearsal
    assert v1.answer.id == "pl_leipzig"
    g.apply_decay(now=NOW, half_life_days=730)
    v2 = eng.answer(q, now=NOW)
    assert v2.answer is not None                 # memory survived


# ---- B5: source confusion guard --------------------------------------------------
def test_b5_confabulation_is_tagged_never_confirmed():
    g, q = b5_source_confusion()
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.CONFIRMED     # the anti-false-memory guarantee
    if v.klass == VerdictClass.ASSUMED:
        assert v.default_rule is not None        # and the guess is tagged


# ---- B6: salience -----------------------------------------------------------------
def test_b6_salient_memory_dominates():
    g, q = b6_salience()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "ev_flight"


# ---- adversarial: UNKNOWN is the right answer ---------------------------------------
def test_near_miss_date_does_not_match():
    """An event exists one day off the requested date — must NOT be served."""
    from apm import EvidenceGraph
    g = EvidenceGraph()
    g.add_node(Node("p_a", NodeType.PERSON, "A"))
    g.add_node(Node("ev", NodeType.EVENT, "Workshop",
                    attrs={"date": "2026-06-27"}))
    g.add_edge(Edge("p_a", "ev", "attends", 0.9))
    q = Query(anchors=["p_a"],
              placeholder=Placeholder(NodeType.EVENT, {"date": "2026-06-26"}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_popular_but_unconnected_candidate_never_surfaces():
    """A high-confidence node with NO path from the anchors is invisible —
    prior fame is not evidence (the anti-prior-bias guarantee)."""
    from apm import EvidenceGraph
    g = EvidenceGraph()
    g.add_node(Node("p_a", NodeType.PERSON, "A"))
    g.add_node(Node("ev_famous", NodeType.EVENT, "The famous gala",
                    attrs={"date": "2026-06-26"}, confidence=0.99))
    q = Query(anchors=["p_a"],
              placeholder=Placeholder(NodeType.EVENT, {"date": "2026-06-26",
                                                       "participants": ["p_a"]}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_empty_graph_is_unknown():
    from apm import EvidenceGraph
    g = EvidenceGraph()
    g.add_node(Node("p_a", NodeType.PERSON, "A"))
    q = Query(anchors=["p_a"], placeholder=Placeholder(NodeType.EVENT, {}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_hebbian_reinforcement_strengthens_used_edges():
    from scenarios.fixtures import s2_movie, NOW as FNOW
    g, q = s2_movie()
    before = [data["obj"].weight for _, _, data in g.g.edges(data=True)
              if data["obj"].rel == "acted_in"]
    Engine(g).answer(q, now=FNOW)
    after_map = {(data["obj"].src, data["obj"].dst): data["obj"].weight
                 for _, _, data in g.g.edges(data=True)}
    assert after_map[("p_actor", "m_target")] > max(before) - 1e-9
