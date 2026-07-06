"""Cued recall (apm/recall.py) — question-conditioned second-chance
extraction. Safety contract: overlay only (base graph untouched), adoption
only upgrades verdicts, implied cued edges stay ASSUMED-capped."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from datetime import datetime

import apm.ingest as ingest_mod
from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass
from apm.gate import Verdict, Chain
from apm.recall import adopt, cued_recall, needs_recall, select_sessions

NOW = datetime.fromisoformat("2026-06-22T10:00:00")

SESSIONS = [
    {"id": "2026/05/01", "ts": "2026-05-01T10:00:00",
     "text": "User: I redeemed a $5 coupon on coffee creamer last Sunday, "
             "it was in my email inbox.\n"
             "Assistant: Target sends coupons to email subscribers."},
    {"id": "2026/05/02", "ts": "2026-05-02T10:00:00",
     "text": "User: my cat likes shrimp."},
]


def _base_graph():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User", source="lme:2026/05/01"))
    g.add_node(Node("ev_r", NodeType.EVENT, "coupon redemption",
                    timestamp="2026-05-01T10:00:00", source="lme:2026/05/01"))
    g.add_edge(Edge("p_u", "ev_r", "attends", 0.9))
    return g


def _fake_extraction(payload):
    """Monkeypatch-able _ask_json returning a fixed extraction."""
    def ask(system, prompt, retries=3, role=None, cacheable=True):
        return payload
    return ask


def test_needs_recall_semantics():
    ch = Chain()
    assert needs_recall(Verdict(VerdictClass.UNKNOWN, None, 0.0, ch))
    weak = Verdict(VerdictClass.ASSUMED, Node("x", NodeType.ORG, "X"), 0.4, ch)
    strong_guess = Verdict(VerdictClass.ASSUMED, Node("x", NodeType.ORG, "X"), 0.9, ch)
    confirmed = Verdict(VerdictClass.CONFIRMED, Node("x", NodeType.ORG, "X"), 0.9, ch)
    assert needs_recall(weak)
    assert needs_recall(strong_guess)   # every ASSUMED is a guess
    assert not needs_recall(confirmed)


def test_select_sessions_prefers_topical(monkeypatch):
    g = _base_graph()
    q = Query(anchors=["p_u", "ev_r"],
              placeholder=Placeholder(NodeType.ORG, {}))
    top = select_sessions(g, q, "Where did I redeem the $5 coupon on coffee "
                          "creamer?", SESSIONS, k=1)
    assert top[0]["id"] == "2026/05/01"     # coupon session, not the cat


def test_cued_recall_converts_unknown_and_keeps_base_clean(monkeypatch):
    g = _base_graph()
    n_base_nodes = len(g.nodes)
    monkeypatch.setattr(ingest_mod, "_ask_json", _fake_extraction({
        "nodes": [{"label": "User", "type": "person", "attrs": {}},
                  {"label": "coupon redemption", "type": "event", "attrs": {},
                   "timestamp": "2026-05-01T10:00:00"},
                  {"label": "Target", "type": "org", "attrs": {}}],
        "edges": [{"src": "coupon redemption", "dst": "Target",
                   "rel": "located_in", "weight": 0.5, "implied": True},
                  {"src": "User", "dst": "Target",
                   "rel": "related_to", "weight": 0.9}]}))
    q = Query(anchors=["p_u", "ev_r"], placeholder=Placeholder(NodeType.ORG, {}),
              topic_text="Where did I redeem the coupon?")
    first = Engine(g).answer(q, now=NOW)
    assert first.klass == VerdictClass.UNKNOWN      # base graph can't answer
    v2, meta = cued_recall(g, q, "Where did I redeem the coupon?",
                           SESSIONS, now=NOW)
    assert v2 is not None and v2.answer.label == "Target"
    assert v2.klass == VerdictClass.ASSUMED          # implied edge: capped
    assert len(g.nodes) == n_base_nodes              # base graph untouched
    final, adopted = adopt(first, v2)
    assert adopted and final.answer.label == "Target"


def test_cued_evidence_cannot_confirm(monkeypatch):
    """Live invention 51a45a95-cued: the cued extractor stamped the
    question's constraint attrs onto its nodes -> strict direct-lookup
    CONFIRM. Anything resting on cued-sourced evidence caps at ASSUMED."""
    monkeypatch.setattr(ingest_mod, "_ask_json", _fake_extraction({
        "nodes": [{"label": "User", "type": "person", "attrs": {}},
                  {"label": "Cartwheel app", "type": "org",
                   "attrs": {"item": "coffee creamer"}}],   # question echo!
        "edges": [{"src": "User", "dst": "Cartwheel app",
                   "rel": "related_to", "weight": 0.9}]}))
    g = _base_graph()
    q = Query(anchors=["p_u", "ev_r"],
              placeholder=Placeholder(NodeType.ORG,
                                      {"item": "coffee creamer"}),
              topic_text="Where did I redeem the coupon?")
    v2, meta = cued_recall(g, q, "Where did I redeem the coupon?",
                           SESSIONS, now=NOW)
    assert v2 is not None
    assert v2.klass != VerdictClass.CONFIRMED    # cued evidence never confirms


def test_repeat_query_cannot_confirm_own_writeback():
    """Invention class #5 (found while building cued recall, but generic):
    an ASSUMED verdict writes back a node carrying the query's constraints
    as attrs; asking AGAIN strict-matched that node and CONFIRMED our own
    guess. Stored assumptions must never confirm."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("or_a", NodeType.ORG, "Shop A",
                    timestamp="2026-01-01T00:00:00"))
    g.add_node(Node("or_b", NodeType.ORG, "Shop B",
                    timestamp="2026-02-01T00:00:00"))
    g.add_edge(Edge("p_u", "or_a", "related_to", 0.9))
    g.add_edge(Edge("p_u", "or_b", "related_to", 0.9))
    import os
    os.environ["APM_NO_ARBITER"] = "1"      # deterministic recency tiebreak
    try:
        q = Query(anchors=["p_u"],
                  placeholder=Placeholder(NodeType.ORG, {"item": "shirt"}))
        v1 = Engine(g).answer(q, now=NOW)          # writes back the guess
        assert v1.klass == VerdictClass.ASSUMED
        v2 = Engine(g).answer(q, now=NOW)          # ask the SAME question
        assert v2.klass != VerdictClass.CONFIRMED  # never confirm own guess
    finally:
        del os.environ["APM_NO_ARBITER"]


def test_adopt_never_displaces_same_class():
    """v1.10 policy (from the 84-question ablation: same-class displacement
    was the dominant loss mode): adopt only refusal->answer or
    ASSUMED->CONFIRMED(base evidence). Never ASSUMED->ASSUMED."""
    ch = Chain()
    conf = Verdict(VerdictClass.CONFIRMED, Node("a", NodeType.ORG, "A"), 0.9, ch)
    weak = Verdict(VerdictClass.ASSUMED, Node("b", NodeType.ORG, "B"), 0.3, ch)
    kept, adopted = adopt(conf, weak)
    assert kept is conf and not adopted
    stronger_assumed = Verdict(VerdictClass.ASSUMED,
                               Node("c", NodeType.ORG, "C"), 0.9, ch)
    kept, adopted = adopt(weak, stronger_assumed)
    assert kept is weak and not adopted        # same-class: never displace
    kept, adopted = adopt(weak, conf)
    assert kept is conf and adopted            # class upgrade: adopt
    unknown = Verdict(VerdictClass.UNKNOWN, None, 0.0, ch)
    kept, adopted = adopt(unknown, weak)
    assert kept is weak and adopted            # refusal -> answer: adopt
