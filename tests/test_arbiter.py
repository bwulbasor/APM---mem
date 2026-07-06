"""LLM tie-arbiter (apm/arbiter.py) — SPEC §3.5 role 2 safety contract:
closed candidate list, ASSUMED-only, silent fallback on any failure."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from datetime import datetime

import apm.arbiter as arbiter
from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _fake_chat(reply):
    def chat(prompt, system=None, temperature=0.0, role=None,
             cacheable=True, json_mode=False, max_tokens=8192):
        return reply
    return chat


def test_arbitrate_picks_from_closed_list(monkeypatch):
    monkeypatch.setattr(arbiter, "chat",
                        _fake_chat(json.dumps({"pick": "white Adidas sneakers"})))
    i = arbiter.arbitrate("Which pair of shoes did I clean last month?",
                          [("Converse", ["owns -> User"]),
                           ("white Adidas sneakers", ["cleaned -> last month"])])
    assert i == 1


def test_arbitrate_rejects_labels_outside_list(monkeypatch):
    monkeypatch.setattr(arbiter, "chat",
                        _fake_chat(json.dumps({"pick": "Nike Air Max"})))
    i = arbiter.arbitrate("Which pair of shoes did I clean?",
                          [("Converse", []), ("white Adidas sneakers", [])])
    assert i is None                       # invention rejected -> fallback


def test_arbitrate_fails_closed_on_garbage(monkeypatch):
    monkeypatch.setattr(arbiter, "chat", _fake_chat("not json at all"))
    assert arbiter.arbitrate("q", [("a", []), ("b", [])]) is None


def test_arbitrate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("APM_NO_ARBITER", "1")
    assert arbiter.arbitrate("q", [("a", []), ("b", [])]) is None


def test_engine_tie_uses_arbiter_and_stays_assumed(monkeypatch):
    """Two candidates tied by evidence; a mocked arbiter picks the second;
    the verdict must be ASSUMED with the arbiter named in the chain."""
    monkeypatch.setattr(arbiter, "chat",
                        _fake_chat(json.dumps({"pick": "Wine tasting"})))
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_a", NodeType.EVENT, "Pottery class",
                    timestamp="2026-02-10T18:00:00"))
    g.add_node(Node("ev_b", NodeType.EVENT, "Wine tasting",
                    timestamp="2026-03-02T10:00:00"))
    g.add_edge(Edge("p_u", "ev_a", "attends", 0.9))
    g.add_edge(Edge("p_u", "ev_b", "attends", 0.9))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              topic_text="Which tasting event did I go to?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.label == "Wine tasting"
    assert v.default_rule == "LLM-arbitrated tiebreak"
