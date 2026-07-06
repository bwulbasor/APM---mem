"""Hybrid tier v2 (apm/hybrid.py): transcript-grounded answers with
mechanically VERIFIED quotes. Contract: no verified quote -> no answer;
disabled -> silent; API failure -> silent; never labeled CONFIRMED."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import apm.hybrid as hybrid
from apm import EvidenceGraph, Node, NodeType, Placeholder, Query

SESSIONS = [{"id": "s1",
             "text": "User: I upgraded last week. My new internet plan is "
                     "500 Mbps fiber from Lumen.\nAssistant: Nice speed!"}]


def _graph():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User", source="lme:s1"))
    return g


def _q():
    return Query(anchors=["p_u"], placeholder=Placeholder(NodeType.ASSERTION, {}),
                 topic_text="What speed is my internet plan?")


def _reply(payload):
    return lambda *a, **k: json.dumps(payload)


def test_verified_quote_accepted(monkeypatch):
    monkeypatch.setattr(hybrid, "chat", _reply(
        {"answer": "Your plan is 500 Mbps fiber.",
         "quote": "My new internet plan is 500 Mbps fiber from Lumen.",
         "session": "s1"}))
    out = hybrid.text_supported_answer(_graph(), _q(), "What speed?", SESSIONS)
    assert out and out["answer"].startswith("Your plan is 500")
    assert out["session"] == "s1"


def test_fabricated_quote_rejected(monkeypatch):
    monkeypatch.setattr(hybrid, "chat", _reply(
        {"answer": "Your plan is 1 Gbps.",
         "quote": "My internet is one gigabit per second.",   # not in text
         "session": "s1"}))
    assert hybrid.text_supported_answer(_graph(), _q(), "q", SESSIONS) is None


def test_model_null_answer_respected(monkeypatch):
    monkeypatch.setattr(hybrid, "chat", _reply({"answer": None}))
    assert hybrid.text_supported_answer(_graph(), _q(), "q", SESSIONS) is None


def test_disabled_and_failure_are_silent(monkeypatch):
    monkeypatch.setenv("APM_NO_HYBRID", "1")
    assert hybrid.text_supported_answer(_graph(), _q(), "q", SESSIONS) is None
    monkeypatch.delenv("APM_NO_HYBRID")
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(hybrid, "chat", boom)
    assert hybrid.text_supported_answer(_graph(), _q(), "q", SESSIONS) is None


def test_refusal_language_detector():
    assert hybrid.looks_like_refusal("I do not know your internet speed.")
    assert hybrid.looks_like_refusal("There is no record of that purchase.")
    assert not hybrid.looks_like_refusal("Your plan is 500 Mbps fiber.")
