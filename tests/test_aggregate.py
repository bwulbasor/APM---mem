"""Aggregation route (apm/aggregate.py): quoted member table, mechanical
row verification, dedupe, arithmetic; PARTIAL when completeness is doubtful."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import apm.aggregate as agg
from apm import EvidenceGraph, Node, NodeType, Placeholder, Query

SESSIONS = [
    {"id": "s1", "text": "User: I spent 3 days camping at Yosemite in May."},
    {"id": "s2", "text": "User: The Utah trip was a 5-day camping adventure."},
    {"id": "s3", "text": "User: I have 40 days of gym membership left."},
]


def _graph():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User", source="lme:s1"))
    return g


def _q(mode="sum"):
    return Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
                 mode=mode, topic_text="How many days did I spend camping?")


def _reply(payload):
    return lambda *a, **k: json.dumps(payload)


def test_sum_over_verified_rows(monkeypatch):
    monkeypatch.setattr(agg, "chat", _reply({"items": [
        {"item": "Yosemite camping", "quantity": 3,
         "quote": "I spent 3 days camping at Yosemite in May.", "session": "s1"},
        {"item": "Utah camping trip", "quantity": 5,
         "quote": "The Utah trip was a 5-day camping adventure.", "session": "s2"},
    ], "complete": True}))
    out = agg.aggregate_answer(_graph(), _q(), "How many days camping?", SESSIONS)
    assert out and out["value"] == 8.0 and not out["partial"]


def test_unverified_rows_dropped_and_marked_partial(monkeypatch):
    monkeypatch.setattr(agg, "chat", _reply({"items": [
        {"item": "Yosemite camping", "quantity": 3,
         "quote": "I spent 3 days camping at Yosemite in May.", "session": "s1"},
        {"item": "Utah camping trip", "quantity": 5,
         "quote": "The Utah trip was a 5-day camping adventure.", "session": "s2"},
        {"item": "Alps camping", "quantity": 7,
         "quote": "I camped a week in the Alps.", "session": "s2"},  # fabricated
    ], "complete": True}))
    out = agg.aggregate_answer(_graph(), _q(), "How many days camping?", SESSIONS)
    assert out["value"] == 8.0 and out["partial"] and out["dropped"] == 1


def test_count_mode_counts_deduped_rows(monkeypatch):
    monkeypatch.setattr(agg, "chat", _reply({"items": [
        {"item": "Yosemite camping", "quantity": 3,
         "quote": "I spent 3 days camping at Yosemite in May.", "session": "s1"},
        {"item": "yosemite camping", "quantity": 3,   # duplicate retelling
         "quote": "I spent 3 days camping at Yosemite in May.", "session": "s1"},
        {"item": "Utah camping trip", "quantity": 5,
         "quote": "The Utah trip was a 5-day camping adventure.", "session": "s2"},
    ], "complete": True}))
    out = agg.aggregate_answer(_graph(), _q(mode="count"),
                               "How many camping trips did I take?", SESSIONS)
    assert out["value"] == 2


def test_is_aggregation_detector():
    assert agg.is_aggregation(_q("count"), "anything")
    assert agg.is_aggregation(_q("sum"), "anything")
    # v1.12.1: single-stated-value "how many" questions stay with the
    # single-quote tier; only parsed count/sum modes aggregate
    assert not agg.is_aggregation(_q("resolve"), "How many bikes do I own?")
    assert not agg.is_aggregation(_q("span"), "How many days before X did Y?")


def test_single_row_table_falls_through(monkeypatch):
    monkeypatch.setattr(agg, "chat", _reply({"items": [
        {"item": "to-watch list", "quantity": 1,
         "quote": "I spent 3 days camping at Yosemite in May.", "session": "s1"},
    ], "complete": True}))
    assert agg.aggregate_answer(_graph(), _q("count"), "q", SESSIONS) is None


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("APM_NO_AGG", "1")
    assert agg.aggregate_answer(_graph(), _q(), "q", SESSIONS) is None
