"""Offline tests for the ingestion utilities (no network — LLM calls not tested here)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from apm import EvidenceGraph, Node, NodeType
from apm.ingest import _extract_json, _norm, _resolve, _slug


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    raw = 'Here you go:\n```json\n{"nodes": []}\n```\nDone!'
    assert _extract_json(raw) == {"nodes": []}


def test_extract_json_with_prose_and_nesting():
    raw = 'Sure! {"a": {"b": [1, 2]}, "c": "x"} hope that helps'
    assert _extract_json(raw) == {"a": {"b": [1, 2]}, "c": "x"}


def test_extract_rejects_no_json():
    with pytest.raises(ValueError):
        _extract_json("I cannot answer that.")


def test_slug():
    assert _slug("General Assembly (2026)!") == "general_assembly_2026"


def test_resolve_exact_and_containment():
    g = EvidenceGraph()
    g.add_node(Node("p_anna", NodeType.PERSON, "Anna"))
    g.add_node(Node("org_iaeste", NodeType.ORG, "IAESTE Austria"))
    assert _resolve(g, "anna") == "p_anna"
    assert _resolve(g, "IAESTE") == "org_iaeste"       # containment
    assert _resolve(g, "Bob") is None


def test_resolve_short_labels_never_containment_match():
    g = EvidenceGraph()
    g.add_node(Node("p_x", NodeType.PERSON, "Maximilian"))
    assert _resolve(g, "Max") is None   # len<4: exact only, no fuzzy grab
