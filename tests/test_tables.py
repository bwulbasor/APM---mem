"""Deterministic markdown-table ingestion (apm/tables.py) — from failure 7161e7e2."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass
from apm.tables import find_tables, ingest_table, parse_table, strip_tables

NOW = datetime.fromisoformat("2026-06-22T10:00:00")

ROTATION = """Here is the final rotation:

| Day | 8 am - 4 pm (Day Shift) | 4 pm - 12 am (Swing) |
| --- | --- | --- |
| Sunday | Admon | Magdy |
| Monday | Mostafa | Admon |

Let me know if you need changes."""


def test_find_and_strip():
    tables = find_tables(ROTATION)
    assert len(tables) == 1
    assert "Admon" in tables[0]
    assert "Admon" not in strip_tables(ROTATION)          # cells removed from prose
    assert "[table extracted separately]" in strip_tables(ROTATION)


def test_parse_shape():
    header, rows = parse_table(find_tables(ROTATION)[0])
    assert header[0] == "Day" and "Day Shift" in header[1]
    assert rows[0][0] == "Sunday" and rows[0][1] == "Admon"


def test_cells_become_linked_assertions():
    g = EvidenceGraph()
    g.add_node(Node("p_admon", NodeType.PERSON, "Admon"))
    added = ingest_table(g, find_tables(ROTATION)[0], "rotation sheet",
                         default_ts="2026-06-01T00:00:00")
    assert added == 4
    labels = [n.label for n in g.nodes.values()]
    assert any("Sunday" in l and "Admon" in l and "Day Shift" in l for l in labels)
    # the Admon cell is linked to the Admon person node -> reachable by activation
    q = Query(anchors=["p_admon"], placeholder=Placeholder(NodeType.ASSERTION, {}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.UNKNOWN
    assert "Admon" in v.answer.label


def test_no_tables_is_noop():
    assert find_tables("just some prose\nwith lines") == []


def test_table_never_floats_disconnected():
    """A table with no resolvable cell entities must still be reachable from
    its context anchor (the person it was shared with)."""
    from apm.graph import Edge
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    ingest_table(g, find_tables(ROTATION)[0], "rotation sheet",
                 default_ts="2026-06-01T00:00:00", link_to="p_u")
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.ASSERTION, {}))
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.UNKNOWN      # cells reachable via the anchor
