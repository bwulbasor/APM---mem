"""Deterministic markdown-table ingestion.

Tables defeat LLM extraction reliably (dense, positional, token-hungry) — but
they are STRUCTURED data; parsing them with an LLM was always the wrong tool.
Each cell becomes an assertion node "row-key · column-header: value" linked to
a table topic node and to any existing entity the cell value resolves to, so
activation from e.g. a person node reaches their cells directly.
"""
from __future__ import annotations

import re

from .entities import resolve
from .graph import Edge, EvidenceGraph, Node, NodeType

_TABLE_RE = re.compile(
    r"(?:^|\n)(\|[^\n]+\|\n\|[\s:|-]+\|\n(?:\|[^\n]+\|(?:\n|$))+)", re.MULTILINE)


def find_tables(text: str) -> list[str]:
    return [m.group(1).strip() for m in _TABLE_RE.finditer(text)]


def strip_tables(text: str) -> str:
    return _TABLE_RE.sub("\n[table extracted separately]\n", text)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_table(table_text: str) -> tuple[list[str], list[list[str]]]:
    lines = [l for l in table_text.splitlines() if l.strip()]
    header = _cells(lines[0])
    rows = [_cells(l) for l in lines[2:]]  # line 1 is the |---| separator
    return header, rows


def ingest_table(graph: EvidenceGraph, table_text: str, context_label: str,
                 default_ts: str | None = None, source: str = "table",
                 link_to: str | None = None) -> int:
    """Every cell -> assertion node, linked to a table topic node and to any
    resolvable entities in the cell / row key / column header. Deterministic.

    `link_to`: context anchor (e.g. the User node). Without it, a table whose
    cells resolve to nothing floats DISCONNECTED and activation can never
    reach it — a table someone shared is always connected to them."""
    header, rows = parse_table(table_text)
    tid = f"to_table_{abs(hash(table_text)) % 10**8}"
    graph.add_node(Node(tid, NodeType.TOPIC, f"Table: {context_label}",
                        timestamp=default_ts, source=source))
    if link_to and link_to in graph.nodes:
        graph.add_edge(Edge(tid, link_to, "shared_with", 0.8,
                            timestamp=default_ts))
    added = 0
    for row in rows:
        if not row:
            continue
        row_key = row[0] or "(row)"
        for j, cell in enumerate(row[1:], start=1):
            if not cell:
                continue
            col = header[j] if j < len(header) and header[j] else f"col{j}"
            label = f"{row_key} · {col}: {cell}"
            nid = f"as_cell_{abs(hash((tid, row_key, col, cell))) % 10**10}"
            if nid in graph.nodes:
                continue
            graph.add_node(Node(nid, NodeType.ASSERTION, label,
                                attrs={"row": row_key, "column": col,
                                       "value": cell},
                                timestamp=default_ts, source=source))
            graph.add_edge(Edge(nid, tid, "part_of", 0.9, timestamp=default_ts))
            # link the cell to every entity it mentions that we know of
            for mention in (cell, row_key, col):
                ent = resolve(graph, mention, None)
                if ent and ent not in (nid, tid):
                    graph.add_edge(Edge(nid, ent, "mentions", 0.85,
                                        timestamp=default_ts))
            added += 1
    return added
