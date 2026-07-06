"""Aggregation route (v1.12) — counts/sums/lists bypass the graph scorer.

Spreading activation is the wrong tool for aggregation: one missed or
spurious member breaks the answer (measured: multi-session 0/6 in every
configuration). This route treats aggregation as EVIDENCE-TABLE BUILDING:

  1. sweep the top-K sessions broadly
  2. LLM lists every matching item WITH a verbatim quote + session id
  3. every quote is mechanically verified against its session (rows that
     fail verification are DROPPED and completeness degrades)
  4. dedupe by normalized item name across sessions
  5. arithmetic over the surviving table — count rows or sum quantities

Output is TEXT_SUPPORTED-tier (quoted evidence, not graph proof); when the
model admits incompleteness or rows were dropped, the answer is delivered
as "at least N" (PARTIAL) rather than a false exact. Disable with
APM_NO_AGG=1.
"""
from __future__ import annotations

import json
import os
import re

from .hybrid import _norm, _quote_in
from .llm import chat
from .recall import select_sessions

K_SESSIONS = 20
PER_SESSION_CHARS = 9000

AGG_SYSTEM = """You build a COMPLETE evidence table for a counting/total question.
Scan ALL transcripts; missing even one matching item makes the answer wrong.
Output ONLY JSON:
{"items":[{"item":"<short distinct name>","quantity":<number>,
           "quote":"<verbatim supporting sentence from the transcript>",
           "session":"<session id>"}],
 "complete": <true only if you are confident EVERY matching item is listed>}

Rules:
- One row per DISTINCT matching item or occurrence. Never merge or summarize.
- quantity: the stated number for that row (days, hours, dollars, times);
  use 1 for a countable object/event without a stated number.
- quote is copied VERBATIM — it is checked mechanically against the
  transcript; a paraphrase invalidates the row.
- Respect the question's criteria strictly (time window, category, place).
- If nothing matches: {"items":[],"complete":true}."""


def is_aggregation(query, question: str) -> bool:
    """Only genuine multi-item questions (v1.12.1: a broad 'how many' regex
    stole single-stated-value questions — '25 titles on my list' — from the
    single-quote tier and answered with a 1-row table). The parser already
    routes real counting/totaling to these modes."""
    return query.mode in ("count", "sum")


def aggregate_answer(graph, query, question: str,
                     sessions: list[dict]) -> dict | None:
    """Returns {'value', 'unit', 'partial', 'items': [...]} or None."""
    if os.environ.get("APM_NO_AGG") or not sessions:
        return None
    chosen = select_sessions(graph, query, question, sessions, K_SESSIONS)
    corpus = "\n\n".join(
        f"=== session {s['id']} ===\n{s['text'][:PER_SESSION_CHARS]}"
        for s in chosen)
    try:
        raw = chat(f"Question: {question}\n\n{corpus}",
                   system=AGG_SYSTEM, temperature=0.0,
                   role="verbalize", json_mode=True, max_tokens=3000)
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return None
    items = data.get("items")
    if not isinstance(items, list):
        return None
    by_id = {s["id"]: s for s in chosen}
    verified, dropped = [], 0
    for it in items:
        quote = str(it.get("quote", ""))
        sess = by_id.get(str(it.get("session", "")))
        pool = [sess] if sess else chosen
        if any(s and _quote_in(quote, s["text"]) for s in pool):
            verified.append(it)
        else:
            dropped += 1
    if not verified:
        return None
    # dedupe by normalized item name (same event retold across sessions)
    seen: dict[str, dict] = {}
    for it in verified:
        key = _norm(str(it.get("item", "")))[:60]
        if key not in seen:
            seen[key] = it
    rows = list(seen.values())
    if len(rows) < 2:
        # a table that DEDUPES to one row wasn't an aggregation — the answer
        # is likely a single stated value ("five sessions"); let the
        # single-quote tier read it instead of counting rows (v1.12.2)
        return None

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 1.0
    if query.mode == "sum" or re.search(r"\bhow much\b|\btotal\b",
                                        question.lower()):
        value = sum(num(it.get("quantity", 1)) for it in rows)
    else:
        value = len(rows)
    partial = bool(dropped) or not data.get("complete", False)
    return {"value": value, "partial": partial, "dropped": dropped,
            "items": [{"item": it.get("item"),
                       "quantity": num(it.get("quantity", 1)),
                       "session": it.get("session")} for it in rows]}
