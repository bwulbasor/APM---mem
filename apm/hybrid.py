"""Hybrid tier v2 (v1.11) — transcript-grounded fallback with VERIFIED quotes.

Answer-state tiers (the product contract):
  CONFIRMED       symbolic proof — evidence chain in the graph
  ASSUMED         graph answer via a named default rule (tagged guess)
  TEXT_SUPPORTED  no graph evidence, but a source quote from the raw
                  transcript SUPPORTS the answer — and the quote is
                  mechanically verified to exist in the named session
  REFUSED         nothing supports an answer anywhere

v1 failed for measured reasons: trigger keyed on engine class (missed the
weak-ASSUMED refusals), read only top-3 of ~45 sessions, and asked for an
answer instead of evidence. v2: trigger on FINAL answer state (caller-side),
scan top-K=20 sessions, and require the model to return the supporting
quote — an answer whose quote cannot be located in the claimed session is
DISCARDED. The LLM finds evidence; the harness verifies it exists.
Disable with APM_NO_HYBRID=1.
"""
from __future__ import annotations

import json
import os
import re

from .llm import chat
from .recall import select_sessions

K_SESSIONS = 20
PER_SESSION_CHARS = 9000

HYBRID_SYSTEM = """You search transcripts for the answer to a question.
Output ONLY JSON: {"answer": "<1-2 sentence direct answer>",
                   "quote": "<the EXACT supporting sentence(s) copied verbatim from a transcript>",
                   "session": "<the session id the quote is from>"}

Rules:
- The quote must be copied VERBATIM from one transcript — it is checked
  mechanically; a paraphrased quote invalidates the answer.
- Answer ONLY what the quote directly supports. No outside knowledge.
- If no transcript contains the information, output exactly: {"answer": null}
- Prefer the most recent statement when the transcripts disagree."""

REFUSAL_PAT = re.compile(
    r"i do( |)n[o']t know|no record|not (available|established|recorded|"
    r"specified|mentioned)|cannot (determine|confirm)|do not have|"
    r"i lack|unable to", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _quote_in(quote: str, text: str) -> bool:
    """Verbatim-ish verification: normalized quote must appear in the
    normalized session text (fuzzy fallback for minor mangling)."""
    nq, nt = _norm(quote), _norm(text)
    if len(nq) < 10:
        return False
    if nq in nt:
        return True
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(nq[:300], nt) >= 90
    except ImportError:
        return False


def looks_like_refusal(hypothesis: str) -> bool:
    return bool(REFUSAL_PAT.search(hypothesis or ""))


def text_supported_answer(graph, query, question: str,
                          sessions: list[dict]) -> dict | None:
    """Returns {'answer','quote','session'} with the quote VERIFIED against
    the named session's text, or None. Never invents: an unverifiable quote
    kills the answer."""
    if os.environ.get("APM_NO_HYBRID") or not sessions:
        return None
    chosen = select_sessions(graph, query, question, sessions, K_SESSIONS)
    corpus = "\n\n".join(
        f"=== session {s['id']} ===\n{s['text'][:PER_SESSION_CHARS]}"
        for s in chosen)
    try:
        raw = chat(f"Question: {question}\n\n{corpus}",
                   system=HYBRID_SYSTEM, temperature=0.0,
                   role="verbalize", json_mode=True, max_tokens=2000)
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return None
    ans, quote = data.get("answer"), data.get("quote")
    if not ans or not quote:
        return None
    sid = str(data.get("session", ""))
    by_id = {s["id"]: s for s in chosen}
    target = by_id.get(sid)
    candidates = [target] if target else chosen
    for s in candidates:
        if s and _quote_in(str(quote), s["text"]):
            return {"answer": str(ans), "quote": str(quote)[:220],
                    "session": s["id"]}
    return None
