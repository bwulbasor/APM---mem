"""Cued recall (v1.9) — question-conditioned second-chance extraction.

THE PLATEAU IT ATTACKS (measured 2026-07-05, n=42: 35.7% acc / 57% refusal):
~half of all misses are facts STATED in the source conversation that
ingest-time extraction dropped, because extraction runs before anyone knows
what will be asked — salience is unknowable at ingest. Every extractor from
31B to frontier drops the SAME low-salience asides.

THE FIX (cognitive analog: cued recall — a retrieval cue triggers re-search
of the episode; memory is not fully indexed upfront): keep the raw session
transcripts; when the engine's first pass FAILS (UNKNOWN or a weak guess),
select the most relevant sessions via the activation trace + lexical
overlap, re-extract ONLY those transcripts with the question as the
extraction focus, merge into a query OVERLAY graph (never the base), and
re-run the deterministic engine.

THE GATE IS UNTOUCHED: the recall LLM produces candidate evidence, never
answers; explicit facts may CONFIRM (they cite the transcript), implied
edges stay ASSUMED-capped (ingest's implied machinery); a failed re-read
just leaves the refusal standing. Wrong cued evidence costs a hedged guess,
never an invention.
"""
from __future__ import annotations

import re

from .activation import spread
from .engine import Engine, Query
from .gate import Verdict, VerdictClass
from .graph import EvidenceGraph
from .scoring import THETA_CONFIRM

# how many sessions to re-read
K_NORMAL = 2
K_AGGREGATE = 6

CUED_HINT = (
    "You are NOT answering the question. A memory search for this question "
    "FAILED; re-read the transcript and extract every atomic fact that could "
    "help answer it — especially low-salience details, asides, delivery "
    "channels, locations, dates, quantities, names, and causal links that a "
    "first read would skim past. Do not omit facts because they seem minor. "
    "Facts only stated indirectly go in as edges with \"implied\": true. "
    "Do not invent unstated facts.\n"
    "The question:\n  {question}\n"
    "Also extract the normal full picture of the transcript, but "
    "PRIORITIZE completeness on anything related to the question.")

AGG_EXTRA = (" This is a COUNT/TOTAL question: extract EVERY matching item "
             "or quantity as its own node — an incomplete enumeration makes "
             "the answer wrong. Do not summarize items away.")


def needs_recall(verdict: Verdict) -> bool:
    """Fire on anything short of CONFIRMED: every ASSUMED is by definition a
    guess (tiebreaks, defaults, caps), and a re-read can only upgrade it —
    adoption never downgrades. CONFIRMED answers are left alone."""
    return verdict.klass in (VerdictClass.UNKNOWN, VerdictClass.ASSUMED)


def select_sessions(graph: EvidenceGraph, query: Query, question: str,
                    sessions: list[dict], k: int) -> list[dict]:
    """Rank sessions by (a) activation mass of their extracted nodes given
    the query anchors — the graph already knows roughly where the topic
    lives — plus (b) lexical overlap with the question as a backup signal.
    `sessions` = [{"id": str, "text": str}]; node.source encodes the id."""
    scores = {s["id"]: 0.0 for s in sessions}
    anchors = [a for a in query.anchors if a in graph.nodes]
    if anchors:
        activated = spread(graph, anchors)
        for nid, paths in activated.items():
            src = graph.node(nid).source
            for s in sessions:
                if s["id"] in src:
                    scores[s["id"]] += max(p.score for p in paths)
    q_tokens = {t.rstrip("s") for t in re.findall(r"[a-z0-9]+", question.lower())
                if len(t) > 3}
    for s in sessions:
        text = s["text"].lower()
        overlap = sum(1 for t in q_tokens if t in text)
        scores[s["id"]] += 0.3 * overlap
    ranked = sorted(sessions, key=lambda s: -scores[s["id"]])
    return ranked[:k]


def cued_recall(graph: EvidenceGraph, engine_query: Query, question: str,
                sessions: list[dict], now=None,
                parse_fn=None) -> tuple[Verdict | None, dict]:
    """Second-chance pass. Returns (new_verdict | None, meta). The caller
    decides adoption; base graph is never touched (overlay copy)."""
    from .ingest import ingest
    aggregate = engine_query.mode in ("count", "sum") or bool(
        re.search(r"\bhow many\b|\btotal\b|\ball\b", question.lower()))
    k = K_AGGREGATE if aggregate else K_NORMAL
    chosen = select_sessions(graph, engine_query, question, sessions, k)
    if not chosen:
        return None, {"triggered": True, "sessions": 0}

    overlay = graph.copy()
    hint = CUED_HINT.format(question=question) + (AGG_EXTRA if aggregate else "")
    added = {"nodes": 0, "edges": 0}
    for s in chosen:
        # full, untruncated text — ingest-time truncation is part of why
        # the fact was lost; chunk on line boundaries to fit context
        text, buf, chunks, n = s["text"], [], [], 0
        for line in text.splitlines():
            if n + len(line) > 5000 and buf:
                chunks.append("\n".join(buf)); buf, n = [], 0
            buf.append(line); n += len(line) + 1
        if buf:
            chunks.append("\n".join(buf))
        for chunk in chunks:
            try:
                st = ingest(overlay, chunk, source=f"cued:{s['id']}",
                            hint=hint, default_ts=s.get("ts"))
                added = {k_: added[k_] + st[k_] for k_ in added}
            except Exception:
                continue
    meta = {"triggered": True, "sessions": len(chosen), **added}
    if not added["nodes"] and not added["edges"]:
        return None, meta

    from .canon import canonicalize
    canonicalize(overlay)
    overlay.auto_contradict()
    overlay.auto_supersede()
    # re-parse against the overlay when a parser is supplied: previously
    # unresolvable anchors/options may now exist
    q2 = engine_query
    if parse_fn is not None:
        try:
            q2 = parse_fn(overlay, question)
        except Exception:
            q2 = engine_query
    verdict = Engine(overlay).answer(q2, now=now)
    verdict.chain.add("note", f"cued recall: re-read {len(chosen)} session(s) "
                              f"for this question (+{added['nodes']} nodes, "
                              f"+{added['edges']} edges in a query overlay)")
    return verdict, meta


def adopt(first: Verdict, second: Verdict | None) -> tuple[Verdict, bool]:
    """v1.10 ADOPTION POLICY (tightened after the 84-question ablation
    measured same-class displacement as the dominant loss mode — cued
    evidence swapped correct hedged answers for wrong ones): a cued verdict
    is adopted ONLY when the first pass had NO answer at all. Refusal ->
    answer is pure upside; everything else keeps the first verdict."""
    if second is None or second.answer is None:
        return first, False
    if first.klass == VerdictClass.UNKNOWN \
            and second.klass != VerdictClass.UNKNOWN:
        return second, True
    if first.klass == VerdictClass.ASSUMED \
            and second.klass == VerdictClass.CONFIRMED:
        # cued evidence itself can never CONFIRM (gate cap) — a CONFIRMED
        # second verdict means BASE-graph evidence became reachable (e.g.
        # the re-parse resolved an option) — a legitimate upgrade
        return second, True
    return first, False
