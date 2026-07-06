"""Embedding support: semantic similarity for reranking weak candidates.

Safety contract: embeddings may REORDER guesses (which ASSUMED candidate to
present), never upgrade a verdict class — the class is fixed by graph evidence
before reranking runs. A bad embedding costs which-guess, not confidence.

Cached in the same sqlite as chat calls; failures degrade to None and callers
fall back to recency.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request

from .llm import CACHE_PATH, _cache, endpoint_keys_for
import os

EMBED_MODEL = os.environ.get("APM_LLM_MODEL_EMBED", "text-embedding-004")


def embed(text: str) -> list[float] | None:
    text = text[:2000]
    conn = _cache()
    key = "emb:" + hashlib.sha256((EMBED_MODEL + text).encode()).hexdigest()
    if conn is not None:
        row = conn.execute("SELECT response FROM llm_cache WHERE key=?",
                           (key,)).fetchone()
        if row:
            return json.loads(row[0])
    base, pool = endpoint_keys_for("embed")
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    vec = None
    for api_key in pool:
        req = urllib.request.Request(
            f"{base}/embeddings", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     "User-Agent": "apm/0.1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            vec = data["data"][0]["embedding"]
            break
        except Exception:
            continue
    if vec is None:
        return None
    if conn is not None:
        conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)",
                     (key, json.dumps(vec)))
        conn.commit()
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


_STOP = {"what", "which", "where", "when", "name", "did", "the", "that",
         "your", "from", "with", "have", "this", "does", "about"}


def semantic_pick(question: str, labels: list[str]) -> int | None:
    """Index of the label most semantically relevant to the question, or None
    if embeddings are unavailable. Selection only — never scores.

    Echo penalty: answers are NEW information — a label that parrots the
    question's own words ("Coupon Binder" for a coupon question) is usually
    the topic restated, not the answer. Penalize shared content tokens."""
    qv = embed(question)
    if qv is None:
        return None
    q_tokens = {t for t in question.lower().replace("?", " ").split()
                if len(t) > 3 and t not in _STOP}
    best_i, best_s = None, -1.0
    for i, label in enumerate(labels):
        lv = embed(label)
        if lv is None:
            continue
        overlap = sum(1 for t in label.lower().split()
                      if t in q_tokens)
        s = cosine(qv, lv) - min(0.2, 0.07 * overlap)
        if s > best_s:
            best_i, best_s = i, s
    return best_i
