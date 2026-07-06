"""LLM tie-arbiter — SPEC §3.5 allowed role 2 (constraint projection filter).

When graph evidence leaves several candidates TIED, the embedding tiebreak
picks by label cosine — which rewards question-word overlap and loses to
theme echoes (dev2: Converse over the white Adidas; Gatorland over the Sugar
Factory). An LLM sees what cosine can't: which candidate, GIVEN ITS GRAPH
CONTEXT, actually answers the question.

Safety contract (why this does not break the gate):
  - The candidate list comes from the ENGINE; the LLM cannot add to it.
  - The verdict class is fixed (ASSUMED) before the arbiter runs — a wrong
    pick costs which-guess, never confidence.
  - Any failure (bad JSON, label not in the closed list, API down) returns
    None and the caller falls back to the embedding tiebreak.
Disable for ablations with APM_NO_ARBITER=1.
"""
from __future__ import annotations

import json
import os

from .llm import chat

ARBITER_SYSTEM = """You pick which candidate best answers a question, using each
candidate's evidence context. Output ONLY JSON: {"pick": "<exact candidate label>"}.

Rules:
- You MUST pick from the given candidates; never invent an answer.
- Prefer the candidate that IS the thing asked for. A candidate that merely
  restates the question's topic is NOT the answer.
- Use the evidence lines (each candidate's graph connections) to judge which
  one the question is about.
- If genuinely none fits better than the others, pick the first."""


def arbitrate(question: str, candidates: list[tuple[str, list[str]]]) -> int | None:
    """Index of the best candidate, or None (caller falls back).
    `candidates` = [(label, [evidence lines]), ...] — a CLOSED list."""
    if os.environ.get("APM_NO_ARBITER") or len(candidates) < 2:
        return None
    lines = []
    for label, ev in candidates:
        lines.append(f"- {label}")
        for e in ev[:5]:
            lines.append(f"    evidence: {e}")
    prompt = f"Question: {question}\n\nCandidates:\n" + "\n".join(lines)
    try:
        raw = chat(prompt, system=ARBITER_SYSTEM, temperature=0.0,
                   role="veto", json_mode=True)
        pick = str(json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["pick"])
    except Exception:
        return None
    labels = [label for label, _ in candidates]
    if pick in labels:
        return labels.index(pick)
    # tolerate minor echo damage: unique case-insensitive prefix match
    low = pick.lower().strip()
    hits = [i for i, l in enumerate(labels)
            if l.lower().startswith(low) or low.startswith(l.lower())]
    return hits[0] if len(hits) == 1 else None
