"""Confidence calculus: noisy-OR over independent paths, constraint matching (SPEC.md §3.2, §6)."""
from __future__ import annotations

from .activation import Path
from .graph import Node

THETA_CONFIRM = 0.6
THETA_FLOOR = 0.2
TIE_EPSILON = 0.05

# soft-match factors
WINDOW_MATCH = 0.8      # constraint falls inside a candidate's declared window
PARTIAL_OVERLAP = 0.7   # partial set overlap (some participants match)
ATTR_UNKNOWN = 0.9      # candidate silent on this constraint — mild penalty, not a fail


def independent_paths(paths: list[Path]) -> list[Path]:
    """Greedy independence filter: accept best-first, reject any path sharing an
    intermediate node with an already-accepted path (SPEC.md §3.2 v0.1 rule)."""
    accepted: list[Path] = []
    used: set[str] = set()
    for p in sorted(paths, key=lambda p: -p.score):
        inter = p.intermediates
        if inter & used:
            continue
        accepted.append(p)
        used |= inter
    return accepted


def noisy_or(paths: list[Path]) -> float:
    prod = 1.0
    for p in independent_paths(paths):
        prod *= (1.0 - p.score)
    return 1.0 - prod


def match_constraints(node: Node, constraints: dict, strict: bool = False,
                      required: frozenset = frozenset()) -> float:
    """Score a candidate node against placeholder constraints. 0.0 = hard mismatch.

    strict=True is the DIRECT-LOOKUP standard: every constraint must match exactly.
    strict=False is the candidate-scoring standard: soft factors apply.
    `required` keys are DEFINING constraints — they name what the unknown IS
    (e.g. word="bank" in lexical disambiguation). A candidate silent on a
    defining key is not a weaker candidate, it is not a candidate at all.
    """
    score = 1.0
    for key, want in constraints.items():
        if key in required and key not in node.attrs:
            return 0.0
        if key == "date_range":
            # want = (from_iso, to_iso); a dated candidate inside the window is a match
            lo, hi = want
            got = node.attrs.get("date")
            window = node.attrs.get("date_window")
            if got and lo <= got <= hi:
                continue
            if window and not (window[1] < lo or window[0] > hi):
                if strict:
                    return 0.0
                score *= WINDOW_MATCH
                continue
            if not strict and got is None and window is None:
                score *= ATTR_UNKNOWN
                continue
            return 0.0
        if key == "date":
            got = node.attrs.get("date")
            window = node.attrs.get("date_window")
            if got == want:
                continue
            if not strict and window and window[0] <= want <= window[1]:
                score *= WINDOW_MATCH
                continue
            if not strict and got is None and window is None:
                score *= ATTR_UNKNOWN
                continue
            return 0.0
        elif key == "participants":
            have = set(node.attrs.get("participants", []))
            want_set = set(want)
            if want_set <= have:
                continue
            if strict:
                return 0.0
            if have & want_set:
                score *= PARTIAL_OVERLAP
            elif not have:
                score *= ATTR_UNKNOWN
            else:
                return 0.0
        else:
            got = node.attrs.get(key)
            if got == want:
                continue
            if not strict and got is None:
                # v1.6 (dev2 gpt4_5dcc0aab: gold 'User cleaned white Adidas
                # sneakers last month' ranked ~25th under noise): a string
                # constraint satisfied by the node's own LABEL text is a
                # match, not a missing attr — extractors put discriminators
                # in labels far more reliably than in attr keys
                if isinstance(want, str) and _label_satisfies(node, want):
                    continue
                # type-echo keys duplicate the node-type system; penalizing
                # their absence double-taxes every candidate (v0.9 flip
                # 1e043500: a 'type: playlist' echo pushed the right answer
                # under the premise-refusal bar)
                if key not in ("type", "kind", "category"):
                    score *= ATTR_UNKNOWN
                continue
            return 0.0
    return score


def unevidenced_keys(node: Node, constraints: dict) -> list[str]:
    """Constraint keys the candidate is SILENT on — no attr, no window, no
    label mention. Distinct from soft matches (a date inside a declared
    window IS evidence): v1.14, CONFIRMED requires every asked condition
    at least evidenced, while soft matches merely discount the score."""
    out = []
    for key, want in constraints.items():
        if key in ("type", "kind", "category"):
            continue                       # type-echo keys, never evidential
        if key == "participants":
            continue    # evidenced by multi-anchor path convergence (S1)
        if key in node.attrs:
            continue
        if key in ("date", "date_range") and node.attrs.get("date_window"):
            continue
        if isinstance(want, str) and _label_satisfies(node, want):
            continue
        if isinstance(want, (list, tuple)) and any(
                isinstance(w, str) and _label_satisfies(node, w) for w in want):
            continue
        out.append(key)
    return out


def _label_satisfies(node: Node, want: str) -> bool:
    """Every content token of the constraint value appears (as a word prefix,
    plural-insensitive) in the node's label/attr text."""
    tokens = [t.rstrip("s") for t in want.lower().split() if len(t) > 2]
    if not tokens:
        return False
    text = (node.label + " " + " ".join(
        str(v) for k, v in node.attrs.items() if k != "aliases")).lower()
    words = [w.rstrip("s") for w in text.replace(",", " ").split()]
    return all(any(w.startswith(t) for w in words) for t in tokens)
