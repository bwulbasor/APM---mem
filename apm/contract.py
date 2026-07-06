"""The representation contract — ONE canonical vocabulary that the parser,
the graph node fields, and structural enrichment must all speak, so the engine
compares like with like.

The delta experiment showed proof chains collapse not because the reasoner or
extractor is wrong, but because the three speak different dialects:
    parser 'type'      vs node 'event_type'
    parser 'July 3rd'  vs node '2026-07-03'
    parser 'agenda'    vs node 'topic'
    parser 'people'    vs node 'participants'
This module canonicalizes KEYS and DATE VALUES on whichever side calls it.

FAITHFULNESS: this only RENAMES/REFORMATS existing fields — it never adds a
claim. Safe for CONFIRMED. Date resolution is anchored to a known session
date, never guessed blind.
"""
from __future__ import annotations

import re
from datetime import datetime

# canonical key <- accepted synonyms
_KEY_CANON = {
    "kind": "kind", "type": "kind", "event_type": "kind", "category": "kind",
    "topic": "topic", "agenda": "topic", "subject": "topic", "about": "topic",
    "participants": "participants", "people": "participants",
    "attendees": "participants", "who": "participants",
    "organization": "organization", "org": "organization", "company": "organization",
    "location": "location", "place": "location", "where": "location",
    "date": "date", "when": "date", "day": "date",
    "date_range": "date_range", "date_window": "date_range",
}

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12}
_MONTHDAY = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b", re.I)
_THE_NTH = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_IN_MONTH = re.compile(r"\b(?:in|back in|last|this past|during)\s+(" + "|".join(_MONTHS) + r")\b", re.I)
_ISO_DAY = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b")


def _last_day(y, m):
    return {1: 31, 2: 29 if y % 4 == 0 and (y % 100 or not y % 400) else 28,
            3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
            11: 30, 12: 31}[m]


def resolve_date_phrase(text, anchor: datetime | None):
    """A date phrase (raw parser string OR node label/attr) -> canonical
    ('date', 'YYYY-MM-DD') for an exact day, or ('date_range', (start, end))
    for a month/window. anchor = the session datetime that fixes the year.
    Returns None if no date is expressed. Shared by both sides so 'July 3rd',
    '2026-07-03' and '2026-07' all canonicalize to the same comparable form."""
    if not text:
        return None
    s = str(text)
    m = _ISO_DAY.search(s)
    if m:
        return ("date", f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    m = _ISO_MONTH.search(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return ("date_range", (datetime(y, mo, 1).date().isoformat(),
                               datetime(y, mo, _last_day(y, mo)).date().isoformat()))
    a = anchor if (anchor and anchor != datetime.min) else None
    m = _MONTHDAY.search(s)
    if m:
        mo, day = _MONTHS[m.group(1).lower()], int(m.group(2))
        yr = (a.year if a and mo >= a.month - 1 else (a.year + 1 if a else 2026))
        try:
            return ("date", datetime(yr, mo, day).date().isoformat())
        except ValueError:
            return None
    m = _IN_MONTH.search(s)
    if m:
        mo = _MONTHS[m.group(1).lower()]
        yr = a.year if (a and mo <= a.month) else (a.year - 1 if a else 2026)
        return ("date_range", (datetime(yr, mo, 1).date().isoformat(),
                               datetime(yr, mo, _last_day(yr, mo)).date().isoformat()))
    if a:
        m = _THE_NTH.search(s)
        if m:
            day = int(m.group(1))
            try:
                return ("date", datetime(a.year, a.month, day).date().isoformat())
            except ValueError:
                return None
    return None


def canon_constraints(constraints: dict, anchor: datetime | None) -> dict:
    """Rewrite a parsed query's constraint dict into the canonical schema:
    canonical keys + normalized date values. Non-date values pass through."""
    out = {}
    for k, v in constraints.items():
        ck = _KEY_CANON.get(k.lower(), k.lower())
        if ck in ("date", "date_range"):
            r = resolve_date_phrase(v if isinstance(v, str) else str(v), anchor)
            if r:
                out[r[0]] = r[1]
                continue
        out[ck] = v
    return out


def canon_key(k: str) -> str:
    return _KEY_CANON.get(k.lower(), k.lower())
