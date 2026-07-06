"""Extractor-agnostic canonicalization — one relation vocabulary for all extractors.

THE PROBLEM (v1.2 Cerebras diagnosis, HANDOFF 2026-07-03): every engine rule
keys on literal relation strings (graph.FUNCTIONAL_RELS, graph.PARENT_RELS,
taxonomy.CONTAINMENT_RELS, fallback's member_of/part_of). Different extractors
emit the same FACT with different verbs (gemma: "resides_in", gemini:
"lives_in") -> rules silently no-op -> the engine refuses answers it HAS.
Accuracy swung 30-48% purely by extractor choice.

THE FIX: a deterministic synonym -> canonical map over edges, plus attr-key
normalization over nodes. Runs FIRST in the post-ingest chain (before
auto_contradict / auto_supersede / taxonomy / enrich / episodes) so every
downstream rule sees one vocabulary. No LLM, idempotent, and unknown
relations pass through untouched — canonicalization can only ever ADD rule
coverage, never destroy signal.
"""
from __future__ import annotations

import re

from .graph import EvidenceGraph, NodeType

# Canonical targets are the exact strings engine rules key on:
#   functional (auto_supersede):  lives_in, works_at, married_to, dating
#   parent ladder (fallback):     member_of, part_of
#   containment (taxonomy):       located_in
#   recall/temporal endpoints:    attends, traveled_to
#   assertion linking:            about, needed_for
#   knowledge update:             contradicts

REL_SYNONYMS = {
    # -> lives_in (functional: newer edge supersedes older)
    "resides_in": "lives_in", "resides_at": "lives_in", "residing_in": "lives_in",
    "lived_in": "lives_in", "lives_at": "lives_in", "living_in": "lives_in",
    "moved_to": "lives_in", "moves_to": "lives_in", "moving_to": "lives_in",
    "relocated_to": "lives_in", "relocates_to": "lives_in", "home_in": "lives_in",
    # -> works_at (functional)
    "employed_by": "works_at", "employed_at": "works_at",
    "works_for": "works_at", "working_at": "works_at", "working_for": "works_at",
    "worked_at": "works_at", "worked_for": "works_at",
    "job_at": "works_at", "hired_by": "works_at", "hired_at": "works_at",
    "position_at": "works_at", "staff_of": "works_at",
    # -> married_to / dating (functional)
    "spouse_of": "married_to", "husband_of": "married_to",
    "wife_of": "married_to", "wed_to": "married_to",
    "in_relationship_with": "dating", "boyfriend_of": "dating",
    "girlfriend_of": "dating",
    # -> member_of (parent ladder)
    "belongs_to": "member_of", "affiliated_with": "member_of",
    "member": "member_of", "is_member_of": "member_of",
    # -> part_of (parent ladder)
    "component_of": "part_of", "included_in": "part_of",
    "subset_of": "part_of", "section_of": "part_of", "belongs_in": "part_of",
    # -> located_in (containment: taxonomy subtree/scoping)
    "is_in": "located_in", "within": "located_in", "inside": "located_in",
    "situated_in": "located_in", "situated_at": "located_in",
    "located_at": "located_in", "found_in": "located_in",
    "in_city": "located_in",
    # -> attends (event participation; span/order endpoint resolution)
    "attended": "attends", "attending": "attends", "attend": "attends",
    "attends_event": "attends",
    "participated_in": "attends", "participates_in": "attends",
    "participating_in": "attends", "took_part_in": "attends",
    "takes_part_in": "attends", "present_at": "attends",
    # -> traveled_to (recency defaults, travel history)
    "travelled_to": "traveled_to", "travels_to": "traveled_to",
    "traveling_to": "traveled_to", "travelling_to": "traveled_to",
    "trip_to": "traveled_to", "flew_to": "traveled_to",
    "flies_to": "traveled_to", "drove_to": "traveled_to",
    "journeyed_to": "traveled_to", "vacationed_in": "traveled_to",
    # -> about (assertion linking)
    "regarding": "about", "concerning": "about", "concerns": "about",
    "refers_to": "about", "references": "about", "describes": "about",
    "mentions": "about", "discusses": "about", "is_about": "about",
    "on_topic_of": "about",
    # -> needed_for (assertion linking)
    "required_for": "needed_for", "necessary_for": "needed_for",
    "needed_by": "needed_for", "required_by": "needed_for",
    "essential_for": "needed_for",
    # -> contradicts / precedes (update + ordering machinery)
    "conflicts_with": "contradicts", "contradicting": "contradicts",
    "happened_before": "precedes", "occurred_before": "precedes",
    "prior_to": "precedes", "preceded": "precedes", "before": "precedes",
    # -> scheduled_at (event timing)
    "scheduled_for": "scheduled_at", "occurs_on": "scheduled_at",
    "happens_on": "scheduled_at", "held_on": "scheduled_at",
    # -> is_a (taxonomy)
    "instance_of": "is_a", "kind_of": "is_a", "is_an": "is_a",
    "is_type_of": "is_a", "type_of": "is_a",
    # -> hosts / related_to (generic)
    "organizes": "hosts", "organized": "hosts", "hosting": "hosts",
    "relates_to": "related_to",
}


def _typed_based_in(src_t, dst_t):
    # a PERSON based somewhere lives there; an org/place is contained there
    return "lives_in" if src_t == NodeType.PERSON else "located_in"


def _typed_went_to(src_t, dst_t):
    # "went to" an EVENT is attendance; "went to" a PLACE is travel
    if dst_t == NodeType.EVENT:
        return "attends"
    if dst_t == NodeType.PLACE:
        return "traveled_to"
    return None  # target type unknown: never guess, leave the verb alone


def _typed_enrolled_in(src_t, dst_t):
    if dst_t == NodeType.EVENT:
        return "attends"
    if dst_t == NodeType.ORG:
        return "member_of"
    return None


def _typed_joined(src_t, dst_t):
    if dst_t == NodeType.EVENT:
        return "attends"
    if dst_t == NodeType.ORG:
        return "member_of"
    return None


def _typed_visited(src_t, dst_t):
    # visiting a place is travel; visiting a person/other stays as stated
    return "traveled_to" if dst_t == NodeType.PLACE else None


# relations whose canonical form depends on endpoint types
TYPED_RELS = {
    "based_in": _typed_based_in,
    "went_to": _typed_went_to,
    "goes_to": _typed_went_to,
    "going_to": _typed_went_to,
    "enrolled_in": _typed_enrolled_in,
    "enrolls_in": _typed_enrolled_in,
    "signed_up_for": _typed_enrolled_in,
    "joined": _typed_joined,
    "joins": _typed_joined,
    "visited": _typed_visited,
    "visits": _typed_visited,
    "visiting": _typed_visited,
}

# attr-key synonyms -> the keys engine rules read (scoring: date/date_window/
# participants; auto_contradict: value/metric; sum mode: unit). Values are
# COPIED under the canonical key only when it is absent — never overwritten.
ATTR_SYNONYMS = {
    "event_date": "date", "on_date": "date", "occurred_on": "date",
    "date_of_event": "date", "start_date": "date",
    "amount": "value", "price": "value", "cost": "value",
    "quantity": "value", "score": "value", "total": "value",
    "duration": "value",
    "measure": "metric", "metric_name": "metric", "metric_type": "metric",
    "units": "unit",
    "attendees": "participants",
}


def _norm_rel(rel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", rel.lower()).strip("_")


def canon_rel(rel: str, src_type: NodeType | None = None,
              dst_type: NodeType | None = None) -> str:
    """Map one relation verb to its canonical form. Deterministic; unknown
    verbs pass through unchanged (normalization aside)."""
    r = _norm_rel(rel)
    typed = TYPED_RELS.get(r)
    if typed:
        out = typed(src_type, dst_type)
        if out:
            return out
    return REL_SYNONYMS.get(r, r)


def canonicalize(graph: EvidenceGraph) -> dict:
    """Rewrite every edge relation and node attr key to canonical form,
    in place. Idempotent. Returns {'edges': n_rewritten, 'attrs': n_copied}."""
    edges = attrs = 0
    for _, _, data in graph.g.edges(data=True):
        e = data["obj"]
        src = graph.nodes.get(e.src)
        dst = graph.nodes.get(e.dst)
        new = canon_rel(e.rel, src.type if src else None,
                        dst.type if dst else None)
        if new != e.rel:
            e.rel = new
            data["rel"] = new
            edges += 1
    for n in graph.nodes.values():
        for syn, canon in ATTR_SYNONYMS.items():
            if syn in n.attrs and canon not in n.attrs:
                n.attrs[canon] = n.attrs[syn]
                attrs += 1
    return {"edges": edges, "attrs": attrs}
