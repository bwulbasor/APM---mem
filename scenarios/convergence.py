"""Convergence benchmark — HAND-BUILT graphs that test the engine's associative
pattern-matching in ISOLATION from extraction. LongMemEval/law-firm conflate
"is the reasoning good?" with "is the extractor good?"; these graphs are built
directly, so a low score here is a reasoning failure, not an extraction one.

Each scenario is the shape APM was designed for: several INDEPENDENT weak cues
that should CONVERGE on one node, with NEAR-MISS distractors that share a SUBSET
of the cues (same org, wrong type / same date, wrong people / same category, no
agenda). The research claim under test: many weak cues beat one strong cue, and
a near-miss must lose to the true convergent answer.

Declarative: each scenario lists nodes, edges, a query, the gold node id, and
the tier we expect. build() returns (graph, query). Run via bench/run_convergence.py.
"""
from __future__ import annotations

from apm import Edge, EvidenceGraph, Node, NodeType, Placeholder, Query

NOW = "2026-06-22T10:00:00"
_TYPE = {"person": NodeType.PERSON, "org": NodeType.ORG, "event": NodeType.EVENT,
         "place": NodeType.PLACE, "time": NodeType.TIME, "topic": NodeType.TOPIC,
         "assertion": NodeType.ASSERTION}


def build(scen: dict):
    g = EvidenceGraph()
    for n in scen["nodes"]:
        nid, typ, label = n[0], n[1], n[2]
        attrs = n[3] if len(n) > 3 else {}
        ts = n[4] if len(n) > 4 else None
        g.add_node(Node(nid, _TYPE[typ], label, attrs=dict(attrs), timestamp=ts))
    for e in scen["edges"]:
        src, dst, rel, w = e[0], e[1], e[2], e[3]
        ts = e[4] if len(e) > 4 else None
        g.add_edge(Edge(src, dst, rel, w, timestamp=ts))
    qs = scen["query"]
    q = Query(anchors=qs["anchors"],
              placeholder=Placeholder(_TYPE[qs["type"]], qs.get("constraints", {}),
                                      required=qs.get("required", [])),
              mode=qs.get("mode", "resolve"),
              options=qs.get("options", []),
              topic_text=qs.get("topic_text"),
              predictive=qs.get("predictive", False))
    return g, q


# ---------------------------------------------------------------------------
SCENARIOS = [
    # 1. Canonical convergence: date + org + 3 people + agenda -> GA.
    #    Near-miss "summer party" shares org + a date window but is a PARTY,
    #    and a calendar clutter event shares the date only.
    {
        "id": "conv_ga_multi_cue",
        "desc": "4 cues converge on the General Assembly; party shares org, clutter shares date",
        "nodes": [
            ("p_anna", "person", "Anna"), ("p_marco", "person", "Marco"),
            ("p_priya", "person", "Priya"), ("org_iaeste", "org", "IAESTE"),
            ("ev_ga", "event", "IAESTE General Assembly",
             {"date": "2026-06-26", "event_type": "meeting",
              "participants": ["p_anna", "p_marco", "p_priya"]}, "2026-06-26T18:00:00"),
            ("ev_party", "event", "IAESTE summer party",
             {"date": "2026-06-26", "event_type": "party"}, "2026-06-26T20:00:00"),
            ("ev_dentist", "event", "Dentist appointment",
             {"date": "2026-06-26", "event_type": "appointment"}, "2026-06-26T09:00:00"),
        ],
        "edges": [
            ("p_anna", "org_iaeste", "member_of", 0.9),
            ("p_marco", "org_iaeste", "member_of", 0.9),
            ("p_priya", "org_iaeste", "member_of", 0.9),
            ("org_iaeste", "ev_ga", "plans", 0.85),
            ("org_iaeste", "ev_party", "plans", 0.85),
            ("p_anna", "ev_ga", "attends", 0.8),
            ("p_marco", "ev_ga", "attends", 0.8),
            ("p_priya", "ev_ga", "attends", 0.8),
        ],
        "query": {"anchors": ["p_anna", "p_marco", "p_priya"], "type": "event",
                  "constraints": {"date": "2026-06-26", "event_type": "meeting",
                                  "participants": ["p_anna", "p_marco", "p_priya"]}},
        "gold": "ev_ga", "expect": "CONFIRMED",
    },
    # 2. Many weak cues beat one strong cue. Target reached from 3 anchors at
    #    weight ~0.6 each; distractor reached from ONE anchor at 0.95.
    {
        "id": "conv_weak_cues_beat_strong",
        "desc": "3 weak independent cues converge; 1 strong single cue should lose",
        "nodes": [
            ("c1", "topic", "Cue: rainy day"), ("c2", "topic", "Cue: jazz playing"),
            ("c3", "topic", "Cue: old bookshop"), ("c4", "topic", "Cue: strong-but-lonely"),
            ("m_target", "topic", "The remembered afternoon", {"kind": "memory"}),
            ("m_distract", "topic", "A vivid but unrelated day", {"kind": "memory"}),
        ],
        "edges": [
            ("c1", "m_target", "evokes", 0.6), ("c2", "m_target", "evokes", 0.6),
            ("c3", "m_target", "evokes", 0.6),
            ("c4", "m_distract", "evokes", 0.95),
        ],
        "query": {"anchors": ["c1", "c2", "c3", "c4"], "type": "topic",
                  "constraints": {"kind": "memory"}},
        "gold": "m_target", "expect": "CONFIRMED",
    },
    # 3. NEAR-MISS TRAP: two events, same org + same date, differ only by type.
    #    Question asks for the MEETING. Distractor is a party. This is the
    #    density-hurts-ranking failure the near-miss penalty must fix.
    {
        "id": "conv_nearmiss_wrong_type",
        "desc": "same org + same date; only event_type disambiguates meeting vs party",
        "nodes": [
            ("org_c", "org", "Chorus society"),
            ("ev_rehearsal", "event", "Board meeting",
             {"date": "2026-07-03", "event_type": "meeting"}, "2026-07-03T18:00:00"),
            ("ev_concert", "event", "Summer concert",
             {"date": "2026-07-03", "event_type": "concert"}, "2026-07-03T20:00:00"),
        ],
        "edges": [
            ("org_c", "ev_rehearsal", "plans", 0.85),
            ("org_c", "ev_concert", "plans", 0.9),   # distractor has HIGHER weight
        ],
        "query": {"anchors": ["org_c"], "type": "event",
                  "constraints": {"date": "2026-07-03", "event_type": "meeting"},
                  "required": ["event_type"]},
        "gold": "ev_rehearsal", "expect": "CONFIRMED",
    },
    # 4. Parent-climb: person -> org -> upcoming meeting -> missing requirement.
    #    No direct fact links the person to the requirement.
    {
        "id": "conv_parent_climb_reason",
        "desc": "person->org->meeting->missing docs; answer is only reachable by climbing",
        "nodes": [
            ("p_x", "person", "Person X"), ("org_y", "org", "Organization Y"),
            ("ev_meet", "event", "Board meeting", {"date": "2026-06-23"}, "2026-06-23T09:00:00"),
            ("a_docs", "assertion", "Signed budget docs missing for the board meeting",
             {}, "2026-06-20T09:00:00"),
        ],
        "edges": [
            ("p_x", "org_y", "member_of", 0.9),
            ("org_y", "ev_meet", "hosts", 0.85),
            ("a_docs", "ev_meet", "needed_for", 0.9),
            ("a_docs", "org_y", "about", 0.8),
        ],
        "query": {"anchors": ["p_x"], "type": "assertion", "constraints": {},
                  "predictive": True},
        "gold": "a_docs", "expect": "ASSUMED",   # predictive -> capped at ASSUMED
    },
    # 5. NEAR-MISS TRAP: same people, wrong time. Two trips with the same group;
    #    question constrains to a date window that only one trip falls in.
    {
        "id": "conv_nearmiss_wrong_time",
        "desc": "same participants; date window must exclude the earlier trip",
        "nodes": [
            ("p_a", "person", "Sam"), ("p_b", "person", "Jo"),
            ("ev_trip1", "event", "Ski trip", {"date": "2026-01-10"}, "2026-01-10T08:00:00"),
            ("ev_trip2", "event", "Lake trip", {"date": "2026-06-15"}, "2026-06-15T08:00:00"),
        ],
        "edges": [
            ("p_a", "ev_trip1", "attends", 0.9), ("p_b", "ev_trip1", "attends", 0.9),
            ("p_a", "ev_trip2", "attends", 0.9), ("p_b", "ev_trip2", "attends", 0.9),
        ],
        "query": {"anchors": ["p_a", "p_b"], "type": "event",
                  "constraints": {"date_range": ("2026-06-01", "2026-06-30")}},
        "gold": "ev_trip2", "expect": "CONFIRMED",
    },
    # 6. Single cue, three tied candidates -> must NOT confirm (honest ASSUMED/UNKNOWN).
    {
        "id": "conv_single_cue_cannot_confirm",
        "desc": "one cue, three equally-linked movies; convergence absent -> not CONFIRMED",
        "nodes": [
            ("p_actor", "person", "The actor"),
            ("m1", "topic", "Movie A", {"kind": "movie"}, "2011-01-01"),
            ("m2", "topic", "Movie B", {"kind": "movie"}, "2015-01-01"),
            ("m3", "topic", "Movie C", {"kind": "movie"}, "2019-01-01"),
        ],
        "edges": [
            ("p_actor", "m1", "acted_in", 0.85), ("p_actor", "m2", "acted_in", 0.85),
            ("p_actor", "m3", "acted_in", 0.85),
        ],
        "query": {"anchors": ["p_actor"], "type": "topic", "constraints": {"kind": "movie"}},
        "gold": None, "expect": "not_CONFIRMED",
    },
    # 7. HYPOTHESIS-NODE territory: cues point at a slot with NO existing node.
    #    date + org known, but the event itself was never recorded. Today this
    #    should be UNKNOWN; after hypothesis nodes it becomes ASSUMED(slot).
    {
        "id": "conv_hypothesis_slot_missing",
        "desc": "date+org+people converge but the event node does not exist yet",
        "nodes": [
            ("p_anna", "person", "Anna"), ("p_marco", "person", "Marco"),
            ("org_iaeste", "org", "IAESTE"),
        ],
        "edges": [
            ("p_anna", "org_iaeste", "member_of", 0.9),
            ("p_marco", "org_iaeste", "member_of", 0.9),
        ],
        "query": {"anchors": ["p_anna", "p_marco"], "type": "event",
                  "constraints": {"date": "2026-06-26"}},
        "gold": None, "expect": "UNKNOWN",   # baseline; hypothesis nodes will change target
    },
    # 8. Convergence with a contradiction: the strongest-linked candidate is a
    #    SUPERSEDED value; a newer value should win.
    {
        "id": "conv_supersede_current_wins",
        "desc": "old role strongly linked, new role added later; current should win",
        "nodes": [
            ("p_z", "person", "Dana"),
            ("r_old", "org", "OldCorp"), ("r_new", "org", "NewCorp"),
        ],
        "edges": [
            ("p_z", "r_old", "works_at", 0.9, "2024-01-01T00:00:00"),
            ("p_z", "r_new", "works_at", 0.9, "2026-05-01T00:00:00"),
        ],
        "query": {"anchors": ["p_z"], "type": "org", "constraints": {},
                  "topic_text": "where does Dana currently work"},
        "gold": "r_new", "expect": "ASSUMED",   # recency default -> tagged
    },
    # 9. Two-cue convergence with a one-cue near-miss at HIGHER individual weight.
    {
        "id": "conv_two_cue_beats_one_strong",
        "desc": "target lit by 2 cues (0.7 each); distractor lit by 1 cue (0.9)",
        "nodes": [
            ("cue_song", "topic", "That song"), ("cue_beach", "place", "The beach"),
            ("cue_generic", "topic", "A common tag"),
            ("m_hit", "topic", "The right summer", {"kind": "memory"}),
            ("m_miss", "topic", "A tagged-but-wrong memory", {"kind": "memory"}),
        ],
        "edges": [
            ("cue_song", "m_hit", "evokes", 0.7), ("cue_beach", "m_hit", "evokes", 0.7),
            ("cue_generic", "m_miss", "evokes", 0.9),
        ],
        "query": {"anchors": ["cue_song", "cue_beach", "cue_generic"], "type": "topic",
                  "constraints": {"kind": "memory"}},
        "gold": "m_hit", "expect": "CONFIRMED",
    },
    # 10. NEAR-MISS TRAP: same category, no agenda match. Two IAESTE events in
    #     the window; only one has the asked agenda item.
    {
        "id": "conv_nearmiss_agenda",
        "desc": "two meetings in window; agenda item distinguishes the right one",
        "nodes": [
            ("org_i", "org", "IAESTE"),
            ("ev_ga", "event", "General Assembly",
             {"date": "2026-06-26", "event_type": "meeting", "agenda": "budget vote"},
             "2026-06-26T18:00:00"),
            ("ev_social", "event", "Committee sync",
             {"date": "2026-06-24", "event_type": "meeting", "agenda": "scheduling"},
             "2026-06-24T18:00:00"),
        ],
        "edges": [
            ("org_i", "ev_ga", "plans", 0.85), ("org_i", "ev_social", "plans", 0.9),
        ],
        "query": {"anchors": ["org_i"], "type": "event",
                  "constraints": {"event_type": "meeting", "agenda": "budget vote"},
                  "required": ["agenda"]},
        "gold": "ev_ga", "expect": "CONFIRMED",
    },
]
