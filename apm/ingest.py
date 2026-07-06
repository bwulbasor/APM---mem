"""Phase 3/4 — the LLM as transducer (SPEC.md §3: steps [1] and [9] ONLY).

Three jobs, all strictly bounded:
  ingest(text)        -> nodes/edges into the graph      (step [0]: memory formation)
  parse_question(q)   -> anchors + placeholder            (step [1] PARSE)
  verbalize(verdict)  -> natural language, chain-grounded (step [9] VERBALIZE)

The LLM never reasons. It translates. Everything between [1] and [9] is the engine.
"""
from __future__ import annotations

import json
import re

from .engine import Placeholder, Query
from .graph import Edge, EvidenceGraph, Node, NodeType
from .llm import chat

# ---------------------------------------------------------------- utilities --

def _extract_json(raw: str) -> dict:
    """Free models wrap JSON in prose/fences; dig it out."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in LLM output: {raw[:200]!r}")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError(f"unbalanced JSON in LLM output: {raw[:200]!r}")


def _ask_json(system: str, prompt: str, retries: int = 3,
              role: str | None = None, cacheable: bool = True) -> dict:
    last = None
    for attempt in range(retries):
        try:
            return _extract_json(chat(prompt, system=system, temperature=0.0,
                                      role=role, cacheable=cacheable,
                                      json_mode=True))
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            last = e
    raise RuntimeError(f"LLM produced unparseable JSON after {retries} tries: {last}")


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", label.lower()).strip()


def _offset_ts(iso_ts: str, seconds: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.fromisoformat(iso_ts) + timedelta(seconds=seconds)).isoformat()


def _resolve(graph: EvidenceGraph, label: str) -> str | None:
    """Entity resolution (type-free, query-side) — delegates to apm.entities."""
    from .entities import resolve
    return resolve(graph, label, None)


_TYPES = {t.value: t for t in NodeType}

# ------------------------------------------------------------------ ingest --

INGEST_SYSTEM = """You convert text into knowledge-graph JSON. Output ONLY a JSON object, no prose.

Schema:
{"nodes":[{"label":"...","type":"person|org|event|place|time|topic|assertion",
           "attrs":{},"timestamp":"ISO-8601 or null"}],
 "edges":[{"src":"<node label>","dst":"<node label>","rel":"<verb_phrase>","weight":0.9}]}

Rules:
- One node per distinct entity. Reuse the exact same label string in edges.
- rel MUST come from this controlled vocabulary (snake_case):
    lives_in, works_at, member_of, part_of, located_in, attends, traveled_to,
    about, needed_for, scheduled_at, precedes, contradicts, married_to, dating,
    knows, owns, hosts, plans, needs, presents_at, is_a, related_to
  Pick the closest fit; use related_to ONLY if nothing fits. Never invent new
  relation verbs. Specifically: residence -> lives_in (not resides_in/based_in/
  moved_to), employment -> works_at, event participation -> attends (not
  went_to/participated_in), trips/visits to places -> traveled_to, place
  containment -> located_in (X located_in Y, never "Y includes/contains X").
- Use member_of/part_of for person->organization or item->collection links.
- Statements ABOUT things ("the docs are missing") become an `assertion` node
  linked with `about` / `needed_for` edges.
- Put dates in attrs as {"date":"YYYY-MM-DD"} on event nodes AND set timestamp.
- Measured VALUES (times, durations, prices, scores, quantities, sizes) must become
  `assertion` nodes carrying the value in the label AND attrs, linked to their
  subject: {"label":"Personal best 25:50 in charity 5K","type":"assertion",
  "attrs":{"value":"25:50","metric":"personal_best"}} + edge about -> charity 5K run.
- attr keys use these exact names: "date" (YYYY-MM-DD), "value", "metric",
  "unit", "participants" — never synonyms like amount/price/event_date. The
  "metric" value names WHAT is measured ("mortgage_pre_approval", "5k_time"),
  never its unit or currency.
- weight: 0.9 for stated facts, 0.7 for implied ones.
- INFERENCE IS ALLOWED but must be LABELED: when context strongly implies a
  connection that is never stated outright (the store a coupon was redeemed
  at, the city an event happened in), add that edge with "implied": true and
  weight 0.5. Implied edges may only CONNECT entities already in your output
  — never invent new entities or values for them. State facts as stated
  edges; guess connections only as implied edges.
- Do not invent facts not in the text (implied edges are connections the
  text supports, not new facts).

Example input: "Lena works at CERN. She presents at the AI summit on 2026-03-05."
Example output:
{"nodes":[{"label":"Lena","type":"person","attrs":{},"timestamp":null},
{"label":"CERN","type":"org","attrs":{},"timestamp":null},
{"label":"AI summit","type":"event","attrs":{"date":"2026-03-05"},"timestamp":"2026-03-05T00:00:00"}],
"edges":[{"src":"Lena","dst":"CERN","rel":"works_at","weight":0.9},
{"src":"Lena","dst":"AI summit","rel":"presents_at","weight":0.9}]}"""


EVENTS_ADDENDUM = """

TYPED EVENTS (critical): when the text describes an EVENT — meeting, call, trip,
deadline, party, appointment, purchase, filing, hearing, etc. — emit it as ONE
event node that CARRIES ITS STRUCTURE IN attrs. Do NOT scatter its parts across
separate assertion nodes; the event and its fields must live together:
  {"label":"IAESTE General Assembly","type":"event",
   "attrs":{"kind":"meeting","date":"2026-06-26",
            "participants":["Anna","Marco","Priya"],
            "topic":"budget vote","organization":"IAESTE"},
   "timestamp":"2026-06-26T00:00:00"}
- "kind": the event category — one of: meeting, call, travel, deadline, social,
  appointment, purchase, filing, hearing, other.
- "participants": the people involved (reuse their EXACT labels).
- "topic": what it is about / its agenda, in a few words.
- "organization": the org it belongs to, if any.
- Put the date ON the event node (attrs.date + timestamp), never in a separate
  node. Keep each event a SINGLE structured node — this is what lets the answer
  be PROVEN, not guessed. Still add the normal edges too (attends, plans, about)."""


CRITIC_SYSTEM = """You audit a knowledge-graph extraction for COMPLETENESS.
Compare the transcript against the already-extracted facts and output ONLY the
STATED facts that are MISSING — especially asides, delivery channels,
frequencies, quantities, dates, names, and minor details a first read skims
past. Salience is not your filter; statedness is.

Output ONLY a JSON object, no prose, in this schema:
{"nodes":[{"label":"...","type":"person|org|event|place|time|topic|assertion",
           "attrs":{},"timestamp":"ISO-8601 or null"}],
 "edges":[{"src":"<node label>","dst":"<node label>","rel":"...","weight":0.9}]}

Rules (same as extraction):
- rel MUST come from: lives_in, works_at, member_of, part_of, located_in,
  attends, traveled_to, about, needed_for, scheduled_at, precedes, contradicts,
  married_to, dating, knows, owns, hosts, plans, needs, presents_at, is_a,
  related_to. Never invent new relation verbs.
- Frequencies, durations, quantities, values become `assertion` nodes with the
  value in the label AND attrs ({"value":...,"metric":...,"unit":...}), linked
  `about` their subject.
- Reuse the EXACT existing node labels when connecting to already-extracted
  entities.
- Implied-but-unstated connections: edge with "implied": true, weight 0.5.
- If nothing is missing, output {"nodes":[],"edges":[]}.
- Do not repeat facts that were already extracted."""


def critic_ingest(graph: EvidenceGraph, text: str, extracted: list[str],
                  source: str = "llm:critic",
                  default_ts: str | None = None) -> dict:
    """[v1.10] Delta-critic pass: sees the transcript AND the first pass's
    output, extracts only what is missing. Runs at INGEST time (before any
    question exists), so unlike cued recall it cannot echo a question and
    only ADDS observed facts. Live probe: recovered the two facts every
    extractor on every key dropped (51a45a95 email-source + frequency)."""
    prompt = ("TRANSCRIPT:\n" + text + "\n\nALREADY EXTRACTED:\n"
              + "\n".join(f"- {x}" for x in extracted))
    data = _ask_json(CRITIC_SYSTEM, prompt, role="extract")
    return _merge_extraction(graph, data, source, default_ts)


def ingest(graph: EvidenceGraph, text: str, source: str = "llm:ingest",
           hint: str | None = None, default_ts: str | None = None) -> dict:
    """LLM extraction -> validated nodes/edges. Returns {'nodes': n, 'edges': m} added
    plus 'summary': the labels/triples added (feeds the delta-critic pass).
    `hint` prepends task context (e.g. "this is a User/Assistant dialogue").
    `default_ts` stamps nodes the LLM left undated (use the session/observation
    date) — episodic memory always knows WHEN it learned something."""
    import os
    system = INGEST_SYSTEM + (EVENTS_ADDENDUM if os.environ.get("APM_EVENTS") else "")
    prompt = f"{hint}\n\n{text}" if hint else text
    data = _ask_json(system, prompt, role="extract")
    return _merge_extraction(graph, data, source, default_ts)


def _merge_extraction(graph: EvidenceGraph, data: dict, source: str,
                      default_ts: str | None) -> dict:
    label_to_id: dict[str, str] = {}
    added_n = added_e = 0
    summary: list[str] = []

    from .entities import remember_alias, resolve as entity_resolve

    for idx, nd in enumerate(data.get("nodes", [])):
        label = str(nd.get("label", "")).strip()
        if not label:
            continue
        # models improvise types outside the schema ("item", "object") —
        # dropping those nodes silently is a recall leak; coerce to topic
        ntype = _TYPES.get(str(nd.get("type", "")).lower(), NodeType.TOPIC)
        # extraction order approximates mention order: offset defaulted stamps
        # by index so same-session facts keep a stable, meaningful sequence
        node_ts = nd.get("timestamp") or (
            _offset_ts(default_ts, idx) if default_ts else None)
        existing = entity_resolve(graph, label, ntype)   # type-gated on ingest
        if existing:
            remember_alias(graph, existing, label)       # aliases compound
            label_to_id[label] = existing
            continue
        node_id = f"{ntype.value[:2]}_{_slug(label)}"
        while node_id in graph.nodes:
            node_id += "_"
        attrs = nd.get("attrs") or {}
        graph.add_node(Node(node_id, ntype, label, attrs=attrs,
                            timestamp=node_ts, source=source))
        label_to_id[label] = node_id
        summary.append(label)
        added_n += 1

    # typed-event participants: materialize attends edges from the structured
    # participants attr so convergence (edges) AND proof (attrs) both exist
    import os as _oe
    if _oe.environ.get("APM_EVENTS"):
        for lbl, nid in list(label_to_id.items()):
            node = graph.nodes.get(nid)
            if node is None or node.type != NodeType.EVENT:
                continue
            parts = node.attrs.get("participants")
            if isinstance(parts, list):
                for p in parts:
                    pid = label_to_id.get(p) or _resolve(graph, str(p))
                    if pid and pid != nid and graph.nodes[pid].type == NodeType.PERSON:
                        graph.add_edge(Edge(pid, nid, "attends", 0.85, source=source))
                        added_e += 1

    from .canon import canon_rel

    for ed in data.get("edges", []):
        src = label_to_id.get(ed.get("src")) or _resolve(graph, str(ed.get("src", "")))
        dst = label_to_id.get(ed.get("dst")) or _resolve(graph, str(ed.get("dst", "")))
        rel = str(ed.get("rel", "related_to")).strip() or "related_to"
        if not src or not dst or src == dst:
            continue
        # extractor-agnostic vocabulary: engine rules key on canonical rels
        rel = canon_rel(rel, graph.nodes[src].type, graph.nodes[dst].type)
        try:
            weight = min(0.95, max(0.1, float(ed.get("weight", 0.8))))
        except (TypeError, ValueError):
            weight = 0.8
        edge_source = source
        if ed.get("implied"):
            # extractor-inferred association (v1.8): may guide retrieval and
            # hedged answers, may never carry a CONFIRMED — weight-capped so
            # it cannot cross theta_confirm alone, provenance-marked so the
            # gate can cap any chain that leans on it
            weight = min(weight, 0.55)
            edge_source = f"{source}:implied"
        graph.add_edge(Edge(src, dst, rel, weight, source=edge_source))
        summary.append(f"{graph.nodes[src].label} {rel} {graph.nodes[dst].label}")
        added_e += 1

    return {"nodes": added_n, "edges": added_e, "summary": summary}


# ------------------------------------------------------------- parse query --

PARSE_SYSTEM = """You convert a question into a graph-query JSON. Output ONLY JSON, no prose.

Schema:
{"anchors":["<entity names mentioned or implied>"],
 "unknown_type":"person|org|event|place|time|topic|assertion",
 "constraints":{},
 "predictive":false,
 "mode":"resolve",
 "options":[],
 "under":null}

Rules:
- anchors: the KNOWN entities in the question (people, orgs, places, dates as context).
- unknown_type: what kind of thing the question asks FOR.
- "How often / how frequently" questions ask for a stored FREQUENCY value —
  use mode "resolve" with unknown_type "assertion", NEVER "count".
- constraints: attributes the answer must satisfy, e.g. {"date":"2026-06-26"}.
- predictive: true only for questions about intents/future ("why is X calling?").
- mode: "resolve" normally. Use "first"/"last" when the question asks WHICH of
  several things happened first/last/earlier/most recently. Put the named
  alternatives in options (e.g. ["the workshop","the yoga session"]); leave
  options empty if the question doesn't name them. Use "count" when the
  question asks HOW MANY things match ("how many trips did I take?").
  Use "suggest" when the question asks for recommendations, suggestions, or
  advice tailored to the asker ("what would go well with my setup?").
  Use "span" when the question asks for the TIME BETWEEN two things ("how many
  days between A and B?", "how long after A did B happen?") — put both named
  things in options. Use "sum" when the question asks for a TOTAL quantity
  across several things ("how many days did I spend camping this year?") —
  set constraints {"unit": "days", "activity": "camping"}. "How long / how
  many weeks did it take me to do X (and Y)" also asks to TOTAL stored
  durations -> mode "sum" with constraints {"unit":"weeks"} and the named
  activities in options; "span" is ONLY for the gap between two DISTINCT
  dated events, never for the duration of activities.
- under: if the question restricts answers to things WITHIN a named container
  (a place, an org, a trip: "restaurants in Tanzania", "events on my Japan
  trip"), put that container's name here; else null.

Example input: "There's something happening on 2026-06-26 with Anna and Marco - what is it?"
Example output:
{"anchors":["Anna","Marco"],"unknown_type":"event",
 "constraints":{"date":"2026-06-26"},"predictive":false}

Example input: "Why might Lena be emailing me?"
Example output:
{"anchors":["Lena"],"unknown_type":"assertion","constraints":{},"predictive":true,
 "mode":"resolve","options":[]}
(intent/why/reason questions ask for an `assertion` — a statement explaining the situation)

Example input: "What was my personal best time in the charity 5K?"
Example output:
{"anchors":["User","charity 5K"],"unknown_type":"assertion",
 "constraints":{},"predictive":false,"mode":"resolve","options":[]}
(questions asking for a VALUE — a time, price, score, quantity — ask for the
`assertion` node that records that value)

Example input: "Which did I attend first, the pottery class or the wine tasting?"
Example output:
{"anchors":["User"],"unknown_type":"event","constraints":{},"predictive":false,
 "mode":"first","options":["pottery class","wine tasting"]}

Example input: "How many days before I bought the iPad did I visit the Holiday Market?"
Example output:
{"anchors":["User"],"unknown_type":"time","constraints":{},"predictive":false,
 "mode":"span","options":["bought the iPad","Holiday Market visit"]}

Example input: "How many weeks did it take me to finish the beginner course and the advanced course?"
Example output:
{"anchors":["User"],"unknown_type":"time","constraints":{"unit":"weeks"},
 "predictive":false,"mode":"sum","options":["beginner course","advanced course"]}
(durations of activities are TOTALED from stored duration facts — sum, not span)"""


def parse_question(graph: EvidenceGraph, question: str,
                   hint: str | None = None) -> Query:
    prompt = f"{hint}\n\n{question}" if hint else question
    data = _ask_json(PARSE_SYSTEM, prompt, role="parse")
    anchors, unresolved = [], []
    for label in data.get("anchors", []):
        node_id = _resolve(graph, str(label))
        if node_id:
            anchors.append(node_id)
        else:
            unresolved.append(str(label))
    ntype = _TYPES.get(str(data.get("unknown_type", "")).lower(), NodeType.TOPIC)
    constraints = data.get("constraints") or {}
    # anchor ids double as participant constraints where the LLM used names
    if "participants" in constraints:
        constraints["participants"] = [
            _resolve(graph, str(p)) or p for p in constraints["participants"]]
    if not anchors:
        raise ValueError(f"no anchors resolvable in graph for: {question!r} "
                         f"(LLM said {data.get('anchors')})")
    mode = str(data.get("mode", "resolve"))
    if mode not in ("resolve", "first", "last", "count", "suggest", "span", "sum"):
        mode = "resolve"
    options = []
    for label in data.get("options", []) or []:
        node_id = None
        if mode in ("span", "first", "last"):
            # temporal endpoints are EVENTS: "Sunday mass at St. Mary's"
            # must resolve to the mass, not the church (a place's timestamp
            # is a mention-stamp — spans over it are silently wrong)
            from .graph import NodeType as _NT
            from .entities import resolve as _eres
            node_id = _eres(graph, str(label), _NT.EVENT)
        node_id = node_id or _resolve(graph, str(label))
        if node_id:
            options.append(node_id)
        else:
            options.append(str(label))  # unresolved: engine reports it honestly
    # constraint VALUES naming real entities are anchors in disguise —
    # "where did I redeem the $5 coupon" parses the coupon into constraints,
    # but as an anchor it powers intersection (the movie-recall mechanic).
    # v1.6 guards (dev2 gpt4_5dcc0aab: 'clean'/'last month' fuzzy-resolved
    # onto the GOLD ASSERTION, recruiting the answer as an anchor — anchors
    # are excluded from candidacy, so the answer became unfindable):
    #   - never promote a node of the unknown's own type (it may BE the answer)
    #   - short values fuzzy-match junk via containment; require some length
    for v in list(constraints.values()):
        if isinstance(v, str) and len(v) >= 6:
            nid = _resolve(graph, v)
            if nid and nid not in anchors and graph.node(nid).type != ntype:
                anchors.append(nid)
    under = None
    if data.get("under"):
        label = str(data["under"])
        under = _resolve(graph, label)
        # v1.14 distinguishing-token check (invention bc8a6e93: "UNCLE's
        # birthday party" fuzzy-resolved onto the NIECE's party node — the
        # question's false premise was silently substituted): if the query
        # names a content token the resolved node's label/aliases lack, the
        # scope is NOT that node — treat as unresolved and let the premise
        # machinery see it
        if under:
            node = graph.node(under)
            hay = _norm(node.label + " " + " ".join(
                node.attrs.get("aliases", [])))
            missing = [t for t in _norm(label).split()
                       if len(t) > 3 and t.rstrip("s") not in
                       {w.rstrip("s") for w in hay.split()}
                       and not any(w.startswith(t[:5]) for w in hay.split())]
            if missing:
                unresolved.append(label)
                under = None
        # an unresolvable SCOPE degrades to unscoped — containers are
        # descriptive ("my city", "the local theater"), not premises;
        # treating them as false-premise signals refused legit questions
        # (v0.8 regression: 58bf7951 knew the answer and refused anyway)
    return Query(anchors=anchors,
                 placeholder=Placeholder(ntype, constraints, under=under),
                 predictive=bool(data.get("predictive", False)),
                 mode=mode, options=options, topic_text=question,
                 unresolved=[u for u in unresolved if u.lower() != "user"])


# ------------------------------------------------------- relevance veto ----

VETO_SYSTEM = """You filter an enumerated candidate list against a question.
Output ONLY JSON: {"relevant": ["<exact labels that answer the question>"]}.
Keep a label ONLY if it clearly satisfies what the question asks about
(e.g. for "items to pick up or return from a store", keep concrete pickup/return
items, drop general advice/plans/tips). When unsure, DROP it."""


def relevance_veto(verdict, question: str, graph: EvidenceGraph):
    """LLM veto over an enumerated ASSUMED verdict (count/suggest bundles).

    Safety property: the veto can only SHRINK the member set — a bad veto costs
    coverage (undercount), never correctness-with-confidence. The verdict stays
    ASSUMED; the enumeration and the filtered count are updated in place.
    """
    members = (verdict.answer.attrs or {}).get("members") if verdict.answer else None
    if not members:
        return verdict
    def describe(m):
        node = graph.node(m)
        rels = sorted({f"{e.rel}->{graph.node(o).label[:30]}"
                       for o, e in graph.neighbors(m)})[:4]
        return f"- {node.label} (connections: {'; '.join(rels) or 'none'})"
    labels = [graph.node(m).label for m in members if m in graph.nodes]
    data = _ask_json(VETO_SYSTEM,
                     f"Question: {question}\nCandidates:\n" +
                     "\n".join(describe(m) for m in members if m in graph.nodes))
    keep = {str(x) for x in data.get("relevant", [])}
    filtered = [m for m in members
                if m in graph.nodes and graph.node(m).label in keep]
    if len(filtered) == len(members):
        return verdict
    dropped = len(members) - len(filtered)
    verdict.answer.attrs["members"] = filtered
    verdict.answer.attrs["count"] = len(filtered)
    verdict.answer.label = str(len(filtered))
    verdict.chain.add("veto", f"relevance veto dropped {dropped} candidate(s) "
                              f"not matching the question; kept: "
                              f"{', '.join(graph.node(m).label for m in filtered) or '(none)'}")
    return verdict


# --------------------------------------------------------------- verbalize --

VERBALIZE_SYSTEM = """You turn a reasoning engine's verdict into one short natural answer.

HARD RULES:
- State ONLY what the verdict and its evidence chain contain. Add NOTHING.
- If the verdict NAMES an answer (CONFIRMED or ASSUMED), you MUST deliver that
  answer. Never refuse or say "I don't know" when the verdict has an answer,
  and never invent reasons ("X does not provide access...") not in the chain.
- CONFIRMED: answer plainly, then give the reason from the chain in one sentence.
- ASSUMED: hedge explicitly ("probably", "my best guess") and NAME the reason
  it is a guess (e.g. "based only on recency", "predicted intent").
- UNKNOWN: say you don't know, state what WAS established, and what evidence
  would resolve it. Never guess.
- If the verdict is a "grounded facts for suggestion" bundle: you MAY compose
  helpful suggestions, but each must be motivated by a listed grounding fact
  ("since you shoot Sony A7..."). Frame clearly as suggestions, never as
  remembered facts.
- 1-3 sentences total. No markdown, no preamble."""


def verbalize(verdict, question: str) -> str:
    payload = f"Question: {question}\n\nEngine verdict:\n{verdict.render()}"
    return chat(payload, system=VERBALIZE_SYSTEM, temperature=0.0,
                role="verbalize").strip()
