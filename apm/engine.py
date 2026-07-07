"""The APM reasoning loop, steps [3]-[8] + [10] (SPEC.md §3).

Steps [1] PARSE and [9] VERBALIZE are the LLM's jobs and arrive in Phase 4;
here queries come in already structured (anchors + placeholder).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .activation import spread
from .fallback import Candidate, project_from_parents, recency_default
from .gate import Chain, Verdict, VerdictClass, path_text
from .graph import EvidenceGraph, Node, NodeType, Status
from .scoring import THETA_CONFIRM, THETA_FLOOR, TIE_EPSILON, match_constraints, noisy_or


@dataclass
class Placeholder:
    type: NodeType
    constraints: dict = field(default_factory=dict)
    # defining constraint keys: candidates missing these attrs are excluded
    required: frozenset = frozenset()
    # taxonomy scope: restrict candidates to the subtree under this node id
    # (a category like cat_place, or an instance parent like Tanzania)
    under: str | None = None


@dataclass
class Query:
    anchors: list[str]
    placeholder: Placeholder
    predictive: bool = False   # event-triggered intent prediction: capped at ASSUMED
    signature: str | None = None  # identity for write-back/retraction
    # mode "resolve" answers WHAT the unknown is; "first"/"last" ORDER a
    # candidate set by time; "count" answers HOW MANY match. options: explicit
    # candidate node ids (else the candidate set comes from the placeholder).
    mode: str = "resolve"
    options: list[str] = field(default_factory=list)
    # the raw question text, for semantic reranking among weak candidates
    topic_text: str | None = None
    # named entities from the question that resolved to NOTHING in the graph.
    # A question referencing an unknown entity may rest on a false premise
    # ("my job at Google" when Google was never mentioned) — abstention signal.
    unresolved: list[str] = field(default_factory=list)

    def sig(self) -> str:
        if self.signature:
            return self.signature
        items = ",".join(f"{k}={v}" for k, v in sorted(self.placeholder.constraints.items()))
        return f"{self.placeholder.type.value}?[{items}]anchors={'|'.join(sorted(self.anchors))}"


class Engine:
    def __init__(self, graph: EvidenceGraph):
        self.graph = graph

    # extraction and parsing routinely disagree on place vs org vs topic for
    # the same real thing ("Target": org or place?) — exact-type matching
    # silently zeroes the candidate set (v0.8 bug 51a45a95). Same-group types
    # match at a small penalty; exact type stays preferred.
    _TYPE_GROUP = {NodeType.PLACE, NodeType.ORG, NodeType.TOPIC}

    def _type_factor(self, node_type, want_type) -> float:
        if node_type == want_type:
            return 1.0
        if node_type in self._TYPE_GROUP and want_type in self._TYPE_GROUP:
            return 0.85
        return 0.0

    def _scope(self, query: Query) -> set[str] | None:
        """Resolve the placeholder's taxonomy scope to a membership set."""
        if not query.placeholder.under:
            return None
        from .taxonomy import subtree
        if query.placeholder.under not in self.graph.nodes:
            return set()
        # scope means "belonging to the container" — that includes things
        # ASSOCIATED with it (a play performed_at the theater), not only
        # containment children (v0.8 bug: subtree-only excluded the answer)
        members = subtree(self.graph, query.placeholder.under)
        members |= {o for o, _ in self.graph.neighbors(query.placeholder.under)}
        return members

    def answer(self, query: Query, now: datetime | None = None,
               write_back: bool = True) -> Verdict:
        if query.mode in ("first", "last"):
            return self._order(query, now)
        if query.mode == "count":
            return self._count(query, now)
        if query.mode == "suggest":
            return self._suggest(query, now)
        if query.mode == "span":
            return self._span(query, now)
        if query.mode == "sum":
            return self._sum(query, now)
        ph = query.placeholder
        chain = Chain()

        # [3] DIRECT LOOKUP -------------------------------------------------
        # a direct hit requires the constraints to identify a UNIQUE node;
        # an ambiguous strict match is not knowledge, it's a candidate set
        strict_hits = [
            node for node in self.graph.by_type(ph.type)
            if node.id not in query.anchors and ph.constraints
            # our own ASSUMED write-backs carry the query's constraints as
            # attrs — a repeat query would strict-match its OWN GUESS and
            # launder it into CONFIRMED (v1.9 live bug: cued second pass
            # direct-confirmed the first pass's tiebreak pick)
            and node.status != Status.ASSUMED
            and match_constraints(node, ph.constraints, strict=True, required=ph.required) == 1.0
        ]
        if len(strict_hits) == 1:
            node = strict_hits[0]
            chain.add("direct", f"{node.label} satisfies all constraints "
                                f"({node.status.value}, source: {node.source})")
            # the direct path must obey the same confirm-caps as ranking
            # (v1.7: strict attr matches bypassed every guard)
            snap = self._snapshot_guard(query, node, chain)
            if snap == "refuse":
                return self._unknown(query, {}, chain, ph,
                                     note="time-scope unverifiable")
            return self._gate(query, Candidate(node, node.confidence, []),
                              chain, now, write_back,
                              force_class=(VerdictClass.ASSUMED if snap
                                           else None),
                              default_rule=("stale-snapshot cap" if snap
                                            else None))
        chain.add("note", f"direct lookup: {len(strict_hits) or 'no'} strict "
                          f"match(es) — not uniquely resolvable (miss)")

        # [4] ACTIVATE ------------------------------------------------------
        activated = spread(self.graph, query.anchors, now)

        # [5] INTERSECT: candidates of the placeholder's type ---------------
        scope = self._scope(query)
        drop_advice = self._drop_advice(query)
        freq_q = self._is_frequency_query(query.topic_text or "")
        subj_toks = self._query_subject_tokens(query.topic_text or "")
        cands: list[Candidate] = []
        for node_id, paths in activated.items():
            node = self.graph.node(node_id)
            type_f = self._type_factor(node.type, ph.type)
            if type_f == 0.0 or node.id in query.anchors:
                continue
            if scope is not None and node.id not in scope:
                continue
            if node.status == Status.RETRACTED:
                continue
            if drop_advice and self._is_advice(node):
                continue     # a recommendation is not the user's OWN fact
            if freq_q and node.type == NodeType.ASSERTION:
                if not self._looks_like_frequency_assertion(node):
                    continue
                if not self._supports_subject_tokens(node, subj_toks):
                    continue
            m = match_constraints(node, ph.constraints, strict=False, required=ph.required)
            if m == 0.0:
                continue
            cands.append(Candidate(node, noisy_or(paths) * m * type_f, paths, m))
        if freq_q and ph.type == NodeType.ASSERTION:
            cands.extend(self._frequency_event_candidates(activated, subj_toks))

        # [6a]+[6b] FALLBACK: parent climb + constraint projection ----------
        best = max(cands, key=lambda c: c.score, default=None)
        if best is None or best.score < THETA_CONFIRM:
            projected = project_from_parents(self.graph, activated, ph.type, ph.constraints, ph.required)
            by_id = {c.node.id: c for c in cands}
            for c in projected:  # keep whichever route scores the candidate higher
                if c.node.id not in by_id or c.score > by_id[c.node.id].score:
                    by_id[c.node.id] = c
            cands = list(by_id.values())
        if freq_q and ph.type == NodeType.ASSERTION:
            cands = [c for c in cands if self._looks_like_frequency_assertion(c.node)
                     and self._supports_subject_tokens(c.node, subj_toks)]

        # [6d] CONTRADICTION: newer evidence attacks older (SPEC.md §6) ------
        contra_notes = self._apply_contradictions(cands)

        # [7] RANK ----------------------------------------------------------
        # keep the unfiltered set: superseded values sit BELOW the floor by
        # construction (auto_supersede x0.35) yet are exactly what a
        # past-value question asks for
        all_cands = list(cands)
        # exact-type dominance: when exact-type candidates clear the floor,
        # cross-type (type-group) candidates are noise, not competition
        exact_ok = [c for c in cands
                    if c.node.type == ph.type and c.score >= THETA_FLOOR]
        if exact_ok:
            cands = exact_ok
        cands.sort(key=lambda c: -c.score)
        for note in contra_notes:
            chain.add("contradiction", note)

        if not cands:
            return self._unknown(query, activated, chain, ph)

        attr_subj = self._attribute_subject_tokens(query)
        if attr_subj:
            kept = [c for c in cands
                    if self._supports_attribute_subject(c.node, attr_subj)]
            if kept:
                if len(kept) < len(cands):
                    chain.add("note", "dropped candidates that do not support "
                                      "the named subject of the requested attribute")
                cands = kept
            else:
                chain.add("note", "no candidate supports the named subject of "
                                  "the requested attribute")
                return self._unknown(query, activated, chain, ph,
                                     note="possible false premise")

        top = cands[0]
        tied = [c for c in cands if top.score - c.score < TIE_EPSILON]

        # premise check: the question referenced entities we have NO memory of.
        # Weak evidence + unknown reference = likely false premise -> refuse.
        if query.unresolved and top.score < 0.5                 and not self._rel_corroborated(query, top):
            chain.add("note", f"question references entities with no memory: "
                              f"{', '.join(query.unresolved)}")
            return self._unknown(query, activated, chain, ph,
                                 note="possible false premise")
        if query.unresolved and ph.type == NodeType.ASSERTION and \
                not self._supports_unresolved_entity(query.unresolved, top.node):
            chain.add("note", f"top assertion does not support the unresolved "
                              f"entity reference(s): {', '.join(query.unresolved)}")
            return self._unknown(query, activated, chain, ph,
                                 note="possible false premise")

        # [6c] DEFAULTS: undiscriminated tie -> semantic pick, else recency ---
        if len(tied) > 1:
            # type-grouping widens tie bands; when exact-type candidates exist
            # in the tie, cross-type ones are noise — drop them before picking
            exact = [c for c in tied if c.node.type == ph.type]
            if exact:
                tied = exact
        if len(tied) > 1 and query.topic_text:
            # the question's own verb discriminates ties: "playlist I CREATED"
            # -> prefer candidates reached via a 'created' edge. Deterministic,
            # graph-native, and runs BEFORE embeddings (names beat topics).
            q_low = query.topic_text.lower()
            def rel_hit(c):
                rels = {e.rel for p_ in c.paths for e in p_.edges}
                return any(tok in q_low for r in rels
                           for tok in r.lower().replace("_", " ").split()
                           if len(tok) > 3)
            rel_matched = [c for c in tied if rel_hit(c)]
            if rel_matched and len(rel_matched) < len(tied):
                chain.add("note", "tie narrowed by question-verb/edge-relation match")
                tied = rel_matched
        if len(tied) > 1:
            pick = None
            if query.topic_text:
                # LLM tie-arbiter (v1.6, SPEC §3.5 role 2): picks among the
                # ENGINE's tied candidates using their graph context; stays
                # ASSUMED; falls back to the embedding tiebreak on any failure
                from .arbiter import arbitrate
                def _ev(c):
                    return sorted({f"{e.rel} -> "
                                   f"{self.graph.node(o).label[:45]}"
                                   for o, e in self.graph.neighbors(c.node.id)
                                   if o in self.graph.nodes})[:5]
                i = arbitrate(query.topic_text,
                              [(c.node.label, _ev(c)) for c in tied])
                if i is not None:
                    pick = tied[i]
                    self._describe(chain, pick)
                    chain.add("default", f"{len(tied)} candidates tied; "
                                         f"LLM arbiter picks {pick.node.label} "
                                         f"from the closed candidate list")
                    return self._gate(query, pick, chain, now, write_back,
                                      force_class=VerdictClass.ASSUMED,
                                      default_rule="LLM-arbitrated tiebreak")
                from .embed import semantic_pick
                i = semantic_pick(query.topic_text,
                                  [c.node.label for c in tied])
                if i is not None:
                    pick = tied[i]
                    self._describe(chain, pick)
                    chain.add("default", f"{len(tied)} candidates tied; semantic "
                                         f"relevance picks {pick.node.label}")
                    return self._gate(query, pick, chain, now, write_back,
                                      force_class=VerdictClass.ASSUMED,
                                      default_rule="semantic tiebreak")
            pick = recency_default(tied)
            if pick is None:
                return self._unknown(query, activated, chain, ph,
                                     note=f"{len(tied)} candidates tied, no default applies")
            self._describe(chain, pick)
            chain.add("default", f"{len(tied)} candidates tied within {TIE_EPSILON}; "
                                 f"recency picks {pick.node.label} ({pick.node.timestamp})")
            return self._gate(query, pick, chain, now, write_back,
                              force_class=VerdictClass.ASSUMED, default_rule="recency")

        self._describe(chain, top)
        # past-value guard (v1.5, dev2 invention 07741c44: "initially keep
        # sneakers in the closet" CONFIRMED when gold was the SUPERSEDED
        # under-the-bed): recency machinery prefers the CURRENT value, but
        # "initially / previously / used to" asks for the OLD one — the
        # current value is precisely what the question is NOT asking.
        _PAST = ("initially", "originally", "previously", "previous ",
                 "used to", "formerly", "at first")
        past_q = query.topic_text and any(
            w in query.topic_text.lower() for w in _PAST)
        if past_q:
            old = [c for c in all_cands if self._is_superseded(c)]
            if old:
                pick = max(old, key=lambda c: c.score)
                self._describe(chain, pick)
                chain.add("note", "past-value question: answering with the "
                                  "SUPERSEDED value (a newer fact replaced it)")
                return self._gate(query, pick, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="superseded (previous) value")
            if top.score >= THETA_CONFIRM:
                chain.add("note", "question asks for a PREVIOUS value but no "
                                  "superseded evidence exists in memory — the "
                                  "current value cannot be confirmed as it")
                return self._gate(query, top, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="past-value guard")
        # stale-snapshot guard (v1.7, slice3 inventions 031748ae/184da446)
        if top.score >= THETA_CONFIRM:
            snap = self._snapshot_guard(query, top.node, chain)
            if snap == "refuse":
                return self._unknown(query, activated, chain, ph,
                                     note="time-scope unverifiable")
            if snap:
                return self._gate(query, top, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="stale-snapshot cap")
        # question-echo cap (v1.4, 58bf7951: engine CONFIRMED "play at
        # community theater" for "what play did I attend at the community
        # theater?"): a candidate built from the question's own words
        # establishes the PREMISE, not the answer — it cannot CONFIRM.
        if top.score >= THETA_CONFIRM and query.topic_text and \
                self._echoes_question(top.node.label, query.topic_text):
            chain.add("note", "top candidate restates the question rather "
                              "than answering it — premise established, "
                              "answer not")
            return self._gate(query, top, chain, now, write_back,
                              force_class=VerdictClass.ASSUMED,
                              default_rule="question-echo cap")
        _CHANGE = ("move", "moved", "relocat", "now ", "currently", "after",
                   "new ", "changed", "switched", "latest", "most recent")
        change_q = query.topic_text and any(
            w in query.topic_text.lower() for w in _CHANGE)
        no_supersede = not any("contradict" in s_.kind or "supersed" in s_.text
                               for s_ in chain.steps)
        if top.score >= THETA_CONFIRM and change_q and no_supersede:
            # the question asks about a CHANGE but nothing in the chain shows
            # one — the updated fact was likely never captured; refuse to
            # CONFIRM the stale value (structural anti-invention)
            chain.add("note", "question implies an update not evidenced in the "
                              "chain — refusing to confirm a possibly-stale value")
            return self._gate(query, top, chain, now, write_back,
                              force_class=VerdictClass.ASSUMED,
                              default_rule="unverified-update guard")
        # recency-conflict guard (v1.16, OOD faithfulness — lawfirm_0015:
        # CONFIRMED a client's OLD mailing address; the update lived in a
        # separate assertion node and the phrasing 'current ... on file' never
        # matched the keyword update-guard). STRUCTURAL, phrasing-independent:
        # if the graph holds a strictly MORE RECENT candidate for this same
        # subject+attribute whose value DIFFERS from the top pick, the top may
        # be stale -> never CONFIRM it; answer with the most recent, tagged.
        # NARROW to genuine value-updates: top and the rival are both ASSERTION
        # nodes about the SAME subject (a client's address, a matter's fee).
        # Competing ENTITIES (the movie intersection) and sequenced EVENTS
        # (what came after the gym) are NOT assertions -> untouched, so this
        # cannot corrupt resolve/ordering where recency is not the answer.
        if (top.score >= THETA_FLOOR and top.node.timestamp
                and top.node.type == NodeType.ASSERTION):
            from .graph import parse_ts as _pts
            _SUBJ = (NodeType.PERSON, NodeType.ORG, NodeType.PLACE, NodeType.TOPIC)
            def _subj(nid):
                return {o for o, _ in self.graph.neighbors(nid)
                        if o in self.graph.nodes
                        and self.graph.node(o).type in _SUBJ}
            t_top, top_subj = _pts(top.node.timestamp), _subj(top.node.id)
            newer = [c for c in all_cands
                     if c.node.id != top.node.id and c.node.timestamp
                     and c.node.type == NodeType.ASSERTION
                     and _pts(c.node.timestamp) > t_top
                     and c.score >= THETA_FLOOR
                     and (_subj(c.node.id) & top_subj)
                     and not self._same_value(top.node, c.node)]
            if newer:
                pick = max(newer, key=lambda c: _pts(c.node.timestamp))
                self._describe(chain, pick)
                chain.add("note", "a more recent, conflicting value exists for "
                                  "the same subject — the older value cannot be "
                                  "confirmed as current; answering with the most "
                                  "recent (tagged, not proof)")
                return self._gate(query, pick, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="recency-conflict (possibly-stale) guard")
        if top.score >= THETA_CONFIRM:
            if query.unresolved:
                # strong evidence for the answer, but part of the question is
                # unknown to us — never CONFIRM over an unverified premise
                chain.add("note", f"unverified references in question: "
                                  f"{', '.join(query.unresolved)}")
                return self._gate(query, top, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="answer despite unverified premise")
            from .scoring import unevidenced_keys
            silent = unevidenced_keys(top.node, ph.constraints)
            if silent:
                # v1.14 (validation inventions c8c3f81d/21436231/bc8a6e93:
                # 'favorite shoes' with no preference evidence, wrong-species
                # catch, right cake for an UNVERIFIED occasion): associative
                # convergence may RANK a candidate, but CONFIRMED demands
                # every asked condition be at least EVIDENCED (soft matches
                # like date windows count; silence does not)
                chain.add("note", f"candidate is well-supported but silent "
                                  f"on asked condition(s): "
                                  f"{', '.join(silent)} — delivered as a "
                                  f"best guess")
                return self._gate(query, top, chain, now, write_back,
                                  force_class=VerdictClass.ASSUMED,
                                  default_rule="constraints not fully evidenced")
            return self._gate(query, top, chain, now, write_back)
        if top.score >= THETA_FLOOR:
            return self._gate(query, top, chain, now, write_back,
                              force_class=VerdictClass.ASSUMED,
                              default_rule="weak evidence (best supported candidate)")
        return self._unknown(query, activated, chain, ph)

    def _order(self, query: Query, now: datetime | None) -> Verdict:
        """[Phase 6] Ordering query: 'which came first/last, A or B?'
        The comparison is deterministic over dated nodes, so a complete date
        set yields CONFIRMED; any undated candidate degrades honestly."""
        from .graph import parse_ts
        from datetime import datetime as _dt
        chain = Chain()
        ph = query.placeholder

        if query.options:
            nodes = [self.graph.node(o) for o in query.options if o in self.graph.nodes]
            missing_opts = [o for o in query.options if o not in self.graph.nodes]
        else:
            activated = spread(self.graph, query.anchors, now)
            drop_advice = self._drop_advice(query)
            nodes = [self.graph.node(nid) for nid, paths in activated.items()
                     if self.graph.node(nid).type == ph.type
                     and not (drop_advice and self._is_advice(self.graph.node(nid)))
                     and match_constraints(self.graph.node(nid), ph.constraints,
                                           required=ph.required) > 0.0]
            missing_opts = []

        def node_date(n: Node):
            ts = n.timestamp or n.attrs.get("date") or self._linked_date(n)
            return parse_ts(ts) if ts else None

        dated = [(n, node_date(n)) for n in nodes]
        undated = [n for n, d in dated if d is None]
        dated = [(n, d) for n, d in dated if d is not None]

        if len(dated) < 2:
            chain.add("note", f"ordering needs >=2 dated candidates; have "
                              f"{len(dated)} dated, {len(undated)} undated, "
                              f"{len(missing_opts)} unresolved")
            return Verdict(VerdictClass.UNKNOWN, None, 0.0, chain,
                           missing_evidence="timestamps/dates on the candidates "
                                            "being compared")
        pick = min(dated, key=lambda x: x[1]) if query.mode == "first" \
            else max(dated, key=lambda x: x[1])
        for n, d in sorted(dated, key=lambda x: x[1]):
            chain.add("direct", f"{n.label} @ {d.isoformat()}")
        if undated or missing_opts:
            names = ", ".join([n.label for n in undated] + missing_opts)
            chain.add("note", f"undatable candidates excluded: {names}")
            return Verdict(VerdictClass.ASSUMED, pick[0], 0.5, chain,
                           default_rule=f"{query.mode} among DATED candidates only")
        if not query.options and not ph.constraints:
            # ordering an UNCONSTRAINED candidate set is never knowledge: the
            # set is "everything of this type", not "the things asked about"
            chain.add("note", "candidate set unconstrained — ordering is a guess")
            return Verdict(VerdictClass.ASSUMED, pick[0], 0.5, chain,
                           default_rule=f"{query.mode} over unconstrained candidates")
        # mention-stamp guard (v1.5, dev2 invention gpt4_0b2f1d21: "coffee
        # maker first" CONFIRMED off session mention-times): a node without a
        # stated date is ordered by WHEN IT WAS TALKED ABOUT — a usable
        # heuristic (v0.5 GPS fix) but never CONFIRMED knowledge. Span mode
        # has had this guard since v0.6; ordering never did.
        stamped = [n.label for n, _ in dated if not n.attrs.get("date")]
        if stamped:
            chain.add("note", f"order inferred from mention time, not stated "
                              f"dates: {', '.join(stamped)}")
            return Verdict(VerdictClass.ASSUMED, pick[0], 0.5, chain,
                           default_rule=f"{query.mode} over mention-time stamps")
        # event-modifier guard (v1.16, OOD faithfulness — lawfirm_0014: "which
        # came first, the deposition or the FIRST RESCHEDULING of the hearing?"
        # CONFIRMED by comparing the deposition date to the hearing's SCHEDULED
        # date. The 'rescheduling' action was never a distinct dated node, so
        # the option resolved to the base hearing. If the question names a
        # modification event that NO candidate node reflects, the ordering is
        # over the wrong events -> degrade, never CONFIRM.
        _MOD = ("reschedul", "postpon", "cancel", "continuance", "amend",
                "renew", "revis", "extension", "correction", "substitut",
                "adjourn")
        qtext = (query.topic_text or "").lower()
        labels = " ".join(n.label.lower() for n, _ in dated)
        unmatched = [m for m in _MOD if m in qtext and m not in labels]
        if unmatched:
            chain.add("note", f"question names a modification event "
                              f"('{unmatched[0]}...') not represented as its own "
                              f"dated node — ordering may compare the wrong "
                              f"events; not confirmable")
            return Verdict(VerdictClass.ASSUMED, pick[0], 0.5, chain,
                           default_rule="unrepresented modification-event")
        return Verdict(VerdictClass.CONFIRMED, pick[0], 0.9, chain)

    def _count(self, query: Query, now: datetime | None) -> Verdict:
        """[Phase 6] Aggregation query: 'how many X ...?'
        Collects the full matching candidate set (no top-1) and answers with the
        count. Every counted node is listed in the chain — the count is exactly
        as trustworthy as its enumeration, so the enumeration IS the answer."""
        ph = query.placeholder
        chain = Chain()
        activated = spread(self.graph, query.anchors, now)
        members = []
        opt_set = set(query.options)
        scope = self._scope(query)
        unresolved_opts = [o for o in query.options if o not in self.graph.nodes]
        for node_id, paths in activated.items():
            node = self.graph.node(node_id)
            if opt_set and node_id not in opt_set:
                continue
            if scope is not None and node_id not in scope:
                continue
            if node.type != ph.type or node.id in query.anchors:
                continue
            if node.status in (Status.RETRACTED, Status.INFERRED, Status.ASSUMED):
                continue  # count observed facts only — never our own inferences
            m = match_constraints(node, ph.constraints, required=ph.required)
            if m == 0.0:
                continue
            # membership strength = best path's raw edge-weight product.
            # Activation decay is a RANKING device; for set membership what
            # matters is how solid the connecting facts are, not hop count.
            support = max((min((e.weight for e in p.edges), default=0.0)
                           for p in paths), default=0.0)
            if support < THETA_FLOOR:
                continue
            members.append((node, support * m))
        # lexical anchor (v1.4, gpt4_59c863d7: counted 173 "model kits"
        # including paint-cleaning advice): when the question NAMES the thing
        # being counted, a member must actually mention it — soft attr
        # matching (ATTR_UNKNOWN) alone admits nearly the whole graph
        import re as _re
        heads = []
        # v1.14.2 (bass invention: counted TROUT-lure advice for a LARGEMOUTH
        # BASS question because only 'item'-style keys fed this filter): any
        # string constraint names what is being counted. Also: rstrip("s")
        # mangled 'bass'->'ba'; singularize one trailing s only.
        for k, v in ph.constraints.items():
            if k in ("type", "kind", "category", "date", "date_range",
                     "participants", "unit"):
                continue
            for s in (v if isinstance(v, list) else [v]):
                if isinstance(s, str) and s.strip():
                    head = s.strip().split()[-1].lower()
                    heads.append(_re.escape(_re.sub(r"s$", "", head)))
        if heads:
            pat = _re.compile(r"\b(?:" + "|".join(heads) + r")(?:s|es)?\b",
                              _re.IGNORECASE)
            def _mentions(n):
                text = n.label + " " + " ".join(
                    str(v) for k, v in n.attrs.items() if k != "aliases")
                return bool(pat.search(text))
            kept = [(n, s) for n, s in members if _mentions(n)]
            if len(kept) < len(members):
                chain.add("note", f"{len(members) - len(kept)} candidate(s) "
                                  f"dropped: never mention what is being "
                                  f"counted ({', '.join(heads)[:40]})")
            members = kept
        if len(members) > 25 and not opt_set and scope is None:
            # a parsed predicate matching this much of the graph is not a
            # membership set; counting it produces confident noise (the
            # "173 model kits" class) — refuse instead
            chain.add("note", f"{len(members)} candidates match the parsed "
                              f"predicate — membership too weak to count")
            return self._unknown(query, activated, chain, ph,
                                 note="count: membership predicate too weak")
        if not members:
            return self._unknown(query, activated, chain, ph,
                                 note="count: no supported members")
        weak = [n for n, s in members if s * 1.0 < THETA_CONFIRM]
        for n, s in sorted(members, key=lambda x: -x[1]):
            chain.add("direct", f"counted: {n.label} (support {s:.2f})")
        count_node = Node(
            id=f"count_{query.sig()}", type=ph.type,
            label=str(len(members)),
            attrs={"count": len(members),
                   "members": [n.id for n, _ in members]},
            source="apm:count")
        if weak:
            chain.add("note", f"{len(weak)} member(s) below confirm threshold "
                              f"— count may be incomplete or inflated")
            return Verdict(VerdictClass.ASSUMED, count_node,
                           min(s for _, s in members), chain,
                           default_rule="count over weakly-supported members")
        import re as _re_num
        self_quantified = [n.label for n, _ in members
                           if _re_num.search(r"\d", n.label)]
        if self_quantified:
            # v1.14 (invention 21436231: counted 5 fishing assertions when
            # one of them SAID "Caught 12 largemouth bass"): members that
            # state their own quantities mean row-counting is the wrong
            # arithmetic — never CONFIRM such a count
            chain.add("note", f"counted member(s) state their own numbers "
                              f"({self_quantified[0][:40]}...) — the row "
                              f"count may not be the asked quantity")
            return Verdict(VerdictClass.ASSUMED, count_node,
                           min(s for _, s in members), chain,
                           default_rule="count over self-quantified members")
        if query.options and not unresolved_opts:
            # the asker NAMED the set to count -> membership is verified -> CONFIRMED
            return Verdict(VerdictClass.CONFIRMED, count_node,
                           min(s for _, s in members), chain)
        if unresolved_opts:
            chain.add("note", f"named items with no memory: "
                              f"{', '.join(unresolved_opts)}")
            return Verdict(VerdictClass.ASSUMED, count_node,
                           min(s for _, s in members), chain,
                           default_rule="count excludes unresolvable named items")
        if scope is not None:
            # taxonomy-scoped membership is deterministic (subtree traversal,
            # not label-vibes) — the strongest count we can make
            chain.add("note", f"membership scoped under "
                              f"'{self.graph.node(ph.under).label}' subtree")
            return Verdict(VerdictClass.CONFIRMED, count_node,
                           min(s for _, s in members), chain)
        # v0.3 rule (learned from invention 0a995998: counted 21, truth was 3):
        # membership came from PARSED constraints — an LLM artifact. A count is
        # only as trustworthy as its membership predicate, so a parsed-predicate
        # count is capped at ASSUMED and the enumeration IS the answer.
        chain.add("note", "membership predicate came from question parsing — "
                          "verify the enumerated items match what you meant")
        return Verdict(VerdictClass.ASSUMED, count_node,
                       min(s for _, s in members), chain,
                       default_rule="count over parsed membership predicate")

    def _span(self, query: Query, now: datetime | None) -> Verdict:
        """[v0.6] Date arithmetic: 'how many days between A and B?' / 'how long
        before B did A happen?' Deterministic subtraction over dated nodes —
        the graph has the dates; the engine does the math (LLMs never should)."""
        from .graph import parse_ts
        chain = Chain()
        nodes = []
        missing = []
        for o in query.options:
            if o in self.graph.nodes:
                nodes.append(self.graph.node(o))
            else:
                missing.append(o)
        if len(nodes) != 2:
            chain.add("note", f"span needs exactly 2 resolvable endpoints; "
                              f"have {len(nodes)}, unresolved: {missing}")
            return Verdict(VerdictClass.UNKNOWN, None, 0.0, chain,
                           missing_evidence="both endpoints must exist in memory "
                                            "with dates")
        if nodes[0].id == nodes[1].id:
            # both option labels resolved to the SAME memory (e831120c: two
            # phrasings of "the MCU movies") — the 0-day span that follows is
            # a resolution artifact, not knowledge
            chain.add("note", f"both endpoints resolve to the same memory "
                              f"('{nodes[0].label}') — a span needs two "
                              f"distinct events")
            return Verdict(VerdictClass.UNKNOWN, None, 0.0, chain,
                           missing_evidence="two distinct dated events")
        dates = []
        for n in nodes:
            ts = n.timestamp or n.attrs.get("date") or self._linked_date(n)
            if not ts:
                chain.add("note", f"'{n.label}' has no date/timestamp")
                return Verdict(VerdictClass.UNKNOWN, None, 0.0, chain,
                               missing_evidence=f"a date for '{n.label}'")
            dates.append(parse_ts(ts))
        days = abs((dates[1].date() - dates[0].date()).days)
        for n, d in zip(nodes, dates):
            chain.add("direct", f"{n.label} @ {d.date().isoformat()}")
        chain.add("direct", f"|{dates[1].date()} - {dates[0].date()}| = {days} days")
        answer = Node(id=f"span_{query.sig()}", type=NodeType.TIME,
                      label=f"{days} days",
                      attrs={"days": days,
                             "endpoints": [n.id for n in nodes]},
                      source="apm:span")
        # session-default stamps record when a thing was MENTIONED, not when it
        # happened; arithmetic over mention-times is a guess, not knowledge
        inferred = [n.label for n in nodes if not n.attrs.get("date")]
        if inferred:
            chain.add("note", f"date(s) inferred from mention time, not stated: "
                              f"{', '.join(inferred)}")
            return Verdict(VerdictClass.ASSUMED, answer, 0.5, chain,
                           default_rule="span over mention-time stamps")
        return Verdict(VerdictClass.CONFIRMED, answer, 0.9, chain)

    def _sum(self, query: Query, now: datetime | None) -> Verdict:
        """[v0.9] Aggregation over quantities: 'how many days did I spend on
        camping trips?' — collect matching members (like count), extract each
        member's numeric quantity (attrs or label), sum deterministically.
        The addition is arithmetic over enumerated evidence; the MEMBERSHIP
        still came from parsing, so the verdict caps at ASSUMED with every
        addend disclosed."""
        import re as _re
        ph = query.placeholder
        chain = Chain()
        activated = spread(self.graph, query.anchors, now)
        unit = str(ph.constraints.get("unit", "")).lower() or None
        # lexical anchor (v1.4, b5ef892d: summed 40 "camping days" from
        # unrelated day-quantified facts once assertions became eligible):
        # when the question names the activity, an addend must mention it
        terms = []
        for k, v in ph.constraints.items():
            if k == "unit":
                continue
            for s in (v if isinstance(v, list) else [v]):
                if isinstance(s, str) and s.strip():
                    base = s.strip().split()[-1].lower().rstrip("s")
                    if base.endswith("ing") and len(base) > 5:
                        base = base[:-3]      # camping -> camp (prefix match)
                    terms.append(_re.escape(base)[:12])
        term_pat = _re.compile(r"\b(?:" + "|".join(terms) + r")",
                               _re.IGNORECASE) if terms else None
        addends = []
        for node_id, paths in activated.items():
            node = self.graph.node(node_id)
            if node.id in query.anchors or node.status != Status.OBSERVED:
                continue
            # quantities live in ASSERTION nodes ("MCU marathon duration:
            # 2 weeks") but the parse types the unknown as `time` — the type
            # gate was silently emptying every sum (v1.4, e831120c/b5ef892d)
            if node.type != NodeType.ASSERTION and \
                    self._type_factor(node.type, ph.type) == 0.0:
                continue
            m = match_constraints(node, ph.constraints, required=ph.required)
            if m == 0.0:
                continue
            text = node.label + " " + " ".join(
                f"{k} {v}" for k, v in node.attrs.items() if k != "aliases")
            if term_pat and not term_pat.search(text):
                continue
            if unit:
                pat = _re.compile(r"(\d+(?:\.\d+)?)[\s-]*" + _re.escape(unit[:-1])
                                  + r"s?", _re.IGNORECASE)
            else:
                pat = _re.compile(r"(\d+(?:\.\d+)?)")
            nums = pat.findall(text)
            if nums:
                support = max((min((e.weight for e in p_.edges), default=0.0)
                               for p_ in paths), default=0.0)
                if support >= THETA_FLOOR:
                    addends.append((float(nums[0]), node))
        if not addends:
            return self._unknown(query, activated, chain, ph,
                                 note="sum: no quantified members found")
        total = sum(a for a, _ in addends)
        total_label = f"{total:g}" + (f" {unit}" if unit else "")
        for a, n in addends:
            chain.add("direct", f"+ {a:g}  {n.label[:60]}")
        chain.add("direct", f"total = {total_label}")
        answer = Node(id=f"sum_{query.sig()}", type=NodeType.TIME,
                      label=total_label,
                      attrs={"total": total,
                             "addends": [n.id for _, n in addends]},
                      source="apm:sum")
        chain.add("note", "membership from question parsing — verify addends")
        return Verdict(VerdictClass.ASSUMED, answer, 0.5, chain,
                       default_rule="sum over parsed membership")

    def _suggest(self, query: Query, now: datetime | None) -> Verdict:
        """[Phase 6] Preference/recommendation query: the engine does NOT answer.
        It retrieves the user's relevant CONFIRMED facts as a grounding bundle;
        the verbalizer may generate suggestions FROM that bundle, tagged as
        suggestion. Generation is permitted precisely because nothing generated
        is asserted as memory — the bundle is the only factual claim."""
        chain = Chain()
        activated = spread(self.graph, query.anchors, now)
        facts = []
        for node_id, paths in activated.items():
            node = self.graph.node(node_id)
            if node.id in query.anchors or node.status != Status.OBSERVED:
                continue
            support = noisy_or(paths)
            if support >= THETA_FLOOR:
                facts.append((node, support))
        facts.sort(key=lambda x: -x[1])
        facts = facts[:8]
        if not facts:
            return self._unknown(query, activated, chain,
                                 query.placeholder,
                                 note="no grounded facts to suggest from")
        for n, s in facts:
            chain.add("grounding", f"{n.label} (support {s:.2f})")
        bundle = Node(
            id=f"bundle_{query.sig()}", type=query.placeholder.type,
            label=f"{len(facts)} grounded facts for suggestion",
            attrs={"members": [n.id for n, _ in facts]},
            source="apm:suggest")
        return Verdict(VerdictClass.ASSUMED, bundle,
                       facts[0][1], chain,
                       default_rule="generative suggestion grounded in listed facts")

    def on_event(self, anchors: list[str], now: datetime | None = None,
                 unknown_type: NodeType = NodeType.ASSERTION) -> Verdict:
        """[Phase 5] Event-triggered pre-emptive inference: no question asked —
        an event happened (a call, a message, an arrival) and the system infers
        the most likely explanation. Always predictive, hence capped at ASSUMED."""
        q = Query(anchors=anchors,
                  placeholder=Placeholder(unknown_type, {}),
                  predictive=True)
        return self.answer(q, now=now)

    # -- helpers -------------------------------------------------------------
    _ECHO_STOP = {"user", "what", "when", "where", "which", "that", "this",
                  "did", "you", "your", "the", "and", "for", "was", "were",
                  "have", "has", "how", "many", "with", "about", "attended",
                  "visited"}

    def _echoes_question(self, label: str, question: str) -> bool:
        """True when a candidate's label is (nearly) a restatement of the
        question's own words — such a node was extracted FROM the sentence
        that poses the mystery and carries no answer. Number-bearing tokens
        are real information and exempt the candidate (value assertions
        share the question's words plus the value being asked for)."""
        import re as _re
        def toks(s):
            return {t for t in _re.findall(r"[a-z0-9:.\-]+", s.lower())
                    if (len(t) > 2 or any(ch.isdigit() for ch in t))
                    and t not in self._ECHO_STOP}
        c = toks(label)
        if len(c) < 3:
            return False           # short labels are names, not restatements
        residue = c - toks(question)
        if any(any(ch.isdigit() for ch in t) for t in residue):
            return False
        return len(residue) / len(c) <= 0.2

    # stale-snapshot guard (v1.7, slice3 inventions 031748ae/184da446:
    # CONFIRMED "5 engineers" for "when I just started" and "page 200" for
    # "how many pages SO FAR" when the state had advanced to 220): a NUMBER
    # answering a time-scoped or current-state question is a snapshot —
    # valid when recorded, not necessarily at the asked time.
    _CURRENT_MARKERS = ("so far", "currently", "as of", "to date",
                        "up to now", "right now", "at the moment")
    _SCOPE_KEYS = ("timing", "when", "period", "as_of", "time_frame")

    def _snapshot_guard(self, query: Query, node: Node, chain: Chain):
        """None = no cap; "cap" = numeric snapshot must stay ASSUMED;
        "refuse" = the question pins a time the evidence says nothing about."""
        import re as _re
        ph = query.placeholder
        has_num = bool(_re.search(r"\d", node.label)) or any(
            _re.search(r"\d", str(v)) for k, v in node.attrs.items()
            if k != "aliases")
        if not has_num:
            return None
        cur_q = query.topic_text and any(
            w in query.topic_text.lower() for w in self._CURRENT_MARKERS)
        scope_val = next((str(ph.constraints[k]) for k in self._SCOPE_KEYS
                          if ph.constraints.get(k)), None)
        if not (cur_q or scope_val):
            return None
        if scope_val:
            from .scoring import _label_satisfies
            if not (node.attrs.get("date") or _label_satisfies(node, scope_val)):
                chain.add("note", f"question is scoped to '{scope_val}' but "
                                  f"the best evidence carries no matching "
                                  f"time — a value from another moment would "
                                  f"be a confident guess")
                return "refuse"
        chain.add("note", "numeric state answered from a stored snapshot — "
                          "the value may have advanced since it was recorded")
        return "cap"

    _ADVICE = re.compile(
        r"^(recommendation|suggestion|tip|advice)\b|"
        r"\b(recommend(s|ed|ation)?|suggest(s|ed|ion)?|you should|"
        r"is considering|are considering|best guess|advisable|"
        r"you could try|option[s]?:)\b", re.I)
    _FREQ = re.compile(r"\b(?:daily|weekly|monthly|yearly|annually|"
                       r"every\s+\w+|once\s+(?:a|an)\s+\w+|"
                       r"twice\s+(?:a|an)\s+\w+|"
                       r"\d+\s+times\s+(?:a|per)\s+\w+|"
                       r"frequency)\b", re.I)
    _WANTS_ADVICE = re.compile(
        r"\b(recommend|suggest|should i|advice|any tips|tip for|"
        r"what to (get|do|buy|look|watch|bake|try)|ideas?)\b", re.I)

    def _drop_advice(self, query: Query) -> bool:
        """Exclude recommendation/hypothetical nodes from ANSWER candidates —
        BUT only when the question wants the USER's own fact, not advice. A
        'Recommendation: Lowepro...' node is not what 'which lens did I buy
        most recently?' is asking for (it beat the real answer, bucket F/C).
        Gated APM_HYGIENE=1."""
        import os as _os
        if not _os.environ.get("APM_HYGIENE"):
            return False
        q = query.topic_text or ""
        return not bool(self._WANTS_ADVICE.search(q))

    @staticmethod
    def _is_frequency_query(text: str) -> bool:
        q = text.lower()
        return ("how often" in q or "how frequently" in q
                or "weekly" in q or "monthly" in q or "daily" in q)

    @staticmethod
    def _query_subject_tokens(text: str) -> set[str]:
        toks = {t for t in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
                if len(t) > 3}
        return toks - {"often", "frequently", "weekly", "monthly", "daily",
                       "every", "currently", "about", "with",
                       "have", "been", "does", "your", "play", "plays",
                       "seeing", "visit", "visits", "meet", "meets",
                       "attend", "attends", "session", "sessions",
                       "appointment", "appointments", "friends", "friend",
                       "local", "park", "therapist", "doctor"}

    def _supports_subject_tokens(self, node: Node, tokens: set[str]) -> bool:
        if not tokens:
            return True
        text = [node.label]
        text.extend(str(v) for k, v in node.attrs.items() if k != "aliases")
        for other_id, _e in self.graph.neighbors(node.id):
            if other_id in self.graph.nodes:
                text.append(self.graph.node(other_id).label)
        hay = {w.rstrip("s") for w in re.sub(r"[^a-z0-9 ]+", " ",
                                             " ".join(text).lower()).split()
               if len(w) > 2}
        return all(any(w.startswith(t[:5]) for w in hay) for t in tokens)

    @staticmethod
    def _attribute_subject_tokens(query: Query) -> set[str]:
        attr = str(query.placeholder.constraints.get("attribute", "")).lower()
        if not attr.endswith(" name"):
            return set()
        toks = [t for t in re.sub(r"[^a-z0-9 ]+", " ", attr).split()
                if len(t) > 3 and t not in {"name"}]
        return set(toks)

    def _supports_attribute_subject(self, node: Node, tokens: set[str]) -> bool:
        if not tokens:
            return True
        text = [node.label]
        text.extend(str(v) for k, v in node.attrs.items() if k != "aliases")
        for other_id, _e in self.graph.neighbors(node.id):
            if other_id in self.graph.nodes:
                text.append(self.graph.node(other_id).label)
        hay = {w.rstrip("s") for w in re.sub(r"[^a-z0-9 ]+", " ",
                                             " ".join(text).lower()).split()
               if len(w) > 2}
        return all(any(w.startswith(t[:5]) for w in hay) for t in tokens)

    def _looks_like_frequency_assertion(self, node: Node) -> bool:
        blob = " ".join([node.label] + [str(v) for k, v in node.attrs.items()
                                        if k != "aliases"]).lower()
        return bool(self._FREQ.search(blob))

    def _frequency_event_candidates(self, activated: dict, subj_toks: set[str]) -> list[Candidate]:
        out = []
        for node_id, paths in activated.items():
            node = self.graph.node(node_id)
            if node.type != NodeType.EVENT:
                continue
            if not self._supports_subject_tokens(node, subj_toks):
                continue
            m = self._FREQ.search(node.label.lower())
            if not m:
                continue
            freq = m.group(0)
            synth = Node(id=f"freq_{node.id}", type=NodeType.ASSERTION,
                         label=freq,
                         attrs={"value": freq, "metric": "frequency",
                                "context": node.label},
                         timestamp=node.timestamp,
                         source=f"apm:freq-event:{node.id}",
                         status=Status.OBSERVED,
                         confidence=node.confidence)
            out.append(Candidate(synth, noisy_or(paths), paths, 1.0,
                                 projection_note=f"{node.label} expresses frequency '{freq}'"))
        return out

    def _is_advice(self, node: Node) -> bool:
        return bool(self._ADVICE.search(node.label))

    def _linked_date(self, node: Node):
        """A dated fact may live in a LINKED value-assertion ('2003 release
        year for X') instead of as a date attr on the node itself. For
        ordering/span, surface it: scan assertion neighbours for a date attr or
        a 4-digit year. Returns ISO string or None. (Proven by the GPT-5.5
        extractor test: the model supplied the years as assertions, but the
        operators only read date attrs ON the option node and refused.)"""
        best = None
        for o, _e in self.graph.neighbors(node.id):
            if o not in self.graph.nodes:
                continue
            nb = self.graph.node(o)
            if nb.type != NodeType.ASSERTION:
                continue
            d = nb.attrs.get("date")
            if d:
                return d
            m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b",
                          f"{nb.attrs.get('value', '')} {nb.label}")
            if m and best is None:
                best = f"{m.group(1)}-01-01"
        return best

    @staticmethod
    def _same_value(a: Node, b: Node) -> bool:
        """Do two candidate nodes express the SAME value, or a genuine change?
        Compared on DISTINCTIVE tokens (generic stopwords removed, numbers
        kept) so a re-mention reads as same while a real update reads as
        different: '...address is 118 Maple Court' vs '...address updated to
        4471 Birchwood Lane' -> different (low token overlap). Domain-agnostic:
        no domain vocabulary, so it generalizes past the address case."""
        import re as _re
        STOP = {"the", "a", "an", "is", "are", "was", "were", "for", "to", "of",
                "on", "in", "at", "my", "your", "his", "her", "their", "and",
                "now", "then", "this", "that", "it", "as", "by", "with", "from"}
        def toks(n):
            ws = _re.sub(r"[^a-z0-9 ]+", " ", n.label.lower()).split()
            return {w for w in ws if len(w) > 2 and w not in STOP}
        ta, tb = toks(a), toks(b)
        if not ta or not tb:
            return a.label.strip().lower() == b.label.strip().lower()
        return len(ta & tb) / len(ta | tb) >= 0.5   # Jaccard on distinctive tokens

    def _is_superseded(self, cand) -> bool:
        """True when the candidate is the OLD value of a knowledge update:
        reached via an auto-superseded edge, or attacked by a NEWER
        contradicts edge — i.e. evidence exists that it was replaced."""
        from .graph import parse_ts
        superseded = getattr(self.graph, "superseded", set())
        for p in cand.paths:
            for e in p.edges:
                if (e.src, e.rel, e.dst) in superseded:
                    return True
        for other_id, edge in self.graph.neighbors(cand.node.id):
            if edge.rel != "contradicts":
                continue
            attacker = self.graph.node(other_id)
            if (edge.src == other_id and attacker.status != Status.RETRACTED
                    and parse_ts(attacker.timestamp)
                    > parse_ts(cand.node.timestamp)):
                return True
        return False

    def _apply_contradictions(self, cands: list[Candidate]) -> list[str]:
        """Temporal precedence rule: a newer node connected by a `contradicts`
        edge suppresses an older candidate; the older never attacks the newer.
        Penalty: score *= (1 - attacker.confidence * edge.weight)."""
        from .graph import parse_ts
        notes = []
        for c in cands:
            for other_id, edge in self.graph.neighbors(c.node.id):
                if edge.rel != "contradicts":
                    continue
                attacker = self.graph.node(other_id)
                if attacker.status == Status.RETRACTED:
                    continue
                if parse_ts(attacker.timestamp) <= parse_ts(c.node.timestamp):
                    continue
                penalty = 1.0 - attacker.confidence * edge.weight
                c.score *= penalty
                notes.append(f"{c.node.label} contradicted by newer "
                             f"'{attacker.label}' ({attacker.timestamp}) "
                             f"-> score x{penalty:.2f}")
        return notes

    def _rel_corroborated(self, query: Query, cand) -> bool:
        """The question's own verb appearing in the candidate's connecting
        edge relations ('playlist I CREATED' via a created edge) is positive
        evidence that overrides false-premise doubt — the unresolved entity
        is then likely unextracted context, not a false premise."""
        if not query.topic_text:
            return False
        q_low = query.topic_text.lower()
        rels = {e.rel for p_ in cand.paths for e in p_.edges}
        return any(tok in q_low for r in rels
                   for tok in r.lower().replace("_", " ").split()
                   if len(tok) > 3)

    @staticmethod
    def _supports_unresolved_entity(unresolved: list[str], node: Node) -> bool:
        import re as _re
        hay = _re.sub(r"[^a-z0-9 ]+", " ",
                      (node.label + " " + " ".join(
                          str(v) for k, v in node.attrs.items()
                          if k != "aliases")).lower())
        words = {w.rstrip("s") for w in hay.split() if len(w) > 2}
        for ref in unresolved:
            toks = [t.rstrip("s") for t in _re.sub(r"[^a-z0-9 ]+", " ",
                                                   ref.lower()).split()
                    if len(t) > 2]
            if toks and all(any(w.startswith(t[:5]) for w in words) for t in toks):
                return True
        return False

    def _describe(self, chain: Chain, cand: Candidate):
        for p in cand.paths:
            chain.add("anchor-path", path_text(self.graph, p))
        if cand.projection_note:
            chain.add("projection", cand.projection_note)

    def _gate(self, query: Query, cand: Candidate, chain: Chain,
              now: datetime | None, write_back: bool,
              force_class: VerdictClass | None = None,
              default_rule: str | None = None) -> Verdict:
        klass = force_class or VerdictClass.CONFIRMED
        note = None
        if query.predictive and klass == VerdictClass.CONFIRMED:
            klass = VerdictClass.ASSUMED  # intents/futures can never be CONFIRMED
            note = "predictive query: capped at ASSUMED"
        if klass == VerdictClass.CONFIRMED and any(
                e.source.endswith(":implied")
                for p in cand.paths for e in p.edges):
            # v1.8: an extractor-INFERRED association in the chain is a
            # tagged guess, not evidence — the whole chain inherits that
            klass = VerdictClass.ASSUMED
            default_rule = default_rule or "chain uses an implied association"
            note = "an edge in this chain was inferred from context, not stated"
        if klass == VerdictClass.CONFIRMED and cand.node.status == Status.ASSUMED:
            # a stored guess (our own write-back) is not evidence
            klass = VerdictClass.ASSUMED
            default_rule = default_rule or "answer is itself an assumption"
            note = "the supporting node is a stored assumption, not an observation"
        if klass == VerdictClass.CONFIRMED and (
                cand.node.source.startswith("cued:") or any(
                    e.source.startswith("cued:")
                    for p in cand.paths for e in p.edges)):
            # v1.9: cued-recall evidence was extracted BY A MODEL THAT SAW
            # THE QUESTION — its nodes echo the question's own constraints
            # (live invention: 'Cartwheel app' CONFIRMED via direct lookup
            # on attrs the cued extractor stamped). Question-conditioned
            # evidence may answer, never confirm.
            klass = VerdictClass.ASSUMED
            default_rule = default_rule or "cued-recall evidence"
            note = ("answer rests on a question-conditioned re-read of the "
                    "source — delivered as a guess, not a certainty")
        if klass == VerdictClass.CONFIRMED and \
                cand.node.source.startswith("apm:freq-event:"):
            klass = VerdictClass.ASSUMED
            default_rule = default_rule or "frequency inferred from event label"
            note = ("the answer is read from a frequency-bearing event label, "
                    "not a dedicated observed value assertion")
        verdict = Verdict(klass, cand.node, min(cand.score, 0.99), chain,
                          default_rule=default_rule, note=note,
                          evidence_edges=[(e.src, e.dst, e.rel)
                                          for p in cand.paths for e in p.edges])
        if write_back and cand.paths:  # direct hits need no write-back
            self._write_back(query, verdict, now, paths=cand.paths)
        return verdict

    def _write_back(self, query: Query, verdict: Verdict, now: datetime | None,
                    paths: list = ()):
        """[10] Store the inference as a node so future queries hit it directly,
        and reinforce the edges that carried it (Hebbian — recall strengthens)."""
        chain_edges = [e for p in paths for e in p.edges]
        if chain_edges:
            self.graph.reinforce(chain_edges, now=now)
        status = (Status.INFERRED if verdict.klass == VerdictClass.CONFIRMED
                  else Status.ASSUMED)
        ph = query.placeholder
        inferred = Node(
            id=f"inf_{query.sig()}",
            type=ph.type,
            label=verdict.answer.label,
            attrs={**verdict.answer.attrs, **ph.constraints},
            timestamp=now.isoformat() if now else None,
            source=f"apm:chain:{verdict.answer.id}",
            status=status,
            confidence=verdict.score,
            resolves=query.sig(),
        )
        if inferred.id not in self.graph.nodes:
            self.graph.add_node(inferred)

    def _stated_value_fallback(self, query: Query, activated: dict,
                               chain: Chain, ph: Placeholder):
        """Reachability fallback (APM_REACH) — bucket B of the failure table:
        the answer is a STATED value living in an ASSERTION node that the
        type/mode filter excluded ('Fitbit usage duration: 9 months', 'Caught
        12 largemouth bass'). Fires ONLY on the UNKNOWN path, so it can never
        displace an existing answer, and returns ASSUMED, so it can never make
        an invention. Reads the value from the best-supported, non-recommendation,
        non-superseded assertion that is topically relevant to the question."""
        import re
        q = (query.topic_text or "").lower()
        if query.unresolved:
            return None
        want_dur = ("how long" in q or ph.type == NodeType.TIME
                    or ph.constraints.get("unit") == "duration")
        want_cnt = (query.mode == "count" or "how many" in q or "how much" in q)
        want_freq = ("how often" in q or "how frequently" in q
                     or "weekly" in q or "monthly" in q or "daily" in q)
        if not (want_dur or want_cnt or want_freq):
            return None
        # do NOT fire on a TIME-SCOPED / snapshot question ('how many pages had
        # I read WHEN I started my job'): the value must be pinned to a past
        # moment the evidence may not cover — reading the current value would
        # convert a correct refusal into a guess (test_v14 031748ae). Defer to
        # the snapshot guard / honest UNKNOWN.
        if any(k in ph.constraints for k in ("timing", "when", "as_of", "at_time")):
            return None
        if re.search(r"\b(when i |had i |at the time|by the time|back when)\b", q):
            return None
        from .graph import parse_ts
        DUR = re.compile(r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+)?\s*"
                         r"(?:hour|day|week|month|year|minute)s?\b", re.I)
        NUM = re.compile(r"\d")
        FREQ = re.compile(r"\b(?:daily|weekly|monthly|yearly|annually|"
                          r"every\s+\w+|once\s+(?:a|an)\s+\w+|"
                          r"twice\s+(?:a|an)\s+\w+|"
                          r"\d+\s+times\s+(?:a|per)\s+\w+)\b", re.I)
        RECO = re.compile(r"^(recommendation|suggest|tip|advice|use |choose |try )", re.I)
        subj = {t for t in re.sub(r"[^a-z0-9 ]", " ", q).split()
                if len(t) > 3 and t not in ("long", "many", "much", "have",
                                            "been", "does", "your", "with",
                                            "often", "frequently")}
        cands = []
        for nid, paths in activated.items():
            n = self.graph.node(nid)
            if n.type != NodeType.ASSERTION or n.status == Status.RETRACTED:
                continue
            if RECO.search(n.label):
                continue
            if want_dur and not DUR.search(n.label):
                continue
            if want_freq and not FREQ.search(n.label):
                continue
            if want_cnt and not want_dur and not NUM.search(n.label):
                continue
            if subj and not any(t in n.label.lower() for t in subj):
                continue
            support = noisy_or(paths)
            if support < THETA_FLOOR:
                continue
            c = Candidate(n, support, paths, 1.0)
            if self._is_superseded(c):     # drop the stale value of an update
                continue
            cands.append(c)
        if not cands:
            return None
        # most recent (current value of a knowledge update), then strongest
        cands.sort(key=lambda c: (parse_ts(c.node.timestamp), c.score), reverse=True)
        pick = cands[0]
        self._describe(chain, pick)
        chain.add("note", "value read from a supporting assertion the type/mode "
                          "filter had excluded (reachability fallback)")
        return self._gate(query, pick, chain, None, False,
                          force_class=VerdictClass.ASSUMED,
                          default_rule="stated-value reachability fallback")

    def _unknown(self, query: Query, activated: dict, chain: Chain,
                 ph: Placeholder, note: str | None = None) -> Verdict:
        import os as _os
        if not _os.environ.get("APM_NO_REACH"):   # bucket-B fix, default ON
            fb = self._stated_value_fallback(query, activated, chain, ph)
            if fb is not None:
                return fb
        # report the best-supported hub so the caller knows what WAS established
        hub, hub_score = None, 0.0
        for node_id, paths in activated.items():
            s = noisy_or(paths)
            if s > hub_score and len(paths) > 1:
                hub, hub_score = self.graph.node(node_id), s
        if hub is not None:
            chain.add("note", f"established: anchors converge on {hub.label} "
                              f"(score {hub_score:.2f}) — but no {ph.type.value} "
                              f"candidate satisfies the constraints")
        missing = (f"a {ph.type.value} node matching {ph.constraints} "
                   f"(e.g. a calendar entry or an agenda item) would resolve this")
        return Verdict(VerdictClass.UNKNOWN, None, 0.0, chain,
                       missing_evidence=missing, note=note)
