"""v1.4 engine fixes — each derived from a named v1.3 miss (autopsied on the
exact bench graphs via build_bench_graph):

  58bf7951      question-echo cap (engine CONFIRMED a restatement of the question)
  e831120c      span endpoint dedupe + sum over duration assertions (MCU 3.5 weeks)
  gpt4_59c863d7 count lexical anchor + oversize guard ("173 model kits")
  852ce960      contradiction propagation across value-duplicate assertions
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import (Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder,
                 Query, VerdictClass)

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


# ------------------------------------------------------- question-echo cap --

def _theater_world():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_echo", NodeType.EVENT, "play at community theater",
                    timestamp="2026-06-01T10:00:00"))
    g.add_edge(Edge("p_u", "ev_echo", "attends", 0.9))
    # second anchor (the parse anchors User AND the theater — the bench echo
    # scored 0.79 via exactly this two-anchor convergence)
    g.add_node(Node("pl_th", NodeType.PLACE, "local community theater"))
    g.add_edge(Edge("ev_echo", "pl_th", "located_in", 0.9))
    return g


def test_echo_candidate_cannot_confirm():
    g = _theater_world()
    q = Query(anchors=["p_u", "pl_th"],
              placeholder=Placeholder(NodeType.EVENT, {}),
              topic_text="What play did I attend at the local community theater?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.CONFIRMED
    assert any("restates the question" in s.text for s in v.chain.steps)


def test_informative_candidate_still_confirms():
    g = _theater_world()
    g.add_node(Node("ev_gm", NodeType.EVENT, "The Glass Menagerie performance",
                    timestamp="2026-06-01T10:00:00"))
    g.add_edge(Edge("p_u", "ev_gm", "attends", 0.95))
    g.add_edge(Edge("ev_gm", "pl_th", "located_in", 0.9))
    g.remove_node("ev_echo")
    q = Query(anchors=["p_u", "pl_th"],
              placeholder=Placeholder(NodeType.EVENT, {}),
              topic_text="What play did I attend at the local community theater?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert "Glass Menagerie" in v.answer.label


def test_number_bearing_labels_are_not_echoes():
    """Value assertions share the question's words PLUS the value asked for —
    the digits are the answer, never an echo."""
    eng = Engine(EvidenceGraph())
    assert not eng._echoes_question(
        "Personal best 25:50 in charity 5K",
        "What was my personal best time in the charity 5K?")
    assert eng._echoes_question(
        "play at community theater",
        "What play did I attend at the local community theater?")


# ------------------------------------------------ span dedupe + sum (MCU) --

def test_span_over_identical_endpoints_refuses():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("to_mcu", NodeType.TOPIC, "MCU movies",
                    timestamp="2026-01-01T00:00:00"))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.TIME, {}),
              mode="span", options=["to_mcu", "to_mcu"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_sum_totals_duration_assertions():
    """The MCU shape: durations stored as assertion nodes; unknown_type from
    the parse is `time`. 2 + 1.5 = 3.5 weeks, ASSUMED with addends in chain."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_mcu", NodeType.ASSERTION,
                    "MCU marathon duration: 2 weeks",
                    timestamp="2026-05-01T10:00:00"))
    g.add_node(Node("as_sw", NodeType.ASSERTION,
                    "Duration of Star Wars marathon: 1.5 weeks",
                    timestamp="2026-05-20T10:00:00"))
    g.add_edge(Edge("p_u", "as_mcu", "about", 0.9))
    g.add_edge(Edge("p_u", "as_sw", "about", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TIME, {"unit": "weeks"}),
              mode="sum",
              topic_text="How many weeks did it take me to watch them all?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.attrs["total"] == 3.5


# ------------------------------------- frequency reachability fallback ----

def test_frequency_reachability_reads_stated_value():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_dr", NodeType.ASSERTION,
                    "I see Dr. Smith every week",
                    timestamp="2026-05-20T10:00:00"))
    g.add_edge(Edge("p_u", "as_dr", "about", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION, {}),
              topic_text="How often do I see my therapist, Dr. Smith?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert "every week" in v.answer.label.lower()


def test_frequency_event_can_supply_answer_shape():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("p_dr", NodeType.PERSON, "Dr. Smith"))
    g.add_node(Node("ev_weekly", NodeType.EVENT,
                    "Weekly sessions with Dr. Smith",
                    timestamp="2026-05-20T10:00:00"))
    g.add_edge(Edge("p_u", "ev_weekly", "attends", 0.9))
    g.add_edge(Edge("p_dr", "ev_weekly", "about", 0.9))
    q = Query(anchors=["p_u", "p_dr"],
              placeholder=Placeholder(NodeType.ASSERTION, {}),
              topic_text="How often do I see my therapist, Dr. Smith?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert "weekly" in v.answer.label.lower()


def test_frequency_reachability_refuses_on_unresolved_entity():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_dr", NodeType.ASSERTION,
                    "I see Dr. Smith every week",
                    timestamp="2026-05-20T10:00:00"))
    g.add_edge(Edge("p_u", "as_dr", "about", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION, {}),
              topic_text="How often do I see Dr. Johnson?",
              unresolved=["Dr. Johnson"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_frequency_event_requires_specific_subject_tokens():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("pl_park", NodeType.PLACE, "local park"))
    g.add_node(Node("ev_tennis", NodeType.EVENT,
                    "weekly tennis sessions",
                    timestamp="2026-05-20T10:00:00"))
    g.add_edge(Edge("p_u", "ev_tennis", "attends", 0.9))
    g.add_edge(Edge("ev_tennis", "pl_park", "located_in", 0.9))
    q = Query(anchors=["p_u", "pl_park"],
              placeholder=Placeholder(NodeType.ASSERTION, {}),
              topic_text="How often do I play table tennis at the local park?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_attribute_name_subject_mismatch_refuses():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_microchip", NodeType.ASSERTION,
                    "Recommendation: Microchip Luna",
                    timestamp="2026-05-20T10:00:00"))
    g.add_node(Node("to_cat", NodeType.TOPIC, "cat"))
    g.add_edge(Edge("p_u", "as_microchip", "related_to", 0.9))
    g.add_edge(Edge("as_microchip", "to_cat", "about", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"attribute": "hamster name"}),
              topic_text="What is the name of my hamster?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN




# ------------------------------------------- count anchor + oversize guard --

def _kit_world(n_noise=0):
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    for i, label in enumerate(["Revell 1:48 B-29 model kit",
                               "Tamiya Spitfire model kit",
                               "Bandai Star Destroyer kit"]):
        nid = f"as_kit{i}"
        g.add_node(Node(nid, NodeType.ASSERTION, label,
                        timestamp=f"2026-0{i+1}-01T10:00:00"))
        g.add_edge(Edge("p_u", nid, "owns", 0.9))
    for i in range(n_noise):
        nid = f"as_noise{i}"
        g.add_node(Node(nid, NodeType.ASSERTION,
                        f"Use metallic paint number {i} for weathering",
                        timestamp="2026-04-01T10:00:00"))
        g.add_edge(Edge("p_u", nid, "about", 0.9))
    return g


def test_count_lexical_anchor_drops_nonmentions():
    g = _kit_world(n_noise=5)
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"item": "model kits"}),
              mode="count", topic_text="How many model kits have I bought?")
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.attrs["count"] == 3          # noise never mentions "kit"


def test_count_refuses_when_predicate_matches_everything():
    g = _kit_world(n_noise=30)
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"category": "hobby"}),   # no item key
              mode="count", topic_text="How many hobby things do I have?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN       # 33 members > 25: not a set


def test_sum_lexical_anchor_drops_unrelated_quantities():
    """The camping shape (b5ef892d): once assertions became summable, noise
    like '40 days of gym membership' inflated the total. An addend must
    mention the named activity."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_c1", NodeType.ASSERTION, "3 days camping at Yosemite",
                    timestamp="2026-05-01T10:00:00"))
    g.add_node(Node("as_c2", NodeType.ASSERTION, "5-day camping trip in Utah",
                    timestamp="2026-06-01T10:00:00"))
    g.add_node(Node("as_gym", NodeType.ASSERTION, "40 days of gym membership left",
                    timestamp="2026-06-10T10:00:00"))
    for a in ("as_c1", "as_c2", "as_gym"):
        g.add_edge(Edge("p_u", a, "about", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.TIME,
                                      {"unit": "days", "activity": "camping"}),
              mode="sum", topic_text="How many days did I spend camping?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.attrs["total"] == 8.0        # gym noise excluded


def test_ordering_over_mention_stamps_cannot_confirm():
    """dev2 invention gpt4_0b2f1d21: ordering CONFIRMED off session
    mention-times. Stated dates (attrs.date) may CONFIRM; bare timestamps
    order the pick but cap at ASSUMED."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_a", NodeType.EVENT, "coffee maker purchase",
                    timestamp="2026-03-01T09:00:01"))          # mention-stamp
    g.add_node(Node("ev_b", NodeType.EVENT, "mattress purchase",
                    timestamp="2026-03-01T09:00:02"))          # mention-stamp
    g.add_edge(Edge("p_u", "ev_a", "owns", 0.9))
    g.add_edge(Edge("p_u", "ev_b", "owns", 0.9))
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.EVENT, {}),
              mode="first", options=["ev_a", "ev_b"])
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED          # never CONFIRMED
    # stated dates still CONFIRM
    g.node("ev_a").attrs["date"] = "2026-02-10"
    g.node("ev_b").attrs["date"] = "2026-02-20"
    v2 = Engine(g).answer(q, now=NOW)
    assert v2.klass == VerdictClass.CONFIRMED
    assert v2.answer.id == "ev_a"


def test_past_value_question_prefers_superseded():
    """dev2 invention 07741c44: 'where do I INITIALLY keep my sneakers' must
    answer the SUPERSEDED location, not confirm the current one."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("pl_bed", NodeType.PLACE, "under the bed"))
    g.add_node(Node("pl_clo", NodeType.PLACE, "closet"))
    old = Edge("p_u", "pl_bed", "lives_in", 0.9, timestamp="2026-01-01T00:00:00")
    new = Edge("p_u", "pl_clo", "lives_in", 0.9, timestamp="2026-06-01T00:00:00")
    g.add_edge(old); g.add_edge(new)
    g.auto_supersede()
    q = Query(anchors=["p_u"], placeholder=Placeholder(NodeType.PLACE, {}),
              topic_text="Where did I initially keep my old sneakers?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.id == "pl_bed"              # the superseded value

    # and with no supersede evidence at all, the current value cannot CONFIRM
    g2 = EvidenceGraph()
    g2.add_node(Node("p_u", NodeType.PERSON, "User"))
    g2.add_node(Node("pl_clo", NodeType.PLACE, "closet"))
    g2.add_node(Node("as_s", NodeType.ASSERTION, "Sneakers stored in closet",
                     timestamp="2026-06-01T00:00:00"))
    g2.add_edge(Edge("p_u", "as_s", "about", 0.9))
    g2.add_edge(Edge("as_s", "pl_clo", "about", 0.9))
    g2.add_edge(Edge("p_u", "pl_clo", "related_to", 0.9))
    v2 = Engine(g2).answer(q, now=NOW)
    assert v2.klass != VerdictClass.CONFIRMED   # guard holds


def _snapshot_world():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_p", NodeType.ASSERTION,
                    "Reading progress: page 200 of the book",
                    attrs={"value": "200", "unit": "pages"},
                    timestamp="2026-05-01T10:00:00"))
    g.add_node(Node("to_b", NodeType.TOPIC, "the book"))
    g.add_edge(Edge("p_u", "as_p", "about", 0.9))
    g.add_edge(Edge("as_p", "to_b", "about", 0.9))
    return g


def test_stale_snapshot_cannot_confirm_current_state():
    """slice3 invention 184da446: 'how many pages SO FAR' answered from an
    old numeric snapshot must be ASSUMED, never CONFIRMED."""
    g = _snapshot_world()
    q = Query(anchors=["p_u", "to_b"],
              placeholder=Placeholder(NodeType.ASSERTION, {"unit": "pages"}),
              topic_text="How many pages of the book have I read so far?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.default_rule == "stale-snapshot cap"


def test_unverifiable_time_scope_refuses():
    """slice3 invention 031748ae: a question pinned to a TIME the evidence
    says nothing about ('when I just started') must refuse, not confirm the
    current value."""
    g = _snapshot_world()
    q = Query(anchors=["p_u", "to_b"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"unit": "pages", "timing": "start"}),
              topic_text="How many pages had I read when I started my job?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN


def test_implied_edge_cannot_confirm_but_enables_assumed():
    """v1.8: an extractor-inferred edge (source ':implied') may power a
    hedged answer but never a CONFIRMED — even when the score is high."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_r", NodeType.EVENT, "Redeemed $5 coupon on creamer",
                    timestamp="2026-05-01T10:00:00"))
    g.add_node(Node("or_t", NodeType.ORG, "Target"))
    g.add_edge(Edge("p_u", "ev_r", "attends", 0.9))
    g.add_edge(Edge("ev_r", "or_t", "located_in", 0.55,
                    source="llm:ingest:implied"))
    g.add_edge(Edge("p_u", "or_t", "related_to", 0.9))
    q = Query(anchors=["p_u", "ev_r"],
              placeholder=Placeholder(NodeType.ORG, {}),
              topic_text="Where did I redeem the coupon?")
    v = Engine(g).answer(q, now=NOW)
    assert v.answer is not None and v.answer.label == "Target"
    assert v.klass == VerdictClass.ASSUMED          # never CONFIRMED
    # the same shape with a STATED edge confirms normally
    g2 = EvidenceGraph()
    g2.add_node(Node("p_u", NodeType.PERSON, "User"))
    g2.add_node(Node("ev_r", NodeType.EVENT, "Redeemed $5 coupon on creamer",
                     timestamp="2026-05-01T10:00:00"))
    g2.add_node(Node("or_t", NodeType.ORG, "Target"))
    g2.add_edge(Edge("p_u", "ev_r", "attends", 0.9))
    g2.add_edge(Edge("ev_r", "or_t", "located_in", 0.9))
    g2.add_edge(Edge("p_u", "or_t", "related_to", 0.9))
    v2 = Engine(g2).answer(Query(anchors=["p_u", "ev_r"],
                                 placeholder=Placeholder(NodeType.ORG, {}),
                                 topic_text="Where did I redeem the coupon?"),
                           now=NOW)
    assert v2.klass == VerdictClass.CONFIRMED


def test_unsatisfied_constraint_cannot_confirm():
    """v1.14 (validation inventions): a strongly-connected candidate that
    soft-misses a question condition ('favorite', an occasion, a species)
    caps at ASSUMED — CONFIRMED demands every asked condition evidenced."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("to_h", NodeType.TOPIC, "Hoka Bondi 7 running shoes"))
    g.add_node(Node("pl_s", NodeType.PLACE, "running store"))
    g.add_edge(Edge("p_u", "to_h", "owns", 0.9))
    g.add_edge(Edge("to_h", "pl_s", "located_in", 0.9))
    g.add_edge(Edge("p_u", "pl_s", "related_to", 0.9))
    q = Query(anchors=["p_u", "pl_s"],
              placeholder=Placeholder(NodeType.TOPIC,
                                      {"attribute": "favorite"}),
              topic_text="What brand are my favorite running shoes?")
    v = Engine(g).answer(q, now=NOW)
    if v.answer is not None and "Hoka" in v.answer.label:
        assert v.klass != VerdictClass.CONFIRMED
    # with the constraint evidenced, CONFIRMED is allowed again
    g.node("to_h").attrs["attribute"] = "favorite"
    v2 = Engine(g).answer(q, now=NOW)
    assert v2.klass == VerdictClass.CONFIRMED


def test_self_quantified_count_cannot_confirm():
    """v1.14 (invention 21436231: counted 5 assertions when one SAID
    'Caught 12 largemouth bass'): members stating their own numbers make
    row-counting unreliable — the count caps at ASSUMED."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("pl_l", NodeType.PLACE, "Lake Michigan"))
    labels = ["Caught 12 largemouth bass on last trip",
              "Wants hooks for largemouth bass fishing"]
    for i, lab in enumerate(labels):
        nid = f"as_f{i}"
        g.add_node(Node(nid, NodeType.ASSERTION, lab,
                        timestamp=f"2026-0{i+1}-01T10:00:00"))
        g.add_edge(Edge("p_u", nid, "about", 0.9))
        g.add_edge(Edge(nid, "pl_l", "located_in", 0.9))
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.ASSERTION,
                                      {"species": "largemouth bass"},
                                      under="pl_l"),
              mode="count",
              topic_text="How many largemouth bass did I catch?")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.CONFIRMED


def test_scope_resolution_rejects_distinguishing_token_mismatch(monkeypatch):
    """v1.14 (invention bc8a6e93: 'UNCLE's birthday party' resolved onto the
    NIECE's party): a scope label with a content token the node lacks is
    unresolved — the premise machinery must see it."""
    import apm.ingest as ing
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("ev_np", NodeType.EVENT, "niece's birthday party"))
    g.add_edge(Edge("p_u", "ev_np", "attends", 0.9))
    monkeypatch.setattr(ing, "_ask_json", lambda *a, **k: {
        "anchors": ["User"], "unknown_type": "assertion", "constraints": {},
        "predictive": False, "mode": "resolve", "options": [],
        "under": "uncle's birthday party"})
    q = ing.parse_question(g, "What did I bake for my uncle's birthday party?")
    assert q.placeholder.under is None
    assert any("uncle" in u for u in q.unresolved)


# --------------------------------------- contradiction value-duplication --

def test_contradiction_propagates_across_value_duplicates():
    """The mortgage shape: the $400k update kills ONE $350k phrasing; the
    duplicate phrasing must not survive at full strength."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("as_new", NodeType.ASSERTION, "Pre-approved for $400,000",
                    attrs={"value": "400000", "metric": "mortgage_pre_approval"},
                    timestamp="2026-06-01T10:00:00"))
    g.add_node(Node("as_old1", NodeType.ASSERTION, "Pre-approved amount: $350,000",
                    attrs={"value": "350000", "metric": "mortgage_pre_approval"},
                    timestamp="2026-01-01T10:00:00"))
    g.add_node(Node("as_old2", NodeType.ASSERTION, "Pre-approval of $350,000",
                    attrs={"value": "350000", "metric": "currency"},   # gemma-style
                    timestamp="2026-01-01T10:00:01"))
    for a in ("as_new", "as_old1", "as_old2"):
        g.add_edge(Edge("p_u", a, "about", 0.9))
    g.auto_contradict()
    contra = {(e.src, e.dst) for _, _, d in g.g.edges(data=True)
              for e in [d["obj"]] if e.rel == "contradicts"}
    assert ("as_new", "as_old1") in contra
    assert ("as_new", "as_old2") in contra       # the propagated edge
