"""Phase 1 milestone tests (SPEC.md §8).

The paradigm is proven when: all four scenarios answer with evidence chains,
S4 comes out ASSUMED (not CONFIRMED), and deleting the agenda node flips S1
from CONFIRMED to UNKNOWN.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apm import Engine, Node, NodeType, Status, VerdictClass
from apm.graph import Edge
from scenarios.fixtures import NOW, s1_iaeste, s2_movie, s3_call, s4_travel


# ---- S1: the fallback ladder -------------------------------------------------
def test_s1_confirmed_via_parent_projection():
    g, q = s1_iaeste()
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.id == "ev_ga"
    assert v.score >= 0.6
    text = v.chain.render()
    assert "IAESTE" in text                      # the parent hub is in the chain
    assert any(s.kind == "projection" for s in v.chain.steps)


def test_s1_flips_to_unknown_without_agenda():
    """THE milestone test: remove the evidence -> the answer must disappear."""
    g, q = s1_iaeste(with_agenda=False)
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN
    assert v.answer is None
    assert v.missing_evidence is not None
    # it still reports what it COULD establish: the IAESTE convergence
    assert "IAESTE" in v.chain.render()


def test_s1_direct_lookup_wins_when_calendar_exists():
    g, q = s1_iaeste(with_calendar=True)
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.id == "ev_cal"
    assert any(s.kind == "direct" for s in v.chain.steps)


def test_s1_write_back_makes_second_query_direct():
    g, q = s1_iaeste()
    eng = Engine(g)
    v1 = eng.answer(q, now=NOW)
    assert v1.klass == VerdictClass.CONFIRMED
    v2 = eng.answer(q, now=NOW)
    assert v2.klass == VerdictClass.CONFIRMED
    assert any(s.kind == "direct" for s in v2.chain.steps)  # inference was stored


# ---- S2: intersection search -------------------------------------------------
def test_s2_three_cues_confirm_the_movie():
    g, q = s2_movie()
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED
    assert v.answer.id == "m_target"
    assert v.score >= 0.6
    # at least two independent anchor paths in the chain
    assert sum(1 for s in v.chain.steps if s.kind == "anchor-path") >= 2


def test_s2_single_cue_cannot_confirm():
    g, q = s2_movie(all_cues=False)
    v = Engine(g).answer(q, now=NOW)
    assert v.klass != VerdictClass.CONFIRMED   # one cue, three tied movies
    if v.klass == VerdictClass.ASSUMED:
        assert v.default_rule == "recency"
        assert v.answer.id == "m_other2"       # the most recent movie


# ---- S3: pre-emptive, event-triggered inference -------------------------------
def test_s3_call_intent_is_assumed_never_confirmed():
    g, q = s3_call()
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED     # intents can never be CONFIRMED
    assert v.answer.id == "a_docs"             # beats the stale-photos distractor
    assert "Org" in v.chain.render() or "org" in v.chain.render()


def test_s3_unreachable_assertion_never_surfaces():
    g, q = s3_call()
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id != "a_invoice"          # Person X is not in Org Z


# ---- S4: corrupted signal -> tagged default -----------------------------------
def test_s4_recency_default_is_assumed():
    g, q = s4_travel()
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.ASSUMED
    assert v.answer.id == "pl_china"           # most recent trip
    assert v.default_rule == "recency"


def test_s4_assumed_is_retracted_by_observed_fact():
    g, q = s4_travel()
    eng = Engine(g)
    v = eng.answer(q, now=NOW)
    assumed_id = f"inf_{q.sig()}"
    assert g.node(assumed_id).status == Status.ASSUMED
    # later, an observed fact answers the same question -> the guess retracts
    g.add_observation(
        Node("pl_greece_trip", NodeType.PLACE, "Greece (confirmed trip)",
             timestamp="2026-06-21T08:00:00", source="ticket_email",
             resolves=q.sig()),
        edges=[Edge("p_x", "pl_greece_trip", "traveled_to", 0.95)])
    assert g.node(assumed_id).status == Status.RETRACTED


# ---- the gate's core invariant -------------------------------------------------
def test_no_bare_guess_is_ever_confirmed():
    """Across all four scenarios, every CONFIRMED verdict must carry a chain
    with at least one direct/anchor-path/projection step."""
    runs = [s1_iaeste(), s1_iaeste(with_calendar=True), s2_movie(),
            s3_call(), s4_travel()]
    for g, q in runs:
        v = Engine(g).answer(q, now=NOW)
        if v.klass == VerdictClass.CONFIRMED:
            kinds = {s.kind for s in v.chain.steps}
            assert kinds & {"direct", "anchor-path", "projection"}, \
                f"CONFIRMED without evidence steps: {v.render()}"
