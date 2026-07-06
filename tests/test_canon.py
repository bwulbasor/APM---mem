"""Extractor-agnostic canonicalization (apm/canon.py).

The v1.2 Cerebras regression: gemma emits the same facts with different
relation verbs than gemini, engine rules keyed on literal strings no-op,
the engine refuses answers it HAS. These tests prove that gemma-style
vocabulary, canonicalized, makes the exact rules that regressed fire again.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apm import Edge, EvidenceGraph, Node, NodeType
from apm.canon import canon_rel, canonicalize
from apm.taxonomy import subtree


# ------------------------------------------------------------ canon_rel ----

def test_synonyms_map_to_canonical():
    assert canon_rel("resides_in") == "lives_in"
    assert canon_rel("relocated_to") == "lives_in"
    assert canon_rel("employed_by") == "works_at"
    assert canon_rel("works_for") == "works_at"
    assert canon_rel("belongs_to") == "member_of"
    assert canon_rel("affiliated_with") == "member_of"
    assert canon_rel("participated_in") == "attends"
    assert canon_rel("situated_in") == "located_in"
    assert canon_rel("required_for") == "needed_for"
    assert canon_rel("regarding") == "about"
    assert canon_rel("conflicts_with") == "contradicts"


def test_unknown_relations_pass_through():
    assert canon_rel("zigzags_with") == "zigzags_with"
    # canonical forms are fixed points
    assert canon_rel("lives_in") == "lives_in"
    assert canon_rel("member_of") == "member_of"


def test_normalization_of_messy_verbs():
    assert canon_rel("Resides In") == "lives_in"
    assert canon_rel("RESIDES-IN") == "lives_in"
    assert canon_rel("  employed by ") == "works_at"


def test_type_disambiguation():
    # a person "based_in" a city lives there; an org is contained there
    assert canon_rel("based_in", NodeType.PERSON, NodeType.PLACE) == "lives_in"
    assert canon_rel("based_in", NodeType.ORG, NodeType.PLACE) == "located_in"
    # "went_to" an event is attendance; a place is travel; unknown stays put
    assert canon_rel("went_to", NodeType.PERSON, NodeType.EVENT) == "attends"
    assert canon_rel("went_to", NodeType.PERSON, NodeType.PLACE) == "traveled_to"
    assert canon_rel("went_to", NodeType.PERSON, None) == "went_to"
    # visiting a place is travel; visiting a person stays as stated
    assert canon_rel("visited", NodeType.PERSON, NodeType.PLACE) == "traveled_to"
    assert canon_rel("visited", NodeType.PERSON, NodeType.PERSON) == "visited"
    # joining an org vs an event
    assert canon_rel("joined", NodeType.PERSON, NodeType.ORG) == "member_of"
    assert canon_rel("joined", NodeType.PERSON, NodeType.EVENT) == "attends"


# --------------------------------------------- rules fire after canon ------

def test_auto_supersede_fires_across_extractor_verbs():
    """The Rachel-class invention, gemma flavor: old residence as 'resides_in',
    the move as 'relocated_to'. Different rel strings = different supersede
    groups = the stale city survives at full weight. Canonicalization unifies
    them so the newer edge downweights the older."""
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_chi", NodeType.PLACE, "Chicago"))
    g.add_node(Node("pl_den", NodeType.PLACE, "Denver"))
    old = Edge("p_r", "pl_chi", "resides_in", 0.9, timestamp="2025-01-01T00:00:00")
    new = Edge("p_r", "pl_den", "relocated_to", 0.9, timestamp="2025-06-01T00:00:00")
    g.add_edge(old)
    g.add_edge(new)

    assert g.auto_supersede() == 0          # verbs differ: rule no-ops (the bug)
    changed = canonicalize(g)
    assert changed["edges"] == 2            # both rewritten to lives_in
    assert g.auto_supersede() == 1          # now the rule fires
    assert old.weight < 0.4                 # stale Chicago downweighted
    assert new.weight == 0.9                # current Denver untouched


def test_taxonomy_subtree_sees_canonicalized_containment():
    """Scoped counting/membership walks CONTAINMENT_RELS; gemma's
    'situated_in' is invisible to it until canonicalized."""
    g = EvidenceGraph()
    g.add_node(Node("pl_tz", NodeType.PLACE, "Tanzania"))
    g.add_node(Node("pl_ar", NodeType.PLACE, "Arusha"))
    g.add_edge(Edge("pl_ar", "pl_tz", "situated_in", 0.9))

    assert "pl_ar" not in subtree(g, "pl_tz")   # the bug
    canonicalize(g)
    assert "pl_ar" in subtree(g, "pl_tz")       # the fix


def test_parent_ladder_sees_canonicalized_membership():
    g = EvidenceGraph()
    g.add_node(Node("p_x", NodeType.PERSON, "Person X"))
    g.add_node(Node("o_i", NodeType.ORG, "IAESTE"))
    g.add_edge(Edge("p_x", "o_i", "affiliated_with", 0.9))

    assert list(g.parents("p_x")) == []
    canonicalize(g)
    assert [dst for dst, _ in g.parents("p_x")] == ["o_i"]


# ------------------------------------------------------------ attr keys ----

def test_attr_synonyms_copied_not_overwritten():
    g = EvidenceGraph()
    g.add_node(Node("as_1", NodeType.ASSERTION, "pre-approval $400,000",
                    attrs={"amount": "$400,000"}))
    g.add_node(Node("ev_1", NodeType.EVENT, "AI summit",
                    attrs={"event_date": "2026-03-05", "date": "2026-03-06"}))
    canonicalize(g)
    assert g.node("as_1").attrs["value"] == "$400,000"
    assert g.node("as_1").attrs["amount"] == "$400,000"   # original kept
    assert g.node("ev_1").attrs["date"] == "2026-03-06"   # never overwritten


def test_canonicalize_is_idempotent():
    g = EvidenceGraph()
    g.add_node(Node("p_r", NodeType.PERSON, "Rachel"))
    g.add_node(Node("pl_chi", NodeType.PLACE, "Chicago"))
    g.add_edge(Edge("p_r", "pl_chi", "resides_in", 0.9))
    first = canonicalize(g)
    assert first["edges"] == 1
    second = canonicalize(g)
    assert second == {"edges": 0, "attrs": 0}
