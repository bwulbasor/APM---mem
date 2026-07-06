"""Taxonomy layer: category super-parents, containment hierarchy, scoped queries.

The Tanzania scenario (Abiram's design): NAMES/PLACES/EVENTS super-parents;
under PLACES sits Tanzania; under Tanzania, links to people and events.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime

from apm import Edge, Engine, EvidenceGraph, Node, NodeType, Placeholder, Query, VerdictClass
from apm.activation import spread
from apm.taxonomy import cat_id, ensure_categories, parents_of, subtree

NOW = datetime.fromisoformat("2026-06-22T10:00:00")


def _world():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    # places with containment: Arusha + Zanzibar inside Tanzania; Oslo elsewhere
    g.add_node(Node("pl_tz", NodeType.PLACE, "Tanzania"))
    g.add_node(Node("pl_arusha", NodeType.PLACE, "Arusha"))
    g.add_node(Node("pl_zanzibar", NodeType.PLACE, "Zanzibar"))
    g.add_node(Node("pl_oslo", NodeType.PLACE, "Oslo"))
    g.add_edge(Edge("pl_arusha", "pl_tz", "located_in", 0.9))
    g.add_edge(Edge("pl_zanzibar", "pl_tz", "located_in", 0.9))
    # people and events linked under Tanzania's places
    g.add_node(Node("p_juma", NodeType.PERSON, "Juma the guide"))
    g.add_edge(Edge("p_juma", "pl_arusha", "lives_in", 0.9))
    g.add_node(Node("ev_safari", NodeType.EVENT, "Serengeti safari",
                    timestamp="2025-07-10T08:00:00"))
    g.add_edge(Edge("ev_safari", "pl_arusha", "located_in", 0.85))
    g.add_node(Node("ev_oslo_talk", NodeType.EVENT, "Oslo conference talk",
                    timestamp="2025-09-01T09:00:00"))
    g.add_edge(Edge("ev_oslo_talk", "pl_oslo", "located_in", 0.85))
    for x in ("pl_arusha", "pl_zanzibar", "pl_oslo", "ev_safari", "ev_oslo_talk"):
        g.add_edge(Edge("p_u", x, "visited" if x.startswith("pl") else "attended", 0.9))
    ensure_categories(g)
    return g


def test_every_instance_gets_a_category_parent():
    g = _world()
    assert cat_id(NodeType.PLACE) in parents_of(g, "pl_tz")
    assert cat_id(NodeType.PERSON) in parents_of(g, "p_juma")


def test_subtree_of_tanzania_contains_its_world():
    """Under Tanzania: its cities, and the events/people attached beneath them."""
    g = _world()
    tz = subtree(g, "pl_tz")
    assert {"pl_arusha", "pl_zanzibar", "ev_safari"} <= tz
    assert "pl_oslo" not in tz and "ev_oslo_talk" not in tz


def test_hub_rule_categories_never_carry_activation():
    """Activation may reach siblings through REAL associations (a shared User
    node), but never through the PLACES category hub."""
    g = _world()
    reached = spread(g, ["pl_oslo"])
    assert cat_id(NodeType.PLACE) not in reached          # can't even enter
    for node_id, paths in reached.items():
        for p in paths:
            for e in p.edges:
                assert not e.src.startswith("cat_"), f"{node_id} via {e.src}"
                assert not e.dst.startswith("cat_"), f"{node_id} via {e.dst}"


def test_hub_rule_kills_sibling_leak_without_user_hub():
    """Remove the legitimate association route: with only the taxonomy linking
    them, one place must NOT light up another."""
    g = EvidenceGraph()
    g.add_node(Node("pl_a", NodeType.PLACE, "Arusha"))
    g.add_node(Node("pl_b", NodeType.PLACE, "Oslo"))
    ensure_categories(g)   # both now share the PLACES parent
    reached = spread(g, ["pl_a"])
    assert "pl_b" not in reached                          # sibling stays dark


def test_scoped_count_events_under_tanzania():
    """'How many events on the Tanzania trip?' — deterministic subtree count."""
    g = _world()
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.EVENT, {}, under="pl_tz"),
              mode="count")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.CONFIRMED             # taxonomy scope = strongest count
    assert v.answer.attrs["count"] == 1
    assert "Serengeti safari" in v.chain.render()


def test_scoped_resolve_excludes_out_of_scope_candidates():
    """'Which event was in Tanzania?' — Oslo talk is better-connected but out of scope."""
    g = _world()
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.EVENT, {}, under="pl_tz"))
    v = Engine(g).answer(q, now=NOW)
    assert v.answer.id == "ev_safari"


def test_missing_scope_node_yields_unknown():
    g = _world()
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.EVENT, {}, under="pl_nowhere"),
              mode="count")
    v = Engine(g).answer(q, now=NOW)
    assert v.klass == VerdictClass.UNKNOWN
