"""v0.8: episode clustering + consolidation (offline parts)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apm import Edge, EvidenceGraph, Node, NodeType
from apm.activation import spread
from apm.episodes import build_episodes
from apm.taxonomy import subtree


def _two_trip_graph():
    """Two dense clusters (Tanzania trip, mortgage saga) + a bridge person."""
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    trip = [("pl_tz", NodeType.PLACE, "Tanzania"),
            ("ev_safari", NodeType.EVENT, "Serengeti safari"),
            ("p_juma", NodeType.PERSON, "Juma the guide"),
            ("ev_hike", NodeType.EVENT, "Kilimanjaro hike"),
            ("t_lodge", NodeType.TOPIC, "Safari lodge")]
    money = [("org_wf", NodeType.ORG, "Wells Fargo"),
             ("a_pre", NodeType.ASSERTION, "$400,000 pre-approval"),
             ("t_house", NodeType.TOPIC, "Craftsman house"),
             ("a_rate", NodeType.ASSERTION, "4% fixed rate"),
             ("p_agent", NodeType.PERSON, "Realtor Dana")]
    for nid, t, label in trip + money:
        g.add_node(Node(nid, t, label))
    # dense intra-cluster edges
    for grp in (trip, money):
        ids = [x[0] for x in grp]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                g.add_edge(Edge(a, b, "related_to", 0.8))
    # user bridges both worlds
    g.add_edge(Edge("p_u", "ev_safari", "attended", 0.9))
    g.add_edge(Edge("p_u", "a_pre", "about", 0.9))
    return g


def test_episodes_discovered_and_scoped():
    g = _two_trip_graph()
    made = build_episodes(g)
    assert made == 2
    eps = [n for n in g.nodes.values() if n.id.startswith("ep_")]
    labels = " | ".join(e.label for e in eps)
    assert "Episode:" in labels
    # each episode subtree contains its members
    for ep in eps:
        members = subtree(g, ep.id)
        assert len(members) >= 5


def test_episode_nodes_never_carry_activation():
    g = _two_trip_graph()
    build_episodes(g)
    reached = spread(g, ["ev_safari"])
    for node_id, paths in reached.items():
        assert not node_id.startswith("ep_")
        for p in paths:
            for e in p.edges:
                assert not e.src.startswith("ep_") and not e.dst.startswith("ep_")


def test_build_episodes_idempotent():
    g = _two_trip_graph()
    assert build_episodes(g) == 2
    assert build_episodes(g) == 2         # rebuild, not duplicate
    assert sum(1 for n in g.nodes if n.startswith("ep_")) == 2


def test_consolidate_merges_rephrasings_and_drops_orphans():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("a_1", NodeType.ASSERTION, "Personal best 25:50 in charity 5K",
                    timestamp="2026-01-01T00:00:00"))
    g.add_node(Node("a_2", NodeType.ASSERTION, "personal best 25:50 charity 5K run",
                    timestamp="2026-01-01T00:00:05"))
    g.add_edge(Edge("a_1", "p_u", "about", 0.9))
    g.add_edge(Edge("a_2", "p_u", "about", 0.9))
    g.add_node(Node("t_orphan", NodeType.TOPIC, "dangling extraction noise"))
    stats = g.consolidate()
    assert stats["merged"] == 1
    assert stats["dropped"] == 1
    assert "t_orphan" not in g.nodes
    survivors = [n for n in g.nodes.values() if n.type == NodeType.ASSERTION]
    assert len(survivors) == 1
    assert survivors[0].attrs.get("aliases")     # merge left an alias trail


def test_consolidate_never_merges_different_values():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("a_1", NodeType.ASSERTION, "$350,000 pre-approval",
                    timestamp="2026-01-01T00:00:00"))
    g.add_node(Node("a_2", NodeType.ASSERTION, "$400,000 pre-approval",
                    timestamp="2026-02-01T00:00:00"))
    g.add_edge(Edge("a_1", "p_u", "about", 0.9))
    g.add_edge(Edge("a_2", "p_u", "about", 0.9))
    assert g.consolidate()["merged"] == 0     # numbers are identity


def test_sum_mode_totals_quantities():
    from datetime import datetime
    from apm import Engine, Placeholder, Query
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    for i, (label, dur) in enumerate([("Yosemite camping trip", "3 days"),
                                      ("Zion camping trip", "2 days"),
                                      ("Acadia camping trip", "3 days")]):
        nid = f"ev_{i}"
        g.add_node(Node(nid, NodeType.EVENT, f"{label} ({dur})",
                        attrs={"duration": dur}))
        g.add_edge(Edge("p_u", nid, "went_on", 0.9))
    # distractor without camping in constraints match... constraints soft-match
    q = Query(anchors=["p_u"],
              placeholder=Placeholder(NodeType.EVENT, {"unit": "days"}),
              mode="sum")
    v = Engine(g).answer(q, now=datetime.fromisoformat("2026-06-22T10:00:00"))
    from apm import VerdictClass as VC
    assert v.klass == VC.ASSUMED
    assert v.answer.attrs["total"] == 8.0
    assert "total = 8" in v.chain.render()
