"""Entity resolution v1: alias-aware, type-gated, deterministic (apm/entities.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from apm import Edge, EvidenceGraph, Node, NodeType
from apm.entities import merge_nodes, normalize, remember_alias, resolve, similarity


def test_stopwords_carry_no_identity():
    assert normalize("my new car") == "car"
    assert similarity("my new car", "the car") == 100.0


def test_reordered_tokens_match():
    assert similarity("charity 5K run", "5K charity run") == 100.0


def test_type_gating_prevents_cross_type_merge():
    g = EvidenceGraph()
    g.add_node(Node("pl_paris", NodeType.PLACE, "Paris"))
    assert resolve(g, "Paris", NodeType.PLACE) == "pl_paris"
    assert resolve(g, "Paris", NodeType.PERSON) is None   # Paris the friend != Paris the city


def test_person_names_held_to_stricter_standard():
    g = EvidenceGraph()
    g.add_node(Node("p_john", NodeType.PERSON, "John"))
    assert resolve(g, "John", NodeType.PERSON) == "p_john"
    assert resolve(g, "Jon", NodeType.PERSON) is None     # 86 < 92: near-name, no merge


def test_alias_memory_strengthens_resolution():
    g = EvidenceGraph()
    g.add_node(Node("t_car", NodeType.TOPIC, "Honda Civic"))
    assert resolve(g, "the car", NodeType.TOPIC) is None  # not yet known as alias
    remember_alias(g, "t_car", "my car")
    assert resolve(g, "the car", NodeType.TOPIC) == "t_car"  # now it is


def test_merge_rewires_edges_and_keeps_aliases():
    g = EvidenceGraph()
    g.add_node(Node("p_u", NodeType.PERSON, "User"))
    g.add_node(Node("t_a", NodeType.TOPIC, "Gym membership"))
    g.add_node(Node("t_b", NodeType.TOPIC, "gym subscription",
                    attrs={"price": "40eur"}))
    g.add_edge(Edge("p_u", "t_b", "pays_for", 0.9))
    merge_nodes(g, "t_a", "t_b")
    assert "t_b" not in g.nodes
    assert "gym subscription" in g.nodes["t_a"].attrs["aliases"]
    assert g.nodes["t_a"].attrs["price"] == "40eur"
    rels = [e.rel for _, e in g.neighbors("t_a")]
    assert "pays_for" in rels                       # edge rewired to survivor


def test_cross_session_same_entity_single_node():
    """The knowledge-update prerequisite: session-3 phrasing must land on the
    session-1 node, or contradiction machinery can never fire."""
    g = EvidenceGraph()
    g.add_node(Node("t_gym", NodeType.TOPIC, "my gym membership"))
    for phrasing in ["the gym membership", "my gym membership plan",
                     "gym membership"]:
        assert resolve(g, phrasing, NodeType.TOPIC) == "t_gym", phrasing
