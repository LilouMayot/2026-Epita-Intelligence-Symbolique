from graphrag_core.extractor import Triple
from graphrag_core.graph import build_knowledge_graph, graph_to_json

TRIPLES = [
    Triple("Alice", "WORKS_AT", "ACME"),
    Triple("Alice", "KNOWS", "Bob"),
    Triple("Bob", "WORKS_AT", "TechCorp"),
]


def test_nodes_present():
    """All entity names from triples appear as nodes in the nx_graph."""
    kg = build_knowledge_graph(TRIPLES, {})
    for name in ("Alice", "ACME", "Bob", "TechCorp"):
        assert name in kg.nx_graph.nodes()


def test_edges_present():
    """Edges derived from triples exist in the nx_graph."""
    kg = build_knowledge_graph(TRIPLES, {})
    assert kg.nx_graph.has_edge("Alice", "ACME")
    assert kg.nx_graph.has_edge("Bob", "TechCorp")


def test_communities_cover_all_nodes():
    """Every node belongs to exactly one community."""
    kg = build_knowledge_graph(TRIPLES, {})
    assert len(kg.communities) >= 1
    covered = {n for c in kg.communities for n in c.nodes}
    assert covered == set(kg.nx_graph.nodes())


def test_node_to_community_complete():
    """node_to_community maps every node to a community id."""
    kg = build_knowledge_graph(TRIPLES, {})
    for node in kg.nx_graph.nodes():
        assert node in kg.node_to_community


def test_graph_to_json_structure():
    """graph_to_json returns the expected keys and correct stat counts."""
    kg = build_knowledge_graph(TRIPLES, {})
    data = graph_to_json(kg)
    assert data["stats"]["node_count"] == 4
    assert data["stats"]["edge_count"] == 3
    assert "nodes" in data and "edges" in data and "communities" in data


def test_empty_graph():
    """An empty triple list produces a graph with zero nodes and communities."""
    kg = build_knowledge_graph([], {})
    data = graph_to_json(kg)
    assert data["stats"]["node_count"] == 0
    assert data["stats"]["community_count"] == 0
