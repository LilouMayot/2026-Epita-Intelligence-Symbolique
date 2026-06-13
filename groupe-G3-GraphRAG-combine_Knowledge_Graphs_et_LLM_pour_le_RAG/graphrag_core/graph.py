"""Knowledge graph construction with RDFLib and Leiden community detection."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import networkx as nx
from rdflib import Graph as RDFGraph, Namespace, URIRef

from .extractor import Triple

KG = Namespace("http://graphrag.local/")
_COLORS = ["#6366f1", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]


@dataclass
class Community:
    """A detected community of nodes in the knowledge graph.

    Attributes:
        id: Zero-based integer identifier for the community.
        nodes: List of node names belonging to this community.
        label: Human-readable label, defaults to an empty string.
    """

    id: int
    nodes: List[str]
    label: str = ""


@dataclass
class KnowledgeGraph:
    """A fully constructed knowledge graph with community structure.

    Attributes:
        rdf: An RDFLib Graph containing the triples as RDF statements.
        nx_graph: A directed NetworkX graph where nodes are entity names
            and edges carry a ``relation`` attribute.
        communities: List of Community instances produced by community
            detection.
        node_to_community: Mapping from node name to the id of its
            community.
    """

    rdf: RDFGraph
    nx_graph: nx.DiGraph
    communities: List[Community]
    node_to_community: Dict[str, int]


def _uri(name: str) -> URIRef:
    """Convert a plain entity name to a local RDF URI.

    Spaces are replaced with underscores so the result is a valid URI.

    Args:
        name: The entity name to convert.

    Returns:
        A URIRef in the ``http://graphrag.local/`` namespace.
    """
    return KG[name.replace(" ", "_")]


def build_knowledge_graph(triples: List[Triple], docs_map: Dict[str, str]) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a list of triples.

    Each Triple is added both to an RDFLib graph (as an RDF statement using
    local URIs) and to a directed NetworkX graph (as a labelled edge).
    Community detection is then run on the undirected projection of the
    NetworkX graph.

    Args:
        triples: Extracted Triple instances to add as graph edges.
        docs_map: Mapping of filename to document text (reserved for future
            use, e.g. attaching source provenance to nodes).

    Returns:
        A KnowledgeGraph with ``rdf``, ``nx_graph``, ``communities``, and
        ``node_to_community`` fully populated.
    """
    rdf = RDFGraph()
    rdf.bind("kg", KG)
    G: nx.DiGraph = nx.DiGraph()
    for t in triples:
        rdf.add((_uri(t.subject), _uri(t.relation), _uri(t.object)))
        G.add_edge(t.subject, t.object, relation=t.relation)
    communities = _detect_communities(G)
    node_to_comm = {n: c.id for c in communities for n in c.nodes}
    return KnowledgeGraph(rdf=rdf, nx_graph=G, communities=communities, node_to_community=node_to_comm)


def _detect_communities(G: nx.DiGraph) -> List[Community]:
    """Detect communities in a directed graph using the Leiden algorithm.

    The directed graph is first converted to undirected for community
    detection.  The function tries to use ``igraph`` and ``leidenalg``
    (Leiden algorithm with modularity optimisation).  If either package is
    missing it falls back to NetworkX connected components, each treated as
    a separate community.

    Args:
        G: A directed NetworkX graph whose nodes are entity name strings.

    Returns:
        A list of Community instances, one per detected community.  Returns
        an empty list when the graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return []
    U = G.to_undirected()
    nodes = list(U.nodes())
    try:
        import igraph as ig
        import leidenalg
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in U.edges()]
        ig_g = ig.Graph(n=len(nodes), edges=edges)
        partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
        return [
            Community(id=i, nodes=[nodes[j] for j in cluster], label=f"Communauté {i + 1}")
            for i, cluster in enumerate(partition)
        ]
    except ImportError:
        return [
            Community(id=i, nodes=list(comp), label=f"Communauté {i + 1}")
            for i, comp in enumerate(nx.connected_components(U))
        ]


def graph_to_json(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Serialise a KnowledgeGraph to a JSON-compatible dictionary.

    The returned structure contains four top-level keys:

    * ``nodes`` — list of node descriptors (id, community id, colour, degree).
    * ``edges`` — list of edge descriptors (source, target, relation label).
    * ``communities`` — list of community descriptors (id, label, nodes).
    * ``stats`` — summary counts (node_count, edge_count, community_count).

    Args:
        kg: The KnowledgeGraph to serialise.

    Returns:
        A dictionary that is directly serialisable to JSON.
    """
    nodes = [
        {
            "id": n,
            "community": kg.node_to_community.get(n, 0),
            "color": _COLORS[kg.node_to_community.get(n, 0) % len(_COLORS)],
            "degree": kg.nx_graph.degree(n),
        }
        for n in kg.nx_graph.nodes()
    ]
    edges = [
        {"source": u, "target": v, "relation": d.get("relation", "")}
        for u, v, d in kg.nx_graph.edges(data=True)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "communities": [{"id": c.id, "label": c.label, "nodes": c.nodes} for c in kg.communities],
        "stats": {
            "node_count": kg.nx_graph.number_of_nodes(),
            "edge_count": kg.nx_graph.number_of_edges(),
            "community_count": len(kg.communities),
        },
    }
