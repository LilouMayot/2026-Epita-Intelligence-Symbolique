# G3 GraphRAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implémenter le pipeline GraphRAG complet — package `graphrag_core` partagé, app web FastAPI + frontend HTML/JS, notebooks enrichis et benchmark.

**Architecture:** Package Python `graphrag_core/` importé par le backend FastAPI et les notebooks. App web single-page avec sidebar persistante, 2 tabs (Q&A + Graphe D3.js), SSE pour le build en live.

**Tech Stack:** Python 3.11+, FastAPI, rdflib, NetworkX, leidenalg/igraph, D3.js, pytest, python-dotenv, openai/anthropic/requests (multi-provider)

**Conventions (CRITIQUES) :**
- Tous les fichiers dans `groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/`
- Commits : `type(G3): description` — jamais d'autre scope, jamais de co-auteur
- `.env` unique à la racine du dossier G3

---

## File Map

```
groupe-G3-GraphRAG-.../
├── .env.example
├── .gitignore                    (modifier)
├── requirements.txt
├── graphrag_core/
│   ├── __init__.py
│   ├── llm.py
│   ├── extractor.py
│   ├── graph.py
│   ├── retriever.py
│   └── pipeline.py
├── app/
│   ├── main.py
│   ├── models.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── tests/
│   ├── __init__.py
│   ├── test_llm.py
│   ├── test_extractor.py
│   ├── test_graph.py
│   ├── test_retriever.py
│   ├── test_pipeline.py
│   └── test_api.py
├── data/
│   ├── hotpotqa/
│   ├── custom/
│   └── download_hotpotqa.py
├── notebooks/
│   ├── SW-11-Python-KnowledgeGraphs.ipynb   (git mv)
│   ├── SW-12-Python-GraphRAG.ipynb          (git mv)
│   ├── SW-4b-Python-SPARQL.ipynb            (git mv)
│   └── benchmark.ipynb                      (nouveau)
└── docs/
    ├── 2026-06-13-G3-graphrag-design.md
    └── 2026-06-13-G3-implementation-plan.md
```

---

## Task 1 — Scaffold du projet

**Files:**
- Create: `groupe-G3-GraphRAG-.../requirements.txt`
- Create: `groupe-G3-GraphRAG-.../.env.example`
- Modify: `groupe-G3-GraphRAG-.../.gitignore`
- Create: `groupe-G3-GraphRAG-.../graphrag_core/__init__.py`
- Create: `groupe-G3-GraphRAG-.../app/__init__.py`
- Create: `groupe-G3-GraphRAG-.../tests/__init__.py`

- [ ] **Step 1: Créer requirements.txt**

```
# groupe-G3-GraphRAG-.../requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-dotenv>=1.0.0
rdflib>=7.0.0
networkx>=3.3
leidenalg>=0.10.2
python-igraph>=0.11.6
openai>=1.40.0
anthropic>=0.34.0
requests>=2.32.0
python-multipart>=0.0.9
pytest>=8.0.0
httpx>=0.27.0
datasets>=2.20.0
```

- [ ] **Step 2: Créer .env.example**

```
# groupe-G3-GraphRAG-.../.env.example
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
APP_HOST=0.0.0.0
APP_PORT=8000
DATA_DIR=./data
```

- [ ] **Step 3: Mettre à jour .gitignore**

Ajouter à `.gitignore` existant :
```
.env
data/custom/
data/hotpotqa/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Créer les dossiers et __init__.py vides**

```bash
cd groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG
mkdir -p graphrag_core app/static tests data/hotpotqa data/custom notebooks
touch graphrag_core/__init__.py app/__init__.py tests/__init__.py
```

- [ ] **Step 5: Déplacer les notebooks existants**

```bash
cd groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG
git mv SW-11-Python-KnowledgeGraphs.ipynb notebooks/
git mv SW-12-Python-GraphRAG.ipynb notebooks/
git mv SW-4b-Python-SPARQL.ipynb notebooks/
```

- [ ] **Step 6: Installer les dépendances**

```bash
pip install -r requirements.txt
```

- [ ] **Step 7: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/
git commit -m "chore(G3): scaffold project structure, move notebooks to notebooks/"
```

---

## Task 2 — `graphrag_core/llm.py`

**Files:**
- Create: `graphrag_core/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_llm.py
import pytest
from unittest.mock import patch, MagicMock

def test_openai_client_complete():
    with patch("graphrag_core.llm.OpenAI") as mock_openai:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_resp
        from graphrag_core.llm import OpenAIClient
        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        assert client.complete("Hello") == "Test response"

def test_get_llm_client_openai():
    with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "sk-test"}):
        with patch("graphrag_core.llm.OpenAI"):
            from graphrag_core.llm import get_llm_client, OpenAIClient
            assert isinstance(get_llm_client(), OpenAIClient)

def test_get_llm_client_unknown_raises():
    with patch.dict("os.environ", {"LLM_PROVIDER": "unknown"}):
        from graphrag_core.llm import get_llm_client
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm_client()
```

- [ ] **Step 2: Vérifier que les tests échouent**

```bash
pytest tests/test_llm.py -v
```
Résultat attendu : `ModuleNotFoundError` ou `ImportError`

- [ ] **Step 3: Implémenter `graphrag_core/llm.py`**

```python
# graphrag_core/llm.py
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def complete(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120
        )
        return resp.json()["response"]

def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if provider == "openai":
        return OpenAIClient(model=model, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        return AnthropicClient(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "ollama":
        return OllamaClient(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
```

- [ ] **Step 4: Vérifier que les tests passent**

```bash
pytest tests/test_llm.py -v
```
Résultat attendu : `3 passed`

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/graphrag_core/llm.py groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_llm.py
git commit -m "feat(G3): add LLM abstraction with OpenAI/Anthropic/Ollama providers"
```

---

## Task 3 — `graphrag_core/extractor.py`

**Files:**
- Create: `graphrag_core/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_extractor.py
from unittest.mock import MagicMock
from graphrag_core.extractor import extract_triples, chunk_text, Triple

def _mock_llm(response: str):
    llm = MagicMock()
    llm.complete.return_value = response
    return llm

def test_extract_triples_basic():
    llm = _mock_llm('[{"subject": "Alice", "relation": "WORKS_AT", "object": "ACME"}]')
    triples = extract_triples("Alice works at ACME.", llm)
    assert len(triples) == 1
    assert triples[0] == Triple("Alice", "WORKS_AT", "ACME")

def test_extract_triples_deduplication():
    llm = _mock_llm('[{"subject":"A","relation":"R","object":"B"},{"subject":"A","relation":"R","object":"B"}]')
    triples = extract_triples("text", llm)
    assert len(triples) == 1

def test_extract_triples_invalid_json_returns_empty():
    llm = _mock_llm("Sorry, I cannot extract anything.")
    assert extract_triples("text", llm) == []

def test_extract_triples_missing_field_skipped():
    llm = _mock_llm('[{"subject": "A", "relation": "R"}]')
    assert extract_triples("text", llm) == []

def test_chunk_text_short_stays_single():
    assert chunk_text("hello", chunk_size=2000) == ["hello"]

def test_chunk_text_long_produces_multiple():
    chunks = chunk_text("x" * 5000, chunk_size=2000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
```

- [ ] **Step 2: Vérifier échec**

```bash
pytest tests/test_extractor.py -v
```
Résultat attendu : `ImportError`

- [ ] **Step 3: Implémenter `graphrag_core/extractor.py`**

```python
# graphrag_core/extractor.py
import json
import re
from dataclasses import dataclass
from typing import List
from .llm import LLMClient

@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str

_PROMPT = """Extract all entities and relations from the text as a JSON array of triples.
Each triple: {{"subject": "...", "relation": "UPPER_SNAKE_CASE", "object": "..."}}.
Output ONLY valid JSON. No explanation.

Text:
{text}"""

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def extract_triples(text: str, llm: LLMClient) -> List[Triple]:
    seen: set = set()
    result: List[Triple] = []
    for chunk in chunk_text(text):
        raw = llm.complete(_PROMPT.format(text=chunk))
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            continue
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        for item in items:
            if not all(k in item for k in ("subject", "relation", "object")):
                continue
            t = Triple(item["subject"].strip(), item["relation"].strip(), item["object"].strip())
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result
```

- [ ] **Step 4: Vérifier passage**

```bash
pytest tests/test_extractor.py -v
```
Résultat attendu : `6 passed`

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/graphrag_core/extractor.py groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_extractor.py
git commit -m "feat(G3): add entity/relation extractor with LLM and chunking"
```

---

## Task 4 — `graphrag_core/graph.py`

**Files:**
- Create: `graphrag_core/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_graph.py
from graphrag_core.extractor import Triple
from graphrag_core.graph import build_knowledge_graph, graph_to_json

TRIPLES = [
    Triple("Alice", "WORKS_AT", "ACME"),
    Triple("Alice", "KNOWS", "Bob"),
    Triple("Bob", "WORKS_AT", "TechCorp"),
]

def test_nodes_present():
    kg = build_knowledge_graph(TRIPLES, {})
    for name in ("Alice", "ACME", "Bob", "TechCorp"):
        assert name in kg.nx_graph.nodes()

def test_edges_present():
    kg = build_knowledge_graph(TRIPLES, {})
    assert kg.nx_graph.has_edge("Alice", "ACME")
    assert kg.nx_graph.has_edge("Bob", "TechCorp")

def test_communities_cover_all_nodes():
    kg = build_knowledge_graph(TRIPLES, {})
    assert len(kg.communities) >= 1
    covered = {n for c in kg.communities for n in c.nodes}
    assert covered == set(kg.nx_graph.nodes())

def test_node_to_community_complete():
    kg = build_knowledge_graph(TRIPLES, {})
    for node in kg.nx_graph.nodes():
        assert node in kg.node_to_community

def test_graph_to_json_structure():
    kg = build_knowledge_graph(TRIPLES, {})
    data = graph_to_json(kg)
    assert data["stats"]["node_count"] == 4
    assert data["stats"]["edge_count"] == 3
    assert "nodes" in data and "edges" in data and "communities" in data

def test_empty_graph():
    kg = build_knowledge_graph([], {})
    data = graph_to_json(kg)
    assert data["stats"]["node_count"] == 0
    assert data["stats"]["community_count"] == 0
```

- [ ] **Step 2: Vérifier échec**

```bash
pytest tests/test_graph.py -v
```

- [ ] **Step 3: Implémenter `graphrag_core/graph.py`**

```python
# graphrag_core/graph.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
import networkx as nx
from rdflib import Graph as RDFGraph, Namespace, URIRef
from .extractor import Triple

KG = Namespace("http://graphrag.local/")
_COLORS = ["#6366f1", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]

@dataclass
class Community:
    id: int
    nodes: List[str]
    label: str = ""

@dataclass
class KnowledgeGraph:
    rdf: RDFGraph
    nx_graph: nx.DiGraph
    communities: List[Community]
    node_to_community: Dict[str, int]

def _uri(name: str) -> URIRef:
    return KG[name.replace(" ", "_")]

def build_knowledge_graph(triples: List[Triple], docs_map: Dict[str, str]) -> KnowledgeGraph:
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
            Community(id=i, nodes=[nodes[j] for j in cluster], label=f"Communauté {i+1}")
            for i, cluster in enumerate(partition)
        ]
    except ImportError:
        return [
            Community(id=i, nodes=list(comp), label=f"Communauté {i+1}")
            for i, comp in enumerate(nx.connected_components(U))
        ]

def graph_to_json(kg: KnowledgeGraph) -> Dict[str, Any]:
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
```

- [ ] **Step 4: Vérifier passage**

```bash
pytest tests/test_graph.py -v
```
Résultat attendu : `6 passed`

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/graphrag_core/graph.py groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_graph.py
git commit -m "feat(G3): add KG builder with rdflib + Leiden community detection"
```

---

## Task 5 — `graphrag_core/retriever.py`

**Files:**
- Create: `graphrag_core/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_retriever.py
from graphrag_core.extractor import Triple
from graphrag_core.graph import build_knowledge_graph
from graphrag_core.retriever import detect_entities, extract_subgraph, subgraph_to_context

_TRIPLES = [
    Triple("Alice", "WORKS_AT", "ACME"),
    Triple("Alice", "KNOWS", "Bob"),
    Triple("Bob", "WORKS_AT", "TechCorp"),
    Triple("Bob", "LIVES_IN", "Paris"),
]

def _kg():
    return build_knowledge_graph(_TRIPLES, {})

def test_detect_entities_case_insensitive():
    kg = _kg()
    assert "Alice" in detect_entities("Where does alice work?", kg)

def test_detect_entities_no_match():
    assert detect_entities("What is the weather?", _kg()) == []

def test_subgraph_one_hop_includes_neighbors():
    result = extract_subgraph(_kg(), ["Alice"], max_hops=1)
    assert "Alice" in result.nodes
    assert "ACME" in result.nodes

def test_subgraph_two_hops_reaches_paris():
    result = extract_subgraph(_kg(), ["Alice"], max_hops=2)
    assert "Paris" in result.nodes

def test_subgraph_trace_hop_numbers():
    result = extract_subgraph(_kg(), ["Alice"], max_hops=2)
    hops = [s.hop for s in result.trace]
    assert 1 in hops
    assert 2 in hops

def test_subgraph_to_context_contains_relation():
    result = extract_subgraph(_kg(), ["Alice"], max_hops=1)
    ctx = subgraph_to_context(result)
    assert "--[" in ctx
    assert "Alice" in ctx
```

- [ ] **Step 2: Vérifier échec**

```bash
pytest tests/test_retriever.py -v
```

- [ ] **Step 3: Implémenter `graphrag_core/retriever.py`**

```python
# graphrag_core/retriever.py
from dataclasses import dataclass
from typing import List, Set
from .graph import KnowledgeGraph

@dataclass
class TraceStep:
    hop: int
    from_node: str
    relation: str
    to_nodes: List[str]

@dataclass
class SubgraphResult:
    nodes: List[str]
    edges: List[tuple]      # (from_node, relation, to_node)
    trace: List[TraceStep]

def detect_entities(question: str, kg: KnowledgeGraph) -> List[str]:
    q = question.lower()
    return [n for n in kg.nx_graph.nodes() if n.lower() in q]

def extract_subgraph(kg: KnowledgeGraph, seed_entities: List[str], max_hops: int = 2) -> SubgraphResult:
    G = kg.nx_graph
    visited: Set[str] = set(seed_entities)
    edges: List[tuple] = []
    trace: List[TraceStep] = []
    frontier: Set[str] = set(seed_entities)

    for hop in range(1, max_hops + 1):
        next_frontier: Set[str] = set()
        for node in frontier:
            if node not in G:
                continue
            for _, nbr, data in G.out_edges(node, data=True):
                edges.append((node, data.get("relation", ""), nbr))
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
            for pred, _, data in G.in_edges(node, data=True):
                edges.append((pred, data.get("relation", ""), node))
                if pred not in visited:
                    visited.add(pred)
                    next_frontier.add(pred)
        if next_frontier:
            trace.append(TraceStep(
                hop=hop,
                from_node=", ".join(sorted(frontier)),
                relation="(traversal)",
                to_nodes=sorted(next_frontier),
            ))
        frontier = next_frontier
        if not frontier:
            break

    unique_edges = list(dict.fromkeys(edges))
    return SubgraphResult(nodes=list(visited), edges=unique_edges, trace=trace)

def subgraph_to_context(result: SubgraphResult) -> str:
    lines = ["Knowledge Graph context:"]
    for src, rel, tgt in result.edges:
        lines.append(f"  {src} --[{rel}]--> {tgt}")
    return "\n".join(lines)
```

- [ ] **Step 4: Vérifier passage**

```bash
pytest tests/test_retriever.py -v
```
Résultat attendu : `6 passed`

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/graphrag_core/retriever.py groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_retriever.py
git commit -m "feat(G3): add multi-hop BFS retriever with trace"
```

---

## Task 6 — `graphrag_core/pipeline.py`

**Files:**
- Create: `graphrag_core/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_pipeline.py
from unittest.mock import MagicMock
from graphrag_core.extractor import Triple
from graphrag_core.graph import build_knowledge_graph
from graphrag_core.pipeline import run, PipelineResult

_TRIPLES = [Triple("Alice", "WORKS_AT", "ACME"), Triple("Alice", "KNOWS", "Bob")]

def _kg():
    return build_knowledge_graph(_TRIPLES, {})

def _llm(answer="Alice works at ACME."):
    m = MagicMock()
    m.complete.return_value = answer
    return m

def test_run_returns_pipeline_result():
    result = run("Where does Alice work?", _kg(), _llm())
    assert isinstance(result, PipelineResult)
    assert result.answer == "Alice works at ACME."

def test_run_subgraph_nodes_populated():
    result = run("Where does Alice work?", _kg(), _llm())
    assert len(result.subgraph_nodes) > 0
    assert "Alice" in result.subgraph_nodes

def test_run_docs_used_matched():
    docs = {"file.txt": "Alice is an employee at ACME corporation."}
    result = run("Where does Alice work?", _kg(), _llm(), docs_map=docs)
    assert any(d.filename == "file.txt" for d in result.docs_used)

def test_run_no_entity_match_uses_fallback():
    result = run("What is the meaning of life?", _kg(), _llm("I don't know."))
    assert result.answer == "I don't know."
    assert len(result.subgraph_nodes) > 0  # fallback to top-degree nodes
```

- [ ] **Step 2: Vérifier échec**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implémenter `graphrag_core/pipeline.py`**

```python
# graphrag_core/pipeline.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .llm import LLMClient
from .graph import KnowledgeGraph
from .retriever import TraceStep, detect_entities, extract_subgraph, subgraph_to_context

@dataclass
class DocReference:
    filename: str
    pages: Optional[List[int]] = None
    sections: Optional[List[str]] = None

@dataclass
class PipelineResult:
    answer: str
    trace: List[TraceStep]
    docs_used: List[DocReference]
    subgraph_nodes: List[str]
    subgraph_edges: List[tuple]

_PROMPT = """You are a precise Q&A assistant. Use ONLY the knowledge graph context below to answer.
If the context is insufficient, say so briefly.

{context}

Question: {question}

Answer:"""

def run(
    question: str,
    kg: KnowledgeGraph,
    llm: LLMClient,
    docs_map: Optional[Dict[str, str]] = None,
    max_hops: int = 2,
) -> PipelineResult:
    seeds = detect_entities(question, kg)
    if not seeds:
        degrees = dict(kg.nx_graph.degree())
        seeds = sorted(degrees, key=degrees.get, reverse=True)[:3]

    subgraph = extract_subgraph(kg, seeds, max_hops=max_hops)
    context = subgraph_to_context(subgraph)
    answer = llm.complete(_PROMPT.format(context=context, question=question))

    docs_used = []
    if docs_map:
        for fname, text in docs_map.items():
            if any(n.lower() in text.lower() for n in subgraph.nodes):
                docs_used.append(DocReference(filename=fname))

    return PipelineResult(
        answer=answer,
        trace=subgraph.trace,
        docs_used=docs_used,
        subgraph_nodes=subgraph.nodes,
        subgraph_edges=subgraph.edges,
    )
```

- [ ] **Step 4: Vérifier passage**

```bash
pytest tests/ -v
```
Résultat attendu : tous les tests passent (llm, extractor, graph, retriever, pipeline)

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/graphrag_core/pipeline.py groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_pipeline.py
git commit -m "feat(G3): add pipeline orchestrator — query to answer with trace"
```

---

## Task 7 — `app/models.py` + `app/main.py`

**Files:**
- Create: `app/models.py`
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Créer `app/models.py`**

```python
# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Any

class TraceStepSchema(BaseModel):
    hop: int
    from_node: str
    relation: str
    to_nodes: List[str]

class DocReferenceSchema(BaseModel):
    filename: str
    pages: Optional[List[int]] = None
    sections: Optional[List[str]] = None

class QueryRequest(BaseModel):
    question: str
    max_hops: int = 2

class QueryResponse(BaseModel):
    answer: str
    trace: List[TraceStepSchema]
    docs_used: List[DocReferenceSchema]
    subgraph_nodes: List[str]
    subgraph_edges: List[List[Any]]

class DatasetSelectRequest(BaseModel):
    dataset: str
```

- [ ] **Step 2: Écrire les tests API**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def reset_state():
    from app.main import state
    state["kg"] = None
    state["docs_map"] = {}
    state["active_dataset"] = "custom"

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)

def test_query_without_kg_400(client):
    assert client.post("/query", json={"question": "test"}).status_code == 400

def test_graph_without_kg_404(client):
    assert client.get("/graph").status_code == 404

def test_dataset_select_valid(client):
    r = client.post("/dataset/select", json={"dataset": "hotpotqa"})
    assert r.status_code == 200
    assert r.json()["active"] == "hotpotqa"

def test_dataset_select_invalid_400(client):
    assert client.post("/dataset/select", json={"dataset": "bad"}).status_code == 400

def test_list_datasets_structure(client):
    r = client.get("/datasets")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body and "datasets" in body

def test_query_with_kg(client):
    from graphrag_core.extractor import Triple
    from graphrag_core.graph import build_knowledge_graph
    from app.main import state
    state["kg"] = build_knowledge_graph([Triple("Alice", "WORKS_AT", "ACME")], {})
    state["docs_map"] = {"f.txt": "Alice works at ACME."}
    with patch("app.main.get_llm_client") as mock:
        mock.return_value.complete.return_value = "Alice works at ACME."
        r = client.post("/query", json={"question": "Where does Alice work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Alice works at ACME."
    assert "trace" in body
    assert "subgraph_nodes" in body
```

- [ ] **Step 3: Implémenter `app/main.py`**

```python
# app/main.py
import os
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from graphrag_core.llm import get_llm_client
from graphrag_core.extractor import extract_triples
from graphrag_core.graph import build_knowledge_graph, graph_to_json
from graphrag_core.pipeline import run as pipeline_run
from app.models import (
    QueryRequest, QueryResponse, DatasetSelectRequest,
    TraceStepSchema, DocReferenceSchema,
)

app = FastAPI(title="GraphRAG Demo")

_BASE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", _BASE.parent / "data"))
CUSTOM_DIR = DATA_DIR / "custom"
HOTPOTQA_DIR = DATA_DIR / "hotpotqa"
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
HOTPOTQA_DIR.mkdir(parents=True, exist_ok=True)

state: dict = {"kg": None, "docs_map": {}, "active_dataset": "custom"}

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        dest = CUSTOM_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(f.filename)
    return {"saved": saved}

@app.post("/reset")
async def reset_custom():
    for f in CUSTOM_DIR.iterdir():
        f.unlink()
    state["kg"] = None
    state["docs_map"] = {}
    return {"status": "reset"}

def _sse(stage: str, progress: int, message: str) -> str:
    return f"data: {json.dumps({'stage': stage, 'progress': progress, 'message': message})}\n\n"

async def _build_gen() -> AsyncGenerator[str, None]:
    try:
        llm = get_llm_client()
        src = HOTPOTQA_DIR if state["active_dataset"] == "hotpotqa" else CUSTOM_DIR
        files = [f for f in src.iterdir() if f.is_file()]
        if not files:
            yield _sse("error", 0, "Aucun document trouvé.")
            return

        docs_map: dict = {}
        all_triples = []
        for i, f in enumerate(files):
            text = f.read_text(encoding="utf-8", errors="ignore")
            docs_map[f.name] = text
            yield _sse("extraction", int(i / len(files) * 40), f"Extraction : {f.name}")
            await asyncio.sleep(0)
            all_triples.extend(extract_triples(text, llm))

        yield _sse("extraction", 40, f"{len(all_triples)} triplets extraits")
        yield _sse("graph_build", 50, "Construction du graphe RDF…")
        await asyncio.sleep(0)
        yield _sse("community_detection", 70, "Détection de communautés (Leiden)…")
        await asyncio.sleep(0)
        kg = build_knowledge_graph(all_triples, docs_map)
        yield _sse("indexing", 90, "Indexation pour le retrieval…")
        await asyncio.sleep(0)
        state["kg"] = kg
        state["docs_map"] = docs_map
        s = graph_to_json(kg)["stats"]
        yield _sse("done", 100, f"KG prêt — {s['node_count']} entités, {s['edge_count']} relations, {s['community_count']} communautés")
    except Exception as e:
        yield _sse("error", 0, str(e))

@app.post("/build")
async def build():
    return StreamingResponse(_build_gen(), media_type="text/event-stream")

@app.get("/graph")
async def get_graph():
    if state["kg"] is None:
        raise HTTPException(404, "KG non construit")
    return graph_to_json(state["kg"])

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if state["kg"] is None:
        raise HTTPException(400, "KG non construit")
    llm = get_llm_client()
    r = pipeline_run(req.question, state["kg"], llm, state["docs_map"], req.max_hops)
    return QueryResponse(
        answer=r.answer,
        trace=[TraceStepSchema(hop=s.hop, from_node=s.from_node, relation=s.relation, to_nodes=s.to_nodes) for s in r.trace],
        docs_used=[DocReferenceSchema(filename=d.filename, pages=d.pages, sections=d.sections) for d in r.docs_used],
        subgraph_nodes=r.subgraph_nodes,
        subgraph_edges=[[e[0], e[1], e[2]] for e in r.subgraph_edges],
    )

@app.get("/datasets")
async def list_datasets():
    return {
        "active": state["active_dataset"],
        "datasets": {
            "custom": {"files": [f.name for f in CUSTOM_DIR.iterdir() if f.is_file()]},
            "hotpotqa": {"files": [f.name for f in HOTPOTQA_DIR.iterdir() if f.is_file()]},
        },
    }

@app.post("/dataset/select")
async def select_dataset(req: DatasetSelectRequest):
    if req.dataset not in ("hotpotqa", "custom"):
        raise HTTPException(400, "Dataset inconnu. Valeurs: hotpotqa | custom")
    state["active_dataset"] = req.dataset
    state["kg"] = None
    return {"active": req.dataset}

app.mount("/", StaticFiles(directory=_BASE / "static", html=True), name="static")
```

- [ ] **Step 4: Vérifier les tests**

```bash
pytest tests/test_api.py -v
```
Résultat attendu : `6 passed`

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/app/
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/tests/test_api.py
git commit -m "feat(G3): add FastAPI backend with SSE build, query, dataset routes"
```

---

## Task 8 — Frontend `app/static/`

**Files:**
- Create: `app/static/style.css`
- Create: `app/static/app.js`
- Create: `app/static/index.html`

- [ ] **Step 1: Créer `app/static/style.css`**

```css
/* app/static/style.css */
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,sans-serif;background:#f0f4ff;color:#1a1f36;height:100vh;overflow:hidden}
.glow{position:fixed;pointer-events:none;border-radius:50%;filter:blur(80px);opacity:.28}
.glow-1{width:420px;height:420px;background:#c7d2fe;top:-110px;right:-90px}
.glow-2{width:300px;height:300px;background:#ddd6fe;bottom:50px;left:320px}

/* Nav */
.nav{background:rgba(255,255,255,.85);backdrop-filter:blur(12px);border-bottom:1px solid rgba(99,102,241,.12);padding:13px 26px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:20;box-shadow:0 1px 20px rgba(99,102,241,.07)}
.nav-logo{font-size:17px;font-weight:700;background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-badge{font-size:11px;background:linear-gradient(135deg,#e0e7ff,#ede9fe);color:#6366f1;border-radius:20px;padding:3px 10px;font-weight:600;border:1px solid rgba(99,102,241,.2)}
.nav-steps{margin-left:auto;display:flex;gap:5px}
.nav-step{font-size:11px;padding:4px 13px;border-radius:20px;font-weight:500;transition:all .2s}
.nav-step.active{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;box-shadow:0 2px 10px rgba(99,102,241,.35)}
.nav-step.done{background:#e0e7ff;color:#6366f1}
.nav-step.pending{color:#94a3b8;border:1px solid #e2e8f0}

/* Layout */
.app{display:grid;grid-template-columns:310px 1fr;height:calc(100vh - 54px)}

/* Sidebar */
.sidebar{background:#fff;border-right:1px solid rgba(99,102,241,.1);display:flex;flex-direction:column;overflow:hidden}
.upload-zone{margin:13px;border:2px dashed rgba(99,102,241,.3);border-radius:13px;padding:16px 12px;text-align:center;background:linear-gradient(135deg,#f5f3ff,#eef2ff);position:relative;overflow:hidden;cursor:pointer}
.upload-zone::before{content:'';position:absolute;top:-40px;left:-40px;width:110px;height:110px;background:radial-gradient(circle,rgba(99,102,241,.15) 0%,transparent 70%);border-radius:50%}
.upload-zone input{display:none}
.upload-icon{font-size:24px;margin-bottom:4px}
.upload-title{font-size:13px;font-weight:600;color:#6366f1}
.upload-sub{font-size:11px;color:#94a3b8;margin-top:2px}
.upload-btn{display:inline-block;margin-top:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-size:12px;font-weight:600;padding:6px 15px;border-radius:20px;box-shadow:0 3px 10px rgba(99,102,241,.3);cursor:pointer}
.dataset-toggle{margin:0 13px 10px;display:flex;background:#f0f4ff;border-radius:10px;padding:3px;gap:3px}
.ds-btn{flex:1;font-size:11px;font-weight:600;padding:6px;border-radius:8px;text-align:center;cursor:pointer;color:#94a3b8;border:none;background:transparent;transition:all .15s}
.ds-btn.active{background:#fff;color:#6366f1;box-shadow:0 1px 6px rgba(99,102,241,.15)}
.docs-list{padding:0 13px;flex:1;overflow-y:auto}
.docs-label{font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px}
.doc-item{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:10px;background:#f8faff;margin-bottom:5px;border:1px solid rgba(99,102,241,.08)}
.doc-name{font-size:12px;font-weight:600;color:#1a1f36}
.doc-meta{font-size:10px;color:#94a3b8;margin-top:1px}
.doc-status-ok{font-size:10px;padding:2px 8px;border-radius:20px;font-weight:600;background:#d1fae5;color:#059669}
.sidebar-actions{padding:13px;display:flex;flex-direction:column;gap:7px}
.build-btn{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-size:13px;font-weight:700;padding:11px;border-radius:12px;text-align:center;box-shadow:0 4px 18px rgba(99,102,241,.35);cursor:pointer;border:none;width:100%}
.build-btn:disabled{opacity:.5;cursor:not-allowed}
.reset-btn{display:none;font-size:12px;color:#94a3b8;background:transparent;border:1px solid #e2e8f0;border-radius:10px;padding:7px;cursor:pointer;width:100%}
.reset-btn.visible{display:block}

/* Progress */
.progress-box{padding:0 13px 10px}
.progress-label{font-size:11px;color:#64748b;display:flex;justify-content:space-between;margin-bottom:4px}
.progress-pct{color:#6366f1;font-weight:700}
.progress-track{background:#f0f4ff;border-radius:20px;height:6px;overflow:hidden}
.progress-fill{height:100%;border-radius:20px;background:linear-gradient(90deg,#6366f1,#8b5cf6);transition:width .3s}
.progress-steps{margin-top:8px;display:flex;flex-direction:column;gap:3px}
.prog-step{font-size:11px;display:flex;align-items:center;gap:6px;color:#cbd5e1}
.prog-step.done{color:#059669}.prog-step.active-s{color:#6366f1;font-weight:600}.prog-dot{width:6px;height:6px;border-radius:50%;background:#e2e8f0;flex-shrink:0}
.prog-step.done .prog-dot{background:#22c55e}
.prog-step.active-s .prog-dot{background:#6366f1;animation:blink 1s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* Right panel */
.right{display:flex;flex-direction:column;overflow:hidden}
.tab-bar{background:#fff;border-bottom:1px solid rgba(99,102,241,.1);padding:0 20px;display:flex;box-shadow:0 1px 10px rgba(99,102,241,.05)}
.tab{font-size:13px;font-weight:600;padding:13px 18px;border-bottom:2.5px solid transparent;color:#94a3b8;cursor:pointer;transition:all .15s}
.tab.active{color:#6366f1;border-bottom-color:#6366f1}
.tab-content{display:none;flex:1;flex-direction:column;overflow:hidden}
.tab-content.active{display:flex}

/* Chat */
.chat-area{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:11px;background:#f8faff}
.msg{max-width:78%;padding:11px 14px;border-radius:15px;font-size:13px;line-height:1.55}
.msg-user{align-self:flex-end;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border-bottom-right-radius:4px;box-shadow:0 3px 10px rgba(99,102,241,.25)}
.msg-bot{align-self:flex-start;background:#fff;border:1px solid rgba(99,102,241,.1);border-bottom-left-radius:4px;box-shadow:0 2px 10px rgba(99,102,241,.06)}
.msg-bot-hdr{font-size:10px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;background:linear-gradient(135deg,#e0e7ff,#ede9fe);color:#6366f1;border-radius:20px;padding:4px 11px;font-weight:600;margin-top:8px;border:1px solid rgba(99,102,241,.18);cursor:pointer;transition:all .15s;user-select:none}
.chip:hover{background:linear-gradient(135deg,#c7d2fe,#ddd6fe)}
.chip.open{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border-color:transparent}
.drawer{margin-top:9px;background:#f5f3ff;border:1px solid rgba(99,102,241,.15);border-radius:11px;overflow:hidden;max-height:0;opacity:0;transition:max-height .35s cubic-bezier(.4,0,.2,1),opacity .25s}
.drawer.open{max-height:320px;opacity:1}
.drawer-inner{padding:13px 15px}
.drawer-label{font-size:10px;font-weight:700;color:#8b5cf6;text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px}
.step-row{display:flex;gap:9px;margin-bottom:6px}
.step-dot-n{width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;font-size:9px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.step-title{font-size:12px;font-weight:600;color:#1a1f36}
.step-desc{font-size:11px;color:#64748b;margin-top:1px}
.etags{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.etag{font-size:10px;background:#fff;border:1px solid rgba(99,102,241,.2);color:#6366f1;border-radius:6px;padding:2px 8px}
.docs-used-list{display:flex;flex-direction:column;gap:4px;margin-top:4px}
.doc-used-item{display:flex;align-items:center;gap:7px;background:#fff;border-radius:8px;padding:6px 10px;border:1px solid rgba(99,102,241,.1);font-size:11px}
.input-bar{padding:13px 20px;background:#fff;border-top:1px solid rgba(99,102,241,.08);display:flex;gap:9px;align-items:center;box-shadow:0 -4px 18px rgba(99,102,241,.04)}
.input-wrap{flex:1;position:relative}
.input-badge{position:absolute;top:-11px;left:50%;transform:translateX(-50%);font-size:10px;font-weight:600;color:#94a3b8;background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:2px 10px;white-space:nowrap;display:none}
.input-badge.visible{display:block}
.chat-input{width:100%;background:#f0f4ff;border:1.5px solid rgba(99,102,241,.15);border-radius:11px;padding:10px 14px;font-size:13px;color:#1a1f36;outline:none;font-family:inherit}
.chat-input:disabled{background:#f8fafc;border-color:#e2e8f0;color:#cbd5e1;cursor:not-allowed}
.send-btn{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;border-radius:11px;padding:10px 17px;font-size:13px;font-weight:600;cursor:pointer;box-shadow:0 3px 10px rgba(99,102,241,.3);font-family:inherit}
.send-btn:disabled{background:#f1f5f9;color:#cbd5e1;box-shadow:none;cursor:not-allowed}

/* Graph tab */
.graph-panel{flex:1;display:flex;flex-direction:column;overflow:hidden;background:#fff}
.graph-toolbar{padding:11px 20px;border-bottom:1px solid rgba(99,102,241,.08);display:flex;align-items:center;gap:10px;flex-shrink:0}
.g-stat{font-size:12px;color:#64748b}.g-stat strong{color:#1a1f36}
.g-sep{width:1px;height:14px;background:#e2e8f0}
.live-dot{width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.2);animation:blink 1.2s ease-in-out infinite}
.comm-filters{margin-left:auto;display:flex;gap:5px;flex-wrap:wrap}
.comm-badge{font-size:11px;padding:3px 10px;border-radius:20px;font-weight:600;cursor:pointer;border:1.5px solid rgba(99,102,241,.25);background:#eef2ff;color:#6366f1}
#graph-svg{flex:1}
.tooltip{position:absolute;background:#fff;border:1px solid rgba(99,102,241,.15);border-radius:10px;padding:10px 14px;box-shadow:0 4px 18px rgba(99,102,241,.12);font-size:12px;pointer-events:none;display:none;z-index:100;min-width:160px}
.tt-name{font-weight:700;color:#1a1f36;margin-bottom:3px}
.tt-type{font-size:10px;color:#6366f1;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
.tt-rel{font-size:11px;color:#64748b;margin-top:2px}
```

- [ ] **Step 2: Créer `app/static/app.js`**

```javascript
// app/static/app.js
const API = '';
let appState = 'upload'; // 'upload' | 'building' | 'ready'
let lastSubgraphNodes = new Set();
let lastSubgraphEdges = new Set();
let simulation = null;
let graphData = null;

// ── State machine ──
function setAppState(s) {
  appState = s;
  const steps = document.querySelectorAll('.nav-step');
  const stateMap = { upload: 0, building: 1, ready: 2 };
  steps.forEach((el, i) => {
    el.className = 'nav-step ' + (i < stateMap[s] ? 'done' : i === stateMap[s] ? 'active' : 'pending');
  });
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const badge = document.getElementById('input-badge');
  const buildBtn = document.getElementById('build-btn');
  if (s === 'ready') {
    input.disabled = false; sendBtn.disabled = false;
    badge.classList.remove('visible'); buildBtn.textContent = '✓ Knowledge Graph construit';
    buildBtn.disabled = true;
    document.getElementById('progress-box').style.display = 'none';
    loadGraph();
  } else if (s === 'building') {
    input.disabled = true; sendBtn.disabled = true;
    badge.textContent = '⚙ Construction du KG en cours…'; badge.classList.add('visible');
    buildBtn.disabled = true;
  } else {
    input.disabled = true; sendBtn.disabled = true;
    badge.textContent = '⏳ Uploadez vos documents d\'abord'; badge.classList.add('visible');
    buildBtn.disabled = false;
  }
}

// ── Upload ──
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#6366f1'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.style.borderColor = ''; handleFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const r = await fetch(`${API}/upload`, { method: 'POST', body: fd });
  const { saved } = await r.json();
  saved.forEach(addDocItem);
  document.getElementById('build-btn').disabled = false;
}

function addDocItem(name) {
  const list = document.getElementById('docs-list');
  const el = document.createElement('div');
  el.className = 'doc-item';
  el.innerHTML = `<span style="font-size:16px">📄</span><div style="flex:1"><div class="doc-name">${name}</div></div><span class="doc-status-ok">✓</span>`;
  list.appendChild(el);
}

// ── Dataset toggle ──
document.querySelectorAll('.ds-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    document.querySelectorAll('.ds-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const ds = btn.dataset.ds;
    await fetch(`${API}/dataset/select`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset: ds }) });
    document.getElementById('reset-btn').classList.toggle('visible', ds === 'custom');
    // Refresh doc list for hotpotqa
    if (ds === 'hotpotqa') {
      const { datasets } = await (await fetch(`${API}/datasets`)).json();
      document.getElementById('docs-list').innerHTML = '';
      datasets.hotpotqa.files.forEach(addDocItem);
      if (datasets.hotpotqa.files.length > 0) document.getElementById('build-btn').disabled = false;
    }
    setAppState('upload');
  });
});

// ── Reset ──
document.getElementById('reset-btn').addEventListener('click', async () => {
  await fetch(`${API}/reset`, { method: 'POST' });
  document.getElementById('docs-list').innerHTML = '';
  document.getElementById('build-btn').disabled = true;
  setAppState('upload');
});

// ── Build ──
const STAGES = ['extraction', 'graph_build', 'community_detection', 'indexing'];
document.getElementById('build-btn').addEventListener('click', async () => {
  setAppState('building');
  document.getElementById('progress-box').style.display = 'block';
  const fill = document.getElementById('progress-fill');
  const pct = document.getElementById('progress-pct');
  const label = document.getElementById('progress-label-text');

  const es = new EventSource(`${API}/build`);
  es.onmessage = e => {
    const { stage, progress, message } = JSON.parse(e.data);
    fill.style.width = `${progress}%`;
    pct.textContent = `${progress}%`;
    label.textContent = message;
    STAGES.forEach(s => {
      const el = document.getElementById(`prog-${s}`);
      if (!el) return;
      const stageIdx = STAGES.indexOf(stage);
      const elIdx = STAGES.indexOf(s);
      el.className = 'prog-step ' + (elIdx < stageIdx ? 'done' : elIdx === stageIdx ? 'active-s' : '');
    });
    if (stage === 'done') { es.close(); setAppState('ready'); }
    if (stage === 'error') { es.close(); alert('Erreur: ' + message); setAppState('upload'); }
  };
});

// ── Tabs ──
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ── Chat ──
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  appendMessage('user', q);
  const typingId = appendMessage('bot', '…');
  try {
    const r = await fetch(`${API}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await r.json();
    updateMessage(typingId, data);
    highlightSubgraph(data.subgraph_nodes, data.subgraph_edges);
  } catch (err) {
    updateMessage(typingId, null, 'Erreur lors de la requête.');
  }
}

let msgId = 0;
function appendMessage(type, text) {
  const area = document.getElementById('chat-area');
  const id = `msg-${++msgId}`;
  const div = document.createElement('div');
  div.id = id;
  div.className = `msg msg-${type === 'user' ? 'user' : 'bot'}`;
  if (type === 'bot') div.innerHTML = `<div class="msg-bot-hdr">✦ GraphRAG</div><span class="msg-text">${text}</span>`;
  else div.textContent = text;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  return id;
}

function updateMessage(id, data, errorText) {
  const el = document.getElementById(id);
  if (!el) return;
  if (errorText) { el.querySelector('.msg-text').textContent = errorText; return; }
  el.querySelector('.msg-text').textContent = data.answer;
  // Build chip + drawer
  const hopCount = data.trace.length;
  const nodeCount = data.subgraph_nodes.length;
  const docCount = data.docs_used.length;
  const chipId = `chip-${id}`;
  const drawerId = `drawer-${id}`;
  const chip = document.createElement('div');
  chip.innerHTML = `
    <span class="chip" id="${chipId}" onclick="toggleDrawer('${chipId}','${drawerId}')">
      ⚡ ${hopCount}-hop · ${nodeCount} entités · ${docCount} docs — voir le détail ↓
    </span>
    <div class="drawer" id="${drawerId}">
      <div class="drawer-inner">
        <div class="drawer-label">🔍 Étapes de raisonnement</div>
        ${data.trace.map(s => `
          <div class="step-row">
            <div class="step-dot-n">${s.hop}</div>
            <div>
              <div class="step-title">Hop ${s.hop} — ${s.relation}</div>
              <div class="step-desc">Depuis : ${s.from_node}</div>
              <div class="etags">${s.to_nodes.map(n => `<span class="etag">${n}</span>`).join('')}</div>
            </div>
          </div>`).join('')}
        <div class="drawer-label" style="margin-top:10px">📄 Documents sources</div>
        <div class="docs-used-list">
          ${data.docs_used.length ? data.docs_used.map(d => `<div class="doc-used-item">📄 <strong>${d.filename}</strong></div>`).join('') : '<div style="font-size:11px;color:#94a3b8">Aucun document tracé</div>'}
        </div>
      </div>
    </div>`;
  el.appendChild(chip);
}

function toggleDrawer(chipId, drawerId) {
  const chip = document.getElementById(chipId);
  const drawer = document.getElementById(drawerId);
  const isOpen = drawer.classList.contains('open');
  drawer.classList.toggle('open');
  chip.classList.toggle('open');
  const base = chip.textContent.replace(/[↑↓]/g, '').trim();
  chip.textContent = base + (isOpen ? ' ↓' : ' ↑');
}

// ── D3 Graph ──
async function loadGraph() {
  try {
    const data = await (await fetch(`${API}/graph`)).json();
    graphData = data;
    renderGraph(data);
    renderCommunityFilters(data.communities);
    document.getElementById('stat-nodes').textContent = data.stats.node_count;
    document.getElementById('stat-edges').textContent = data.stats.edge_count;
    document.getElementById('stat-comms').textContent = data.stats.community_count;
  } catch (_) {}
}

function renderGraph(data) {
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();
  const W = document.getElementById('graph-svg').clientWidth;
  const H = document.getElementById('graph-svg').clientHeight;
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', e => g.attr('transform', e.transform)));

  svg.append('defs').append('marker')
    .attr('id', 'arr').attr('markerWidth', 6).attr('markerHeight', 6).attr('refX', 14).attr('refY', 3).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', '#c7d2fe');

  const link = g.append('g').selectAll('line').data(data.edges).enter().append('line')
    .attr('stroke', '#ddd6fe').attr('stroke-width', 1.5).attr('marker-end', 'url(#arr)')
    .attr('class', d => `edge-${d.source}-${d.target}`);

  const node = g.append('g').selectAll('g').data(data.nodes).enter().append('g')
    .attr('class', 'node').call(d3.drag().on('start', dragstart).on('drag', dragged).on('end', dragend));

  node.append('circle').attr('r', d => 8 + Math.min(d.degree * 2, 14))
    .attr('fill', '#fff').attr('stroke', d => d.color).attr('stroke-width', 2.5)
    .style('filter', d => `drop-shadow(0 1px 6px ${d.color}55)`);

  node.append('text').text(d => d.id).attr('text-anchor', 'middle').attr('dy', 4)
    .attr('font-size', 9).attr('fill', d => d.color).attr('font-weight', '600')
    .style('pointer-events', 'none');

  const tooltip = document.getElementById('graph-tooltip');
  node.on('mouseover', (event, d) => {
    const rels = data.edges.filter(e => e.source === d.id || e.target === d.id)
      .map(e => `<div class="tt-rel">${e.source} →[${e.relation}]→ ${e.target}</div>`).join('');
    tooltip.innerHTML = `<div class="tt-name">${d.id}</div><div class="tt-type">Communauté ${d.community + 1}</div>${rels}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (event.pageX + 12) + 'px';
    tooltip.style.top = (event.pageY - 20) + 'px';
  }).on('mousemove', event => {
    tooltip.style.left = (event.pageX + 12) + 'px';
    tooltip.style.top = (event.pageY - 20) + 'px';
  }).on('mouseout', () => { tooltip.style.display = 'none'; });

  simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.edges).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function highlightSubgraph(nodes, edges) {
  if (!graphData) return;
  const nodeSet = new Set(nodes);
  const edgeSet = new Set(edges.map(e => `${e[0]}-${e[2]}`));
  d3.selectAll('.node circle').attr('stroke', d => nodeSet.has(d.id) ? '#f59e0b' : d.color)
    .attr('stroke-width', d => nodeSet.has(d.id) ? 3 : 2.5);
  d3.selectAll('line').attr('stroke', d => edgeSet.has(`${d.source.id || d.source}-${d.target.id || d.target}`) ? '#f59e0b' : '#ddd6fe')
    .attr('stroke-width', d => edgeSet.has(`${d.source.id || d.source}-${d.target.id || d.target}`) ? 2.5 : 1.5);
}

function renderCommunityFilters(communities) {
  const COLORS = ['#6366f1','#8b5cf6','#f59e0b','#10b981','#ef4444','#3b82f6'];
  const container = document.getElementById('comm-filters');
  container.innerHTML = '';
  communities.forEach((c, i) => {
    const btn = document.createElement('span');
    btn.className = 'comm-badge';
    btn.style.background = COLORS[i % COLORS.length] + '22';
    btn.style.color = COLORS[i % COLORS.length];
    btn.style.borderColor = COLORS[i % COLORS.length] + '44';
    btn.textContent = `● ${c.label}`;
    container.appendChild(btn);
  });
}

function dragstart(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
function dragend(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

// Init
setAppState('upload');
```

- [ ] **Step 3: Créer `app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⬡ GraphRAG — EPITA 2026</title>
<link rel="stylesheet" href="/style.css">
<script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
<div class="glow glow-1"></div>
<div class="glow glow-2"></div>

<nav class="nav">
  <span class="nav-logo">⬡ GraphRAG</span>
  <span class="nav-badge">EPITA 2026</span>
  <div class="nav-steps">
    <span class="nav-step active">① Upload</span>
    <span class="nav-step pending">② Build KG</span>
    <span class="nav-step pending">③ Query</span>
  </div>
</nav>

<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div id="drop-zone" class="upload-zone">
      <input type="file" id="file-input" multiple accept=".pdf,.txt,.md">
      <div class="upload-icon">📄</div>
      <div class="upload-title">Glissez vos documents ici</div>
      <div class="upload-sub">PDF, TXT, Markdown</div>
      <div class="upload-btn">+ Ajouter des fichiers</div>
    </div>

    <div class="dataset-toggle">
      <button class="ds-btn active" data-ds="custom">Corpus custom</button>
      <button class="ds-btn" data-ds="hotpotqa">HotpotQA</button>
    </div>

    <div class="docs-list" style="padding:0 13px;flex:1;overflow-y:auto">
      <div class="docs-label">Documents actifs</div>
      <div id="docs-list"></div>
    </div>

    <div id="progress-box" class="progress-box" style="display:none">
      <div class="progress-label">
        <span id="progress-label-text">Initialisation…</span>
        <span class="progress-pct" id="progress-pct">0%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div class="progress-steps">
        <div class="prog-step" id="prog-extraction"><div class="prog-dot"></div>Extraction des entités</div>
        <div class="prog-step" id="prog-graph_build"><div class="prog-dot"></div>Construction du graphe RDF</div>
        <div class="prog-step" id="prog-community_detection"><div class="prog-dot"></div>Détection de communautés (Leiden)</div>
        <div class="prog-step" id="prog-indexing"><div class="prog-dot"></div>Indexation</div>
      </div>
    </div>

    <div class="sidebar-actions">
      <button class="build-btn" id="build-btn" disabled>⚡ Construire le Knowledge Graph</button>
      <button class="reset-btn" id="reset-btn">↺ Réinitialiser le corpus</button>
    </div>
  </aside>

  <!-- Right panel -->
  <div class="right">
    <div class="tab-bar">
      <div class="tab active" data-tab="qa">💬 Q&amp;A</div>
      <div class="tab" data-tab="graph">🕸 Graphe de connaissances</div>
    </div>

    <!-- Q&A Tab -->
    <div class="tab-content active" id="tab-qa">
      <div class="chat-area" id="chat-area">
        <div class="msg msg-bot">
          <div class="msg-bot-hdr">✦ GraphRAG</div>
          <span class="msg-text">Bienvenue ! Uploadez des documents et construisez le Knowledge Graph pour commencer.</span>
        </div>
      </div>
      <div class="input-bar">
        <div class="input-wrap">
          <div class="input-badge visible" id="input-badge">⏳ Uploadez vos documents d'abord</div>
          <input class="chat-input" id="chat-input" placeholder="Posez une question sur vos documents…" disabled>
        </div>
        <button class="send-btn" id="send-btn" disabled>Envoyer →</button>
      </div>
    </div>

    <!-- Graph Tab -->
    <div class="tab-content" id="tab-graph">
      <div class="graph-panel">
        <div class="graph-toolbar">
          <div class="live-dot"></div>
          <div class="g-stat"><strong id="stat-nodes">—</strong> entités</div>
          <div class="g-sep"></div>
          <div class="g-stat"><strong id="stat-edges">—</strong> relations</div>
          <div class="g-sep"></div>
          <div class="g-stat"><strong id="stat-comms">—</strong> communautés</div>
          <div class="comm-filters" id="comm-filters"></div>
        </div>
        <svg id="graph-svg" style="flex:1;width:100%"></svg>
      </div>
    </div>
  </div>
</div>

<div class="tooltip" id="graph-tooltip"></div>
<script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Tester manuellement l'app**

```bash
cd groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG
cp .env.example .env  # puis remplir les vraies clés
uvicorn app.main:app --reload --port 8000
```

Ouvrir http://localhost:8000, vérifier :
- Upload d'un fichier TXT → apparaît dans la liste
- Clic "Construire le KG" → progress SSE s'affiche étape par étape
- Input activé après build
- Question dans Q&A → réponse + chip → clic chip → drawer avec trace
- Tab Graphe → nœuds D3 visibles, tooltip au survol

- [ ] **Step 5: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/app/static/
git commit -m "feat(G3): add frontend — sidebar, Q&A chat, D3 graph, SSE progress"
```

---

## Task 9 — Données HotpotQA

**Files:**
- Create: `data/download_hotpotqa.py`

- [ ] **Step 1: Créer `data/download_hotpotqa.py`**

```python
# data/download_hotpotqa.py
"""
Télécharge un subset HotpotQA (500 exemples) et l'écrit dans data/hotpotqa/.
Usage: python data/download_hotpotqa.py
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).parent / "hotpotqa"
OUT.mkdir(exist_ok=True)

ds = load_dataset("hotpot_qa", "fullwiki", split="validation", streaming=True)
samples = []
for i, row in enumerate(ds):
    if i >= 500:
        break
    samples.append(row)

# Écrire les contextes comme fichiers texte
for i, sample in enumerate(samples):
    context_parts = []
    for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
        context_parts.append(f"# {title}\n" + " ".join(sentences))
    text = "\n\n".join(context_parts)
    (OUT / f"sample_{i:04d}.txt").write_text(text, encoding="utf-8")

# Écrire les questions/réponses pour le benchmark
qa_pairs = [{"question": s["question"], "answer": s["answer"]} for s in samples]
(OUT / "qa_pairs.json").write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✓ {len(samples)} samples écrits dans {OUT}")
```

- [ ] **Step 2: Exécuter le script**

```bash
python data/download_hotpotqa.py
```
Résultat attendu : `✓ 500 samples écrits dans data/hotpotqa`

- [ ] **Step 3: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/data/download_hotpotqa.py
git commit -m "feat(G3): add HotpotQA download script (500 samples)"
```

---

## Task 10 — `notebooks/benchmark.ipynb`

**Files:**
- Create: `notebooks/benchmark.ipynb`

- [ ] **Step 1: Créer le notebook benchmark**

Créer `notebooks/benchmark.ipynb` avec les cellules suivantes dans l'ordre :

**Cellule 1 — titre (markdown)**
```markdown
# Benchmark GraphRAG vs RAG Vectoriel — G3 EPITA 2026

Comparaison sur un subset HotpotQA (500 questions multi-hop).
Métriques : F1 token-level, Exact Match.
```

**Cellule 2 — imports**
```python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))
from dotenv import load_dotenv
load_dotenv("../.env")

import numpy as np
import matplotlib.pyplot as plt
from graphrag_core.llm import get_llm_client
from graphrag_core.extractor import extract_triples
from graphrag_core.graph import build_knowledge_graph, KnowledgeGraph
from graphrag_core.pipeline import run as graphrag_run
```

**Cellule 3 — chargement données**
```python
DATA = Path("../data/hotpotqa")
qa_pairs = json.loads((DATA / "qa_pairs.json").read_text())
sample_files = sorted(DATA.glob("sample_*.txt"))
docs_map = {f.name: f.read_text(encoding="utf-8") for f in sample_files[:50]}
print(f"{len(qa_pairs)} questions, {len(docs_map)} documents chargés")
```

**Cellule 4 — construction du KG**
```python
llm = get_llm_client()
all_triples = []
for fname, text in docs_map.items():
    all_triples.extend(extract_triples(text, llm))
    print(f"  {fname}: {len(all_triples)} triplets total")
kg = build_knowledge_graph(all_triples, docs_map)
print(f"\nKG: {kg.nx_graph.number_of_nodes()} entités, {kg.nx_graph.number_of_edges()} relations, {len(kg.communities)} communautés")
```

**Cellule 5 — fonctions de scoring**
```python
def f1_score(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)

def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())
```

**Cellule 6 — RAG vectoriel baseline**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = list(docs_map.values())
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(corpus)

def rag_vectoriel(question: str, top_k: int = 3) -> str:
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    context = "\n\n".join(corpus[i][:500] for i in top_indices)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"
    return llm.complete(prompt)
```

**Cellule 7 — benchmark sur 50 questions**
```python
N = 50
questions = qa_pairs[:N]

results = {"graphrag": {"f1": [], "em": []}, "rag_v": {"f1": [], "em": []}}

for i, qa in enumerate(questions):
    q, gold = qa["question"], qa["answer"]
    # GraphRAG
    gr = graphrag_run(q, kg, llm, docs_map)
    results["graphrag"]["f1"].append(f1_score(gr.answer, gold))
    results["graphrag"]["em"].append(exact_match(gr.answer, gold))
    # RAG vectoriel
    rv = rag_vectoriel(q)
    results["rag_v"]["f1"].append(f1_score(rv, gold))
    results["rag_v"]["em"].append(exact_match(rv, gold))
    if i % 10 == 0:
        print(f"  {i}/{N} questions traitées")

print("\n=== Résultats ===")
for method, scores in results.items():
    print(f"{method}: F1={np.mean(scores['f1']):.3f}, EM={np.mean(scores['em']):.3f}")
```

**Cellule 8 — visualisation**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
methods = ["GraphRAG", "RAG Vectoriel"]
f1_means = [np.mean(results["graphrag"]["f1"]), np.mean(results["rag_v"]["f1"])]
em_means = [np.mean(results["graphrag"]["em"]), np.mean(results["rag_v"]["em"])]
colors = ["#6366f1", "#94a3b8"]

axes[0].bar(methods, f1_means, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("F1 Score (token-level)", fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("F1")

axes[1].bar(methods, em_means, color=colors, edgecolor="white", linewidth=1.5)
axes[1].set_title("Exact Match", fontweight="bold")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("EM")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8faff")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("GraphRAG vs RAG Vectoriel — HotpotQA (50 questions)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../docs/benchmark_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cellule 9 — analyse des hops**
```python
hop_counts = [len(r.trace) for r in [graphrag_run(qa["question"], kg, llm, docs_map) for qa in questions[:20]]]
plt.figure(figsize=(6, 3))
plt.hist(hop_counts, bins=range(0, max(hop_counts)+2), color="#6366f1", alpha=0.85, edgecolor="white")
plt.title("Distribution du nombre de hops par réponse GraphRAG")
plt.xlabel("Nombre de hops")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
```

Pour créer le notebook `.ipynb`, utiliser `jupytext` ou Jupyter Lab pour créer le fichier avec ces cellules.

- [ ] **Step 2: Commit**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/notebooks/benchmark.ipynb
git commit -m "feat(G3): add benchmark notebook — GraphRAG vs RAG vectoriel on HotpotQA"
```

---

## Task 11 — Suite des tests et vérification finale

- [ ] **Step 1: Lancer tous les tests**

```bash
cd groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG
pytest tests/ -v --tb=short
```
Résultat attendu : tous les tests passent (llm, extractor, graph, retriever, pipeline, api)

- [ ] **Step 2: Vérifier que l'app tourne**

```bash
uvicorn app.main:app --port 8000
```
Aller sur http://localhost:8000 — tester le flux complet (upload → build → query → graph tab).

- [ ] **Step 3: Commit final**

```bash
git add groupe-G3-GraphRAG-combine_Knowledge_Graphs_et_LLM_pour_le_RAG/
git commit -m "chore(G3): final integration check — all tests pass"
```

---

## Checklist de couverture spec

| Exigence spec | Tâche |
|---|---|
| `graphrag_core/llm.py` multi-provider | Task 2 |
| `graphrag_core/extractor.py` | Task 3 |
| `graphrag_core/graph.py` + Leiden | Task 4 |
| `graphrag_core/retriever.py` BFS + trace | Task 5 |
| `graphrag_core/pipeline.py` | Task 6 |
| `.env` unique | Task 1 |
| `app/main.py` routes + SSE | Task 7 |
| Frontend sidebar + tabs | Task 8 |
| Input désactivé états ①② | Task 8 (app.js `setAppState`) |
| Chip + drawer de détail | Task 8 (app.js `updateMessage`) |
| Toggle HotpotQA / custom | Task 8 (app.js dataset toggle) |
| Bouton Réinitialiser (custom seulement) | Task 8 (app.js reset-btn) |
| D3.js graphe force-directed | Task 8 (app.js `renderGraph`) |
| Highlight sous-graphe en ambre | Task 8 (app.js `highlightSubgraph`) |
| Download HotpotQA | Task 9 |
| `benchmark.ipynb` | Task 10 |
| Notebooks déplacés dans `notebooks/` | Task 1 |
