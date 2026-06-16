# G3 — GraphRAG : Visualisation cliquable, résolution d'entités par embeddings, retriever hybride

**Projet** : EPITA 2026 — Intelligence Symbolique
**Groupe** : Paul Witkowski & Matteo Atkinson
**Sujet** : Combine Knowledge Graphs et LLM pour le RAG
**Date** : 2026-06-16

---

## 1. Contexte et problème

Trois limitations identifiées dans l'implémentation actuelle (`docs/2026-06-13-G3-graphrag-design.md`) :

1. **Visualisation illisible et qui lague.** Le snapshot pré-construit `data/hotpotqa.graphrag` contient **30 770 nœuds uniques et 39 606 relations**. `loadGraph()` (`app/static/app.js`) récupère la totalité du graphe via `GET /graph` et lance une simulation D3 force-directed sur l'ensemble — ingérable dans un navigateur à cette échelle, peu importe le style CSS. Le snapshot `ai_history.graphrag` (39 nœuds) s'affiche correctement, confirmant que le problème est l'échelle, pas le rendu.
2. **Aucune résolution d'entités.** `extractor.extract_triples` ne déduplique que des triplets identiques au caractère près (`Triple` exact match). Deux mentions de la même entité sous des formes différentes ("Obama" / "Barack Obama") deviennent deux nœuds distincts dans `graph.build_knowledge_graph`, sans aucune vérification de similarité avant création.
3. **Retriever limité au mot-clé.** `retriever.detect_entities` ne trouve une entité de départ (seed) que si son nom exact apparaît textuellement dans la question (regex whole-word). Aucune capacité à rapprocher une question paraphrasée ("le réalisateur de Sinister") d'une entité nommée différemment dans le graphe ("Scott Derrickson") — un cas pourtant central dans les questions multi-hop de HotpotQA.

Ce document couvre les trois correctifs. (B) est un prérequis technique de (C) — les deux partagent la même infrastructure d'embeddings. (A) est indépendant.

---

## 2. Visualisation du graphe — vue communautés cliquable

### 2.1 Principe

On ne rend plus jamais "tout le graphe" d'un coup. La navigation se fait par drill-down progressif, en s'appuyant sur les communautés déjà calculées par Leiden (`graph._detect_communities`, existant, inchangé) :

- **Vue d'ensemble** : une bulle par communauté (taille proportionnelle au nombre de membres). Payload minuscule, indépendant de la taille du graphe.
- **Vue détail** : clic sur une bulle → charge les nœuds de cette communauté (plafonnés, top-N par degré).
- **Exploration locale** : clic sur un nœud déjà affiché → charge ses voisins immédiats et les **ajoute** à la vue en cours (le graphe grossit par exploration, comme Obsidian/Roam ; rien ne disparaît, les nœuds déjà positionnés ne sont pas perturbés).
- **Focus Q&A** : si une question vient d'être posée, l'onglet Graphe affiche directement le sous-graphe de la réponse (déjà ≤150 nœuds via le pipeline existant) plutôt que la vue d'ensemble — c'est déjà la vue la plus utile dans ce cas.

### 2.2 Backend — nouvelles routes (`app/main.py`)

`GET /graph` est remplacé par trois routes :

| Méthode | Route | Réponse |
|---------|-------|---------|
| `GET` | `/graph/overview` | `{ communities: [{id, label, size, color}], stats: {node_count, edge_count, community_count} }` — aucune liste de nœuds, payload constant quelle que soit la taille du graphe. |
| `GET` | `/graph/community/{id}?limit=150` | `{ nodes: [...], edges: [...], truncated: bool, total_in_community: int }` — nœuds de la communauté triés par degré décroissant, tronqués à `limit` (défaut 150, même valeur que `_HUB_DEGREE_THRESHOLD`/`max_nodes` du retriever pour cohérence). `truncated=true` si `total_in_community > limit` ; le frontend peut alors rappeler la même route avec `limit` plus grand pour "charger plus". |
| `GET` | `/graph/node/{name}/neighbors?limit=40` | `{ nodes: [...], edges: [...], truncated: bool }` — voisins à 1 hop du nœud (le nœud interrogé n'est pas réinclus), tronqués si c'est un hub à fort degré. `edges` ne contient que les arêtes entre le nœud interrogé et chacun de ses voisins (pas les arêtes entre voisins eux-mêmes). |

Toutes les routes lèvent `404` si `state["kg"] is None`, comme l'actuel `GET /graph`.

Nouvelles fonctions dans `graph.py` pour servir ces routes (équivalent de `graph_to_json` mais filtré) :
- `community_overview(kg) -> dict`
- `community_detail(kg, community_id, limit) -> dict`
- `node_neighbors(kg, node_name, limit) -> dict`

### 2.3 Frontend (`app/static/app.js`)

Nouvel état `viewMode: 'overview' | 'detail'` (en plus de `appState` existant).

- `loadGraph()` est remplacé par `loadGraphOverview()` qui appelle `/graph/overview` et rend des bulles SVG (une par communauté, taille ∝ `size`, clic → `openCommunity(id)`).
- `openCommunity(id)` : fetch `/graph/community/{id}`, remplace le contenu du SVG par le rendu détail (réutilise la logique D3 force-directed existante, mais sur un sous-ensemble borné).
- Clic sur un nœud en vue détail → `expandNode(name)` : fetch `/graph/node/{name}/neighbors`, fusionne les nouveaux nœuds (dédoublonnage par id) et arêtes (dédoublonnage par paire source-target) dans `graphData`, puis redémarre la simulation avec `alpha(0.5)` — les nœuds déjà positionnés gardent leurs coordonnées (`x`/`y` non réinitialisés), seuls les nouveaux sont placés par la simulation.
- Bouton "← Vue d'ensemble" toujours visible en mode détail → revient à `loadGraphOverview()`.
- Les variables `lastSubgraphNodes` / `lastSubgraphEdges` (déclarées mais jamais utilisées actuellement) sont enfin exploitées : après une réponse Q&A, elles sont peuplées et l'onglet Graphe, si ouvert ensuite, affiche directement ce sous-graphe focus au lieu de la vue d'ensemble.
- Pagination "charger plus" sur une grosse communauté : même mécanisme que `expandNode` (refetch avec `limit` plus grand, merge dans la vue existante).

---

## 3. Infrastructure d'embeddings + résolution d'entités

### 3.1 Choix techniques

- **Modèle** : `sentence-transformers/all-MiniLM-L6-v2`, local, CPU, anglais. Choisi après vérification que tout le corpus disponible (HotpotQA, démo `ai_history`, code/prompts) est en anglais — pas besoin du modèle multilingue plus lourd.
- **Pourquoi pas une API d'embeddings** : Anthropic n'a pas d'endpoint d'embeddings natif. Un modèle local évite de dépendre d'`OPENAI_API_KEY` (non garanti présent) ou d'ajouter un troisième provider (Voyage AI) juste pour ça, et reste cohérent avec le caractère multi-provider du projet (`LLM_PROVIDER=openai|anthropic|ollama`).
- **Coût** : nouvelle dépendance `sentence-transformers` (+ `torch` en transitif, installation plus lourde ~500MB-1GB) ; téléchargement unique du modèle (~80MB) au premier lancement, mis en cache localement par la librairie.

### 3.2 Nouveau module `graphrag_core/embeddings.py`

- Singleton paresseux : le modèle est chargé une seule fois par process (réutilisé pour le build ET pour chaque requête `/query`).
- `embed_texts(texts: list[str]) -> np.ndarray` — encodage batché, vecteurs L2-normalisés (similarité cosinus = produit scalaire simple).

### 3.3 Résolution d'entités (`graph.py`)

Nouvelle fonction :

```python
@dataclass
class MergeEvent:
    alias: str
    canonical: str
    similarity: float

def resolve_entities(
    triples: List[Triple], threshold: float = 0.90
) -> Tuple[List[Triple], List[MergeEvent], Dict[str, np.ndarray]]:
    ...
```

Algorithme :
1. Collecter tous les noms d'entités uniques (sujets + objets) des triplets, dans l'ordre d'apparition.
2. Les encoder en un seul batch (`embed_texts`).
3. Parcourir les noms dans l'ordre ; pour chaque nom, comparer (cosinus) aux embeddings des noms déjà acceptés comme canoniques (comparaison vectorisée par batch, pas nœud-par-nœud naïf) :
   - similarité max ≥ `threshold` → alias du nœud canonique correspondant (le premier nom vu fait foi), on enregistre un `MergeEvent`.
   - sinon → nouveau nœud canonique.
4. Réécrire tous les triplets avec les noms canoniques (table de correspondance `alias -> canonical`), puis dédupliquer les triplets devenus identiques (même logique `seen`/set déjà utilisée dans `extract_triples`).
5. Retourner les triplets réécrits, la liste des `MergeEvent`, et le dict `{nom_canonique: embedding}` (réutilisé directement pour peupler `KnowledgeGraph.node_embeddings`, pas de recalcul).

`KnowledgeGraph` (dataclass) gagne un champ `node_embeddings: Dict[str, np.ndarray]`.

**Limite connue** : comparaison en O(n × nb_canoniques_courant). Pire cas (aucune fusion) sur les ~30 770 entités du snapshot HotpotQA : jusqu'à ~1-2 min, dominé par les multiplications matricielles batchées. Acceptable car l'extraction LLM en amont prend déjà plus longtemps ; à reconsidérer (ex: blocage préalable par préfixe/longueur) si le corpus grossissait d'un ordre de grandeur.

### 3.4 Intégration dans le pipeline de build (`app/main.py`)

`_build_gen()` insère une étape entre `extraction` et `graph_build` :

```python
yield _sse("entity_resolution", 45, "Resolving entities…")
all_triples, merges, embeddings = resolve_entities(all_triples)
for m in merges:
    yield _sse("entity_resolution", 48, f"{m.alias} → merged into {m.canonical} ({m.similarity:.2f})")
```

Le frontend (`STAGES` dans `app.js`, checklist dans `index.html`) gagne une 5ᵉ étape `entity_resolution` entre "Extraction" et "Graph build", visible dans la barre de progression — utile pour démontrer la fonctionnalité en soutenance.

### 3.5 Persistance (export/import `.graphrag`)

Les embeddings et les `MergeEvent` ne sont **pas** persistés dans le zip d'export. Au format actuel (`triples.json` + `docs.json`), les triplets restent la seule source de vérité ; à l'import, `resolve_entities` et l'encodage sont simplement relancés. Cohérent avec le design existant, pas de changement de format de fichier.

---

## 4. Retriever hybride — sémantique + mot-clé avant BFS

### 4.1 Pourquoi combiner plutôt que remplacer

Le mot-clé exact (`detect_entities`, existant, inchangé) a une précision quasi parfaite mais un rappel faible (rate toute paraphrase). Le sémantique seul rattrape les paraphrases mais peut sous-performer sur des entités rares/numériques que l'embedding représente mal. L'union des deux ne coûte presque rien (le mot-clé est déjà calculé, le sémantique est un produit matriciel sur au plus quelques dizaines de milliers de vecteurs, <50ms) et ne peut quasiment jamais faire moins bien que l'un des deux seuls.

### 4.2 Nouvelle fonction (`retriever.py`)

```python
def semantic_seed_entities(
    question: str, kg: KnowledgeGraph, top_k: int = 5, min_similarity: float = 0.3
) -> List[str]:
    ...
```

Embed la question (même modèle, même module `embeddings.py`), similarité cosinus contre `kg.node_embeddings`, renvoie les `top_k` noms au-dessus du seuil plancher `min_similarity`. Le seuil plancher évite de retourner des seeds non pertinentes quand la question ne correspond vraiment à rien dans le graphe (sans lui, top-k renverrait toujours 5 résultats même hors-sujet).

### 4.3 Intégration (`pipeline.py`)

```python
seeds = set(detect_entities(question, kg)) | set(semantic_seed_entities(question, kg))
if not seeds:
    # fallback existant inchangé : top nœuds par degré
    ...
```

`extract_subgraph` (BFS, cap `max_nodes`, seuil hub) n'est pas modifié — il consomme déjà une liste de seeds sans se soucier de leur origine.

---

## 5. Tests

- `tests/test_graph.py` : `resolve_entities` — fusion de deux noms similaires au-dessus du seuil, absence de fusion sous le seuil, réécriture correcte des triplets, `MergeEvent` produits.
- `tests/test_retriever.py` : `semantic_seed_entities` — respect de `top_k`, filtrage par `min_similarity`.
- `tests/test_api.py` : remplace les tests de l'ancien `GET /graph` par les 3 nouvelles routes (overview / community / neighbors), y compris le cas `truncated=True`.
- `embeddings.py` est testé via un mock du modèle (`unittest.mock`, même pattern que `test_llm.py`), pour ne pas dépendre du téléchargement réel du modèle pendant les tests/CI.
- Pas de test E2E navigateur automatisé pour le drill-down D3 (hors scope de la suite pytest existante) — validation manuelle du flow clic-pour-explorer dans le navigateur après implémentation.

---

## 6. Hors scope

- Édition manuelle des fusions d'entités (accepter/rejeter une fusion proposée) — la fusion est automatique, sans interface de révision.
- Index de recherche approximé (FAISS/ANN) — à 30 770 nœuds, la comparaison brute-force en mémoire (~47MB de vecteurs float32) est suffisante ; pas justifié à cette échelle.
- Persistance des embeddings dans le format d'export `.graphrag`.
- Modèle d'embeddings multilingue — tout le corpus actuel est en anglais.
