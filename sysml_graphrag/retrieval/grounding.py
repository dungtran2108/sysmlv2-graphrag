"""Frozen local embedding grounding for the routed candidate."""
from typing import Any, Optional, Set

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util

from sysml_graphrag.retrieval.node_text import node_grounding_text, node_text_digest


def ground_terms_via_embeddings(
    G: nx.DiGraph,
    query_text: str,
    embedder: Optional[SentenceTransformer] = None,
    max_delta: float = 0.12,
    min_similarity: float = 0.35,
    top_k: int = 10,
    precomputed_node_embs: Optional[torch.Tensor] = None,
    **kwargs: Any
) -> Set[str]:
    """Grounds user query terms to graph seed anchor nodes using dynamic relative

    delta thresholding and discrete embedding cosine similarity.
    """
    node_ids = list(G.nodes())
    if not node_ids:
        return set()

    if precomputed_node_embs is not None and len(precomputed_node_embs) != len(node_ids):
        raise ValueError('Embedding row count differs from graph nodes; rebuild the cache for the current scoped IDs.')
    cache_ids = kwargs.get('precomputed_node_ids')
    if precomputed_node_embs is not None and cache_ids is not None and list(cache_ids) != node_ids:
        raise ValueError('Embedding node order differs from the graph; rebuild the cache.')
    if precomputed_node_embs is not None and G.graph.get('grounding_text_version') and (
            kwargs.get('precomputed_node_text_sha256') != node_text_digest(G)):
        raise ValueError('Embedding text fingerprint differs or is missing; rebuild the cache using current node grounding text.')

    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Node Embedding Cache handling
    if precomputed_node_embs is not None:
        node_embs = precomputed_node_embs
    else:
        node_texts = [
            node_grounding_text(n, G.nodes[n])
            for n in node_ids
        ]
        node_embs = embedder.encode(node_texts, convert_to_tensor=True)

    query_emb = embedder.encode(query_text, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, node_embs)[0]
    max_score = float(torch.max(scores))
    if max_score < min_similarity:
        return set()
    cutoff = max(max_score - max_delta, min_similarity)
    return {node_ids[idx] for idx, score in enumerate(scores) if float(score) >= cutoff}
