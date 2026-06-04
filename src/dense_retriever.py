"""
src/dense_retriever.py
======================
Dense retriever using FAISS IndexFlatIP (exact inner product search).

Since embeddings are L2-normalized, inner product == cosine similarity.
IndexFlatIP gives exact (non-approximate) nearest neighbour search,
appropriate for our 56,921-product corpus.

For 100M+ products at Amazon scale, swap IndexFlatIP for:
  - FAISS IVFFlat  : approximate, ~10x speedup, <1% recall loss
  - FAISS HNSW     : graph-based ANN, best recall/latency tradeoff
  - ScaNN (Google) : state-of-the-art ANN for production

References:
  - Johnson et al., "Billion-scale similarity search with GPUs" (FAISS paper)
  - Used in DPR, BLaIR, and Amazon's own retrieval systems
"""

import numpy as np
import faiss
from typing import Callable, List, Tuple


class DenseRetriever:
    """
    Dense retriever using FAISS IndexFlatIP.

    Exact inner product search (= cosine similarity for L2-normalized embeddings).

    Usage:
        retriever = DenseRetriever(corpus_ids, corpus_embs)
        results   = retriever.retrieve(query_emb, k=10)
        results   = retriever.batch_retrieve(query_embs, k=10)
    """

    def __init__(self, corpus_ids: List[str], corpus_embs: np.ndarray):
        """
        Args:
            corpus_ids  : list of product_id strings
            corpus_embs : (N, D) float32 numpy array of L2-normalized embeddings
        """
        assert len(corpus_ids) == corpus_embs.shape[0], (
            f"Mismatch: {len(corpus_ids)} ids vs {corpus_embs.shape[0]} embeddings"
        )
        self.corpus_ids  = corpus_ids
        self.corpus_embs = np.ascontiguousarray(corpus_embs, dtype=np.float32)

        dim = corpus_embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # exact inner product search
        self.index.add(self.corpus_embs)
        print(f"[FAISS] IndexFlatIP built: {self.index.ntotal:,} vectors, dim={dim}")

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_ids)

    def retrieve(self, query_emb, k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k products for a single query embedding.

        Args:
            query_emb : (D,) or (1, D) float32 numpy array (L2-normalized)
            k         : number of results

        Returns:
            list of (product_id, cosine_similarity) tuples, ordered descending
        """
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        scores, indices = self.index.search(query_emb, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:   # FAISS returns -1 for invalid entries
                results.append((self.corpus_ids[idx], float(score)))
        return results

    def batch_retrieve(self, query_embs, k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieval for multiple query embeddings.

        Args:
            query_embs : (Q, D) float32 numpy array of L2-normalized embeddings
            k          : number of results per query

        Returns:
            list of Q lists, each containing (product_id, score) tuples
        """
        query_embs = np.ascontiguousarray(query_embs, dtype=np.float32)
        scores, indices = self.index.search(query_embs, k)

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for idx, score in zip(q_indices, q_scores):
                if idx >= 0:
                    results.append((self.corpus_ids[idx], float(score)))
            all_results.append(results)
        return all_results

    @classmethod
    def build(
        cls,
        encoder_fn: Callable[[List[str]], np.ndarray],
        corpus_ids: List[str],
        corpus_docs: List[str],
        batch_size: int = 8,
    ) -> "DenseRetriever":
        """
        Build a DenseRetriever by encoding the corpus with encoder_fn.

        Args:
            encoder_fn  : callable (texts -> np.ndarray), use model.encode_docs
            corpus_ids  : list of product_id strings
            corpus_docs : list of product document strings
            batch_size  : encoding batch size (8 for BERT to fit in memory)

        Returns:
            DenseRetriever with FAISS index over encoded corpus
        """
        print(f"[DenseRetriever] Encoding {len(corpus_docs):,} documents...")
        corpus_embs = encoder_fn(corpus_docs)
        print(f"[DenseRetriever] Corpus encoded: shape={corpus_embs.shape}")
        return cls(corpus_ids, corpus_embs)

    def get_corpus_embs(self) -> np.ndarray:
        """Return stored corpus embeddings."""
        return self.corpus_embs
