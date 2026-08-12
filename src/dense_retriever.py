"""
src/dense_retriever.py
Dense retriever using pure numpy (no FAISS) — works on Apple Silicon M2.
For 57k vectors at 768-dim, numpy search is fast enough (~1s per batch).
"""
import numpy as np
from typing import Callable, List, Tuple


class DenseRetriever:

    def __init__(self, corpus_ids: List[str], corpus_embs: np.ndarray):
        assert len(corpus_ids) == corpus_embs.shape[0]
        self.corpus_ids  = corpus_ids
        self.corpus_embs = np.ascontiguousarray(corpus_embs, dtype=np.float32)
        print(f"[Retriever] Built: {len(corpus_ids):,} vectors, dim={corpus_embs.shape[1]}")

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_ids)

    def _to_numpy(self, arr) -> np.ndarray:
        if hasattr(arr, 'cpu'):
            arr = arr.cpu()
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()
        return np.ascontiguousarray(arr, dtype=np.float32)

    def retrieve(self, query_emb, k: int = 10) -> List[Tuple[str, float]]:
        query_emb = self._to_numpy(query_emb).flatten()
        scores    = self.corpus_embs @ query_emb
        top_k     = np.argsort(scores)[::-1][:k]
        return [(self.corpus_ids[i], float(scores[i])) for i in top_k]

    def batch_retrieve(self, query_embs, k: int = 10) -> List[List[Tuple[str, float]]]:
        query_embs = self._to_numpy(query_embs)
        # Matrix multiply: (Q, D) @ (D, N) -> (Q, N)
        all_scores = query_embs @ self.corpus_embs.T
        all_results = []
        for scores in all_scores:
            top_k = np.argsort(scores)[::-1][:k]
            all_results.append([(self.corpus_ids[i], float(scores[i])) for i in top_k])
        return all_results

    @classmethod
    def build(cls, encoder_fn, corpus_ids, corpus_docs, batch_size=8):
        corpus_embs = encoder_fn(corpus_docs)
        return cls(corpus_ids, corpus_embs)

    def get_corpus_embs(self) -> np.ndarray:
        return self.corpus_embs
