"""
src/hybrid_retriever.py
=======================
Hybrid retriever combining BM25 and dense retrieval via
Reciprocal Rank Fusion (RRF).

RRF was proposed by Cormack, Clarke & Buettcher (SIGIR 2009).
It combines ranked lists without requiring score calibration:

    RRF_score(d) = Σ_r 1 / (k + rank_r(d))

where k=60 is the standard constant that dampens the influence
of very high-ranked items.

Why RRF works:
  - BM25 captures lexical overlap (exact keyword matches)
  - Dense retrieval captures semantic similarity (paraphrase, synonyms)
  - Their errors are partially complementary → fusion improves both
  - RRF is robust to score distribution mismatch between systems
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) of BM25 and Dense retrievers.

    score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
    k=60 (standard RRF constant, empirically robust)

    Products not found by a retriever get rank = N+1 (excluded from
    fusion with that component).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        fetch_k: int = 100,   # how many candidates to fetch before RRF
    ):
        """
        Args:
            bm25    : BM25Retriever for lexical retrieval
            dense   : DenseRetriever for semantic retrieval
            rrf_k   : RRF constant (default 60, Cormack et al.)
            fetch_k : number of candidates from each retriever before fusion
        """
        self.bm25    = bm25
        self.dense   = dense
        self.rrf_k   = rrf_k
        self.fetch_k = fetch_k

    def retrieve(
        self,
        query_text: str,
        query_emb: np.ndarray,    # (D,) L2-normalized
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k products via RRF fusion.

        Args:
            query_text : raw query text (for BM25)
            query_emb  : L2-normalized query embedding (for dense)
            k          : number of final results

        Returns:
            list of (product_id, rrf_score) tuples, ordered descending
        """
        # BM25 ranked list
        bm25_results  = self.bm25.retrieve(query_text, k=self.fetch_k)
        bm25_rank_map = {pid: rank+1 for rank, (pid, _) in enumerate(bm25_results)}

        # Dense ranked list
        dense_results  = self.dense.retrieve(query_emb, k=self.fetch_k)
        dense_rank_map = {pid: rank+1 for rank, (pid, _) in enumerate(dense_results)}

        # Compute RRF scores over union of candidates
        all_pids = set(bm25_rank_map.keys()) | set(dense_rank_map.keys())
        N = self.fetch_k + 1   # rank for missing documents

        rrf_scores: Dict[str, float] = {}
        for pid in all_pids:
            r_bm25  = bm25_rank_map.get(pid, N)
            r_dense = dense_rank_map.get(pid, N)
            rrf_scores[pid] = (
                1.0 / (self.rrf_k + r_bm25)
                + 1.0 / (self.rrf_k + r_dense)
            )

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:k]

    def batch_retrieve(
        self,
        query_texts: List[str],
        query_embs: np.ndarray,    # (Q, D)
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieval for multiple queries.

        Args:
            query_texts : list of raw query strings
            query_embs  : (Q, D) numpy array of L2-normalized embeddings
            k           : number of results per query

        Returns:
            list of Q result lists
        """
        all_results = []
        for i, (text, emb) in enumerate(zip(query_texts, query_embs)):
            results = self.retrieve(text, emb, k=k)
            all_results.append(results)
        return all_results
