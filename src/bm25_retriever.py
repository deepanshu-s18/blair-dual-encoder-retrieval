"""
src/bm25_retriever.py
=====================
BM25Okapi retriever wrapping rank-bm25.
Used for:
  1. Lexical retrieval baseline
  2. Hard negative mining for Model 3

BM25 serves as the upper bound for lexical retrieval and as
a strong hard negative miner: products with high BM25 score
(lexical overlap) but that are NOT the true product are
difficult negatives for contrastive training.
"""

import os
import pickle
import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization for BM25."""
    text = text.lower()
    # Remove punctuation except hyphens (important for product names)
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    # Filter empty tokens
    tokens = [t for t in tokens if t.strip()]
    return tokens


class BM25Retriever:
    """
    BM25Okapi retriever over the product corpus.

    Tokenizes the corpus at build time and supports efficient
    retrieval for any query text.

    BM25 scores:
        score(q, d) = Σ_i IDF(q_i) * (tf(q_i, d) * (k1+1)) / (tf(q_i, d) + k1*(1-b+b*|d|/avgdl))

    where k1=1.5, b=0.75 (Okapi defaults).
    """

    def __init__(
        self,
        corpus_ids: List[str],
        corpus_docs: List[str],
    ):
        """
        Args:
            corpus_ids  : list of product_id strings
            corpus_docs : list of product document strings (same order)
        """
        assert len(corpus_ids) == len(corpus_docs), (
            f"Mismatch: {len(corpus_ids)} ids vs {len(corpus_docs)} docs"
        )
        self.corpus_ids  = corpus_ids
        self.corpus_docs = corpus_docs

        print(f"[BM25] Tokenizing {len(corpus_docs):,} documents...")
        tokenized_corpus = [_tokenize(doc) for doc in tqdm(corpus_docs, desc="Tokenizing")]

        print("[BM25] Building BM25Okapi index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Index built. Corpus size: {len(corpus_docs):,}")

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_ids)

    def retrieve(
        self,
        query_text: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k products for a single query.

        Args:
            query_text : the review text (query)
            k          : number of results to return

        Returns:
            list of (product_id, bm25_score) tuples, ordered by score descending
        """
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            # Empty query — return empty
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append((self.corpus_ids[idx], float(scores[idx])))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieval for multiple queries.

        Args:
            queries : list of query strings
            k       : number of results per query

        Returns:
            list of lists of (product_id, score) tuples
        """
        results = []
        for q in tqdm(queries, desc="BM25 retrieving", unit="query"):
            results.append(self.retrieve(q, k=k))
        return results

    def save(self, path: str):
        """Pickle the retriever to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[BM25] Saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """Load a pickled BM25Retriever."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[BM25] Loaded from {path} (corpus size: {obj.corpus_size:,})")
        return obj
