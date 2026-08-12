"""
src/dataset.py
==============
PyTorch Dataset for contrastive retrieval training.

Two negative sampling modes:
  - "random": sample a random product from the corpus each step (easy negatives)
  - "bm25":   use precomputed BM25 top-k results (hard negatives)

Each __getitem__ returns:
  (query_text, positive_doc_text, negative_doc_text)
"""

import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class RetrievalDataset(Dataset):
    """
    Dataset for contrastive training of bi-encoder / dual-encoder.

    Each item contains:
        - query_text    : the review text
        - positive_doc  : the product document for that review
        - negative_doc  : a negative sample (random or BM25 hard)

    Negative sampling:
        mode="random" → random product from corpus, re-sampled each epoch
        mode="bm25"   → precomputed hard negatives (BM25 top hit ≠ true positive)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        corpus_docs: List[str],              # list of all product doc strings
        corpus_id_to_doc: Dict[str, str],    # product_id → product_doc
        tokenizer: BertTokenizer,
        max_len: int = 128,
        mode: str = "random",                # "random" or "bm25"
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        seed: int = 42,
    ):
        super().__init__()
        assert mode in ("random", "bm25"), f"Unknown mode: {mode}"
        if mode == "bm25" and hard_negatives is None:
            raise ValueError("mode='bm25' requires hard_negatives dict")

        self.df = df.reset_index(drop=True)
        self.corpus_docs = corpus_docs
        self.corpus_id_to_doc = corpus_id_to_doc
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.hard_negatives = hard_negatives
        self.rng = random.Random(seed)

        # Build index: row index → query_id (for hard negative lookup)
        if "query_id" not in self.df.columns:
            self.df = self.df.reset_index().rename(columns={"index": "query_id"})

        # Cache product_id column for hard negative lookup
        self._product_ids = self.df["product_id"].tolist()
        self._review_texts = self.df["review_text"].tolist()
        self._positive_docs = [
            corpus_id_to_doc.get(pid, "") for pid in self._product_ids
        ]

        # Filter out rows where positive doc is empty
        valid = [i for i, d in enumerate(self._positive_docs) if d.strip()]
        if len(valid) < len(self.df):
            print(f"[Dataset] Dropping {len(self.df)-len(valid)} rows with missing product docs")
        self._valid_indices = valid

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]

        query_text    = self._review_texts[real_idx]
        positive_doc  = self._positive_docs[real_idx]
        product_id    = self._product_ids[real_idx]

        # Sample negative
        if self.mode == "random":
            negative_doc = self._sample_random_negative(product_id)
        else:
            negative_doc = self._sample_bm25_negative(real_idx, product_id)

        return query_text, positive_doc, negative_doc

    def _sample_random_negative(self, true_product_id: str) -> str:
        """Sample a random product from the corpus, excluding the true positive."""
        while True:
            neg_doc = self.rng.choice(self.corpus_docs)
            # Chance of sampling the same product is very low for large corpus,
            # but we guard against it anyway
            if neg_doc != self.corpus_id_to_doc.get(true_product_id, "__NONE__"):
                return neg_doc

    def _sample_bm25_negative(self, row_idx: int, true_product_id: str) -> str:
        """
        Sample from precomputed BM25 hard negatives.
        Falls back to random if no hard negatives for this query.
        """
        negs = self.hard_negatives.get(true_product_id, [])
        if not negs:
            return self._sample_random_negative(true_product_id)
        # Random choice among the k hard negatives
        neg_pid = self.rng.choice(negs)
        return self.corpus_id_to_doc.get(neg_pid, self._sample_random_negative(true_product_id))


def collate_fn(tokenizer: BertTokenizer, max_len: int = 128):
    """
    Closure that returns a collate function for DataLoader.

    Returns a function that tokenizes (query, positive, negative) triples
    and returns a dict with input_ids + attention_masks.
    """
    def _collate(batch):
        queries   = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        negatives = [item[2] for item in batch]

        def tokenize(texts):
            return tokenizer(
                texts,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        q_enc = tokenize(queries)
        p_enc = tokenize(positives)
        n_enc = tokenize(negatives)

        return {
            "query_input_ids":   q_enc["input_ids"],
            "query_attn_mask":   q_enc["attention_mask"],
            "pos_input_ids":     p_enc["input_ids"],
            "pos_attn_mask":     p_enc["attention_mask"],
            "neg_input_ids":     n_enc["input_ids"],
            "neg_attn_mask":     n_enc["attention_mask"],
        }

    return _collate


def build_hard_negatives_bm25(
    train_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    k: int = 3,
    cache_path: str = "artifacts/cache/hard_negatives.json",
) -> Dict[str, List[str]]:
    import json as _json, os as _os
    from pathlib import Path as _Path
    # Load from cache if exists — avoids 6hr recomputation
    if _os.path.exists(cache_path):
        print(f"[HardNeg] Loading from cache: {cache_path}")
        with open(cache_path) as _f:
            return _json.load(_f)
    print(f"[HardNeg] Cache not found — computing (this takes ~6 hrs)...")
    """
    Precompute BM25 hard negatives for all training queries.

    For each unique product in the training set:
      1. Use the representative review as the query
      2. Retrieve BM25 top-k results
      3. Exclude the true positive product
      4. Return the remaining results as hard negatives

    Using product_id as the key (one entry per unique product):
    Multiple reviews for the same product share the same hard negatives.

    Args:
        train_df   : training DataFrame with columns [review_text, product_id]
        corpus_df  : corpus DataFrame with columns [product_id, product_doc]
        k          : number of hard negatives per query

    Returns:
        dict: {product_id: [neg_product_id_1, ..., neg_product_id_k]}
    """
    from src.bm25_retriever import BM25Retriever

    corpus_ids  = corpus_df["product_id"].tolist()
    corpus_docs = corpus_df["product_doc"].tolist()

    print("[HardNeg] Building BM25 index over corpus...")
    retriever = BM25Retriever(corpus_ids, corpus_docs)

    # For each unique product, use first review as representative query
    unique_products = train_df.groupby("product_id").first().reset_index()
    print(f"[HardNeg] Computing hard negatives for {len(unique_products):,} unique products...")

    hard_negatives: Dict[str, List[str]] = {}
    no_neg_count = 0

    for _, row in unique_products.iterrows():
        product_id = row["product_id"]
        query_text = row["review_text"]

        # Retrieve top-k+5 to have enough candidates after filtering true positive
        results = retriever.retrieve(query_text, k=k + 5)

        negs = [pid for pid, _score in results if pid != product_id][:k]
        if not negs:
            no_neg_count += 1
        hard_negatives[product_id] = negs

    print(f"[HardNeg] Done. {no_neg_count} products had no BM25 negatives (will use random).")
    # Save to cache so next run is instant
    import json as _json, os as _os
    from pathlib import Path as _Path
    _Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as _f:
        _json.dump(hard_negatives, _f)
    print(f"[HardNeg] Saved to cache: {cache_path}")
    return hard_negatives
