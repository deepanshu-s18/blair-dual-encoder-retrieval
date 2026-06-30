"""
evaluate_hybrid.py
==================
Evaluate hybrid retrieval (BM25 + Dense via Reciprocal Rank Fusion).

RRF captures complementary signals:
  - BM25 → lexical precision (exact keyword overlap)
  - Dense → semantic recall (paraphrase, vocabulary mismatch)

Usage:
    python evaluate_hybrid.py \
        --checkpoint artifacts/models/dual_hardneg_seed42/best_model \
        --output-dir results/hybrid/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.encoder import load_encoder
from src.metrics import compute_metrics, aggregate, print_metrics_table, hits_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hybrid BM25 + Dense retrieval")
    parser.add_argument("--data-dir",     type=str, default="data/")
    parser.add_argument("--output-dir",   type=str, default="results/hybrid/")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to dense model checkpoint (best dual encoder)")
    parser.add_argument("--rrf-k",        type=int, default=60,
                        help="RRF constant k (default 60, Cormack et al.)")
    parser.add_argument("--fetch-k",      type=int, default=100,
                        help="Candidates from each retriever before fusion")
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--query-batch",  type=int, default=32)
    parser.add_argument("--k-values",     type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--bm25-cache",   type=str, default="artifacts/bm25_index.pkl",
                        help="Path to cached BM25 index")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Hybrid Retrieval Evaluation (BM25 + Dense → RRF)")
    print("=" * 60)

    # Load data
    corpus_df = pd.read_parquet(os.path.join(args.data_dir, "corpus.parquet"))
    test_df   = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    corpus_ids  = corpus_df["product_id"].tolist()
    corpus_docs = corpus_df["product_doc"].tolist()

    print(f"  Corpus: {len(corpus_df):,} products")
    print(f"  Test:   {len(test_df):,} queries")

    # ── BM25 Retriever ──────────────────────────────────────────
    print("\nBuilding BM25 retriever...")
    if os.path.exists(args.bm25_cache):
        bm25 = BM25Retriever.load(args.bm25_cache)
    else:
        bm25 = BM25Retriever(corpus_ids, corpus_docs)
        Path(os.path.dirname(args.bm25_cache)).mkdir(parents=True, exist_ok=True)
        bm25.save(args.bm25_cache)

    # ── Dense Retriever ─────────────────────────────────────────
    print(f"\nLoading dense model from {args.checkpoint}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = load_encoder(args.checkpoint).to(device)

    print(f"Encoding corpus (batch={args.encode_batch})...")
    corpus_embs = model.encode_docs(corpus_docs, batch_size=args.encode_batch)

    dense = DenseRetriever(corpus_ids, corpus_embs)

    # ── Hybrid Retriever ─────────────────────────────────────────
    hybrid = HybridRetriever(bm25, dense, rrf_k=args.rrf_k, fetch_k=args.fetch_k)
    print(f"  RRF k={args.rrf_k}, fetch_k={args.fetch_k}")

    # ── Encode Queries ───────────────────────────────────────────
    query_texts = test_df["review_text"].tolist()
    print(f"\nEncoding {len(query_texts):,} test queries...")
    query_embs  = model.encode_queries(query_texts, batch_size=args.query_batch)

    # ── Evaluate ─────────────────────────────────────────────────
    print(f"\nEvaluating hybrid retrieval on {len(test_df):,} queries...")
    per_query_metrics = []
    hits_list         = []
    start = time.time()

    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
        product_id = row["product_id"]
        query_text = row["review_text"]
        query_emb  = query_embs[i]

        results       = hybrid.retrieve(query_text, query_emb, k=max(args.k_values))
        retrieved_ids = [pid for pid, _ in results]

        metrics = compute_metrics(retrieved_ids, product_id, k_values=tuple(args.k_values))
        per_query_metrics.append(metrics)

        hits_list.append({
            "query_text":    query_text,
            "product_id":    product_id,
            "hits@10":       hits_at_k(retrieved_ids, product_id, k=10),
            "retrieved_ids": ",".join(retrieved_ids[:10]),
            **metrics,
        })

    elapsed = time.time() - start
    latency_ms = (elapsed / len(test_df)) * 1000
    agg = aggregate(per_query_metrics)

    print(f"  Done in {elapsed:.1f}s ({latency_ms:.1f}ms/query)")
    print_metrics_table(agg, title="Hybrid BM25 + Dense (RRF)")

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "label":       "Hybrid RRF",
            "n_queries":   int(len(test_df)),
            "corpus_size": int(len(corpus_df)),
            "latency_ms":  round(latency_ms, 3),
            **agg,
        }, f, indent=2)
    pd.DataFrame(hits_list).to_parquet(
        os.path.join(args.output_dir, "per_query_metrics.parquet"), index=False
    )

    print(f"\nResults saved to: {args.output_dir}")
    print("[DONE] Hybrid evaluation complete.")

    return agg


if __name__ == "__main__":
    main()
