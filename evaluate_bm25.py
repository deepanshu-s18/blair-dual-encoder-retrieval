"""
evaluate_bm25.py
================
Evaluate BM25 Okapi lexical retrieval on the test set.

Establishes the lexical retrieval ceiling. BM25 works well when query and
product share keywords, but completely fails on vocabulary mismatch cases
(e.g., "worked without codes" ↔ "universal remote control").

Usage:
    python evaluate_bm25.py --data-dir data/ --output-dir results/bm25/
    python evaluate_bm25.py --data-dir data/ --output-dir results/bm25/ --cache  # uses pkl cache
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from src.bm25_retriever import BM25Retriever
from src.metrics import compute_metrics, aggregate, print_metrics_table, hits_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval baseline")
    parser.add_argument("--data-dir",   type=str, default="data/",
                        help="Directory with parquet files")
    parser.add_argument("--output-dir", type=str, default="results/bm25/",
                        help="Directory to save metrics")
    parser.add_argument("--cache",      action="store_true",
                        help="Cache/reload BM25 index from pickle")
    parser.add_argument("--cache-path", type=str, default="artifacts/bm25_index.pkl",
                        help="Path to cache the BM25 index")
    parser.add_argument("--k-values",  type=int, nargs="+", default=[1, 5, 10],
                        help="Recall@k values to compute")
    return parser.parse_args()


def load_data(data_dir: str):
    corpus_path = os.path.join(data_dir, "corpus.parquet")
    test_path   = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run build_dataset.py first.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found: {test_path}. Run build_dataset.py first.")

    corpus_df = pd.read_parquet(corpus_path)
    test_df   = pd.read_parquet(test_path)

    print(f"  Corpus size : {len(corpus_df):,} unique products")
    print(f"  Test pairs  : {len(test_df):,} review-product pairs")

    return corpus_df, test_df


def build_or_load_retriever(
    corpus_ids: list,
    corpus_docs: list,
    cache: bool,
    cache_path: str,
) -> BM25Retriever:
    """Build BM25 index or load from cache."""
    if cache and os.path.exists(cache_path):
        print(f"  Loading BM25 index from cache: {cache_path}")
        return BM25Retriever.load(cache_path)

    retriever = BM25Retriever(corpus_ids, corpus_docs)

    if cache:
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        retriever.save(cache_path)

    return retriever


def evaluate(
    retriever: BM25Retriever,
    test_df: pd.DataFrame,
    k_values: list,
) -> tuple:
    """
    Evaluate BM25 on test set.

    Returns:
        (per_query_metrics, aggregated_metrics)
    """
    print(f"\nEvaluating on {len(test_df):,} test queries...")

    per_query_metrics = []
    hits_list = []
    start_time = time.time()

    for i, row in test_df.iterrows():
        query_text  = row["review_text"]
        product_id  = row["product_id"]

        # Retrieve top-10
        results     = retriever.retrieve(query_text, k=max(k_values))
        retrieved_ids = [pid for pid, _ in results]

        # Compute metrics
        metrics = compute_metrics(retrieved_ids, product_id, k_values=tuple(k_values))
        per_query_metrics.append(metrics)

        # Track hits@10 for McNemar test
        hits_list.append({
            "query_text":   query_text,
            "product_id":   product_id,
            "hits@10":      hits_at_k(retrieved_ids, product_id, k=10),
            "retrieved_ids": ",".join(retrieved_ids[:10]),
            **metrics,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1:,}/{len(test_df):,}] Elapsed: {elapsed:.0f}s")

    total_time = time.time() - start_time
    latency_ms = (total_time / len(test_df)) * 1000
    print(f"  Done in {total_time:.1f}s ({latency_ms:.1f}ms/query)")

    agg = aggregate(per_query_metrics)
    return hits_list, agg, latency_ms


def main():
    args = parse_args()

    print("=" * 60)
    print("BM25 Okapi Retrieval Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    corpus_df, test_df = load_data(args.data_dir)

    corpus_ids  = corpus_df["product_id"].tolist()
    corpus_docs = corpus_df["product_doc"].tolist()

    # Build/load BM25 index
    print("\nBuilding BM25 index...")
    retriever = build_or_load_retriever(
        corpus_ids, corpus_docs,
        cache=args.cache,
        cache_path=args.cache_path,
    )

    # Evaluate
    hits_list, agg_metrics, latency_ms = evaluate(retriever, test_df, args.k_values)

    # Print results
    print_metrics_table(agg_metrics, title="BM25 Okapi Results")

    # Save outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "label":       "BM25 Okapi",
            "n_queries":   int(len(test_df)),
            "corpus_size": int(len(corpus_df)),
            "latency_ms":  round(latency_ms, 3),
            **agg_metrics,
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save per-query metrics (for McNemar test)
    pq_df = pd.DataFrame(hits_list)
    pq_path = os.path.join(args.output_dir, "per_query_metrics.parquet")
    pq_df.to_parquet(pq_path, index=False)
    print(f"Per-query metrics saved to: {pq_path}")

    # Print formatted summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS — BM25 Okapi")
    print("=" * 60)
    for k, v in sorted(agg_metrics.items()):
        print(f"  {k:<20} : {v:.4f}")
    print("=" * 60)
    print("\n[DONE] BM25 evaluation complete.")

    return agg_metrics


if __name__ == "__main__":
    main()
