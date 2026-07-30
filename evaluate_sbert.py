"""
evaluate_sbert.py
=================
Evaluate sentence-transformers (all-MiniLM-L6-v2) as a zero-shot baseline.

This is the baseline every Amazon scientist will ask about:
"Why not just use a pre-trained sentence transformer?"

Usage:
    pip install sentence-transformers
    python evaluate_sbert.py --data-dir data/ --output-dir results/sbert/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.dense_retriever import DenseRetriever
from src.metrics import ndcg_at_k, recall_at_k, mrr as compute_mrr, hits_at_k


def parse_args():
    p = argparse.ArgumentParser(description="SBERT zero-shot baseline")
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--output-dir", type=str, default="results/sbert/")
    p.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────
    print(f"[SBERT] Loading data from {args.data_dir}")
    test = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
    corpus = pd.read_parquet(os.path.join(args.data_dir, "corpus.parquet"))

    print(f"[SBERT] Test queries: {len(test):,}")
    print(f"[SBERT] Corpus products: {len(corpus):,}")

    # ── Load sentence-transformers model ───────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install sentence-transformers")
        sys.exit(1)

    print(f"[SBERT] Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # ── Encode corpus ──────────────────────────────────────────────
    print(f"[SBERT] Encoding {len(corpus):,} products...")
    t0 = time.time()
    corpus_embs = model.encode(
        corpus["product_doc"].tolist(),
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    corpus_time = time.time() - t0
    print(f"[SBERT] Corpus encoded in {corpus_time:.1f}s, shape={corpus_embs.shape}")

    # ── Encode queries ─────────────────────────────────────────────
    print(f"[SBERT] Encoding {len(test):,} queries...")
    t0 = time.time()
    query_embs = model.encode(
        test["review_text"].tolist(),
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    query_time = time.time() - t0
    print(f"[SBERT] Queries encoded in {query_time:.1f}s, shape={query_embs.shape}")

    # ── Build FAISS index and retrieve ─────────────────────────────
    corpus_ids = corpus["product_id"].tolist()
    retriever = DenseRetriever(corpus_ids, corpus_embs.astype(np.float32))

    print(f"[SBERT] Retrieving top-10 for {len(test):,} queries...")
    t0 = time.time()
    all_results = retriever.batch_retrieve(query_embs.astype(np.float32), k=10)
    search_time = (time.time() - t0) / len(test) * 1000  # ms per query

    # ── Compute metrics ────────────────────────────────────────────
    ndcgs, recalls, mrrs, r1s, hit_list = [], [], [], [], []

    for i, (_, row) in enumerate(test.iterrows()):
        retrieved_ids = [r[0] for r in all_results[i]]
        true_id = row["product_id"]

        ndcgs.append(ndcg_at_k(retrieved_ids, true_id, k=10))
        recalls.append(recall_at_k(retrieved_ids, true_id, k=10))
        mrrs.append(compute_mrr(retrieved_ids, true_id))
        r1s.append(recall_at_k(retrieved_ids, true_id, k=1))
        hit_list.append(hits_at_k(retrieved_ids, true_id, k=10))

    metrics = {
        "ndcg@10": float(np.mean(ndcgs)),
        "recall@10": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "recall@1": float(np.mean(r1s)),
        "latency_ms": float(search_time),
        "model": args.model_name,
        "embedding_dim": int(corpus_embs.shape[1]),
    }

    # ── Save results ───────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    per_query = pd.DataFrame({
        "product_id": test["product_id"].values,
        "ndcg@10": ndcgs,
        "recall@10": recalls,
        "mrr": mrrs,
        "recall@1": r1s,
        "hits@10": hit_list,
    })
    per_query.to_parquet(
        os.path.join(args.output_dir, "per_query_metrics.parquet"), index=False
    )

    # ── Print results ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SBERT Baseline: {args.model_name}")
    print(f"{'='*55}")
    print(f"  NDCG@10   : {metrics['ndcg@10']:.4f}")
    print(f"  Recall@10 : {metrics['recall@10']:.4f}")
    print(f"  MRR       : {metrics['mrr']:.4f}")
    print(f"  Recall@1  : {metrics['recall@1']:.4f}")
    print(f"  Latency   : {metrics['latency_ms']:.1f} ms/query")
    print(f"  Dim       : {metrics['embedding_dim']}")
    print(f"{'='*55}")
    print(f"  Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
