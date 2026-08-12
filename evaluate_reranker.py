"""
evaluate_reranker.py
====================
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) applied to top-10 candidates
from the hybrid retriever.

Pipeline:
  Stage 1 (recall)   : Hybrid BM25 + Dense → top-10 candidates
  Stage 2 (precision): Cross-encoder reranker → rerank top-10

This achieves best precision at highest latency — appropriate for final
re-ranking in a production two-stage retrieval pipeline.

Usage:
    python evaluate_reranker.py \
        --stage1-results results/hybrid \
        --checkpoint artifacts/models/dual_hardneg_seed42/best_model \
        --output-dir results/reranker/
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

from src.metrics import compute_metrics, aggregate, print_metrics_table, hits_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-encoder reranker evaluation")
    parser.add_argument("--stage1-results", type=str, default="results/hybrid/",
                        help="Directory with hybrid per_query_metrics.parquet")
    parser.add_argument("--data-dir",       type=str, default="data/")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Dense encoder checkpoint (for fallback Stage 1)")
    parser.add_argument("--output-dir",     type=str, default="results/reranker/")
    parser.add_argument("--reranker-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="HuggingFace cross-encoder model")
    parser.add_argument("--top-k-stage1",   type=int, default=10,
                        help="Number of candidates to rerank")
    parser.add_argument("--k-values",       type=int, nargs="+", default=[1, 5, 10])
    return parser.parse_args()


def load_corpus_lookup(data_dir: str) -> dict:
    corpus_df = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
    return dict(zip(corpus_df["product_id"], corpus_df["product_doc"]))


def load_reranker(model_name: str, device: str):
    """Load cross-encoder model from HuggingFace."""
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Loading cross-encoder: {model_name}")
        reranker = CrossEncoder(model_name, device=device)
        return reranker, "sentence_transformers"
    except ImportError:
        print("  sentence_transformers not found, using raw HuggingFace...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        model     = model.to(device)
        return (tokenizer, model), "hf_raw"


def rerank_with_crossencoder(
    query: str,
    candidate_ids: list,
    corpus_lookup: dict,
    reranker,
    mode: str,
    device: str,
) -> list:
    """
    Rerank candidate product IDs for a single query.

    Returns:
        list of product_id strings, reranked best-first
    """
    candidates = [(pid, corpus_lookup.get(pid, "")) for pid in candidate_ids]
    candidates = [(pid, doc) for pid, doc in candidates if doc]  # filter empty

    if not candidates:
        return candidate_ids

    if mode == "sentence_transformers":
        pairs  = [(query, doc) for _, doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip([pid for pid, _ in candidates], scores),
                        key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked]
    else:
        tokenizer, model = reranker
        model.eval()
        pairs = [(query, doc) for _, doc in candidates]
        with torch.no_grad():
            enc = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc    = {k: v.to(device) for k, v in enc.items()}
            scores = model(**enc).logits.squeeze(-1).cpu().numpy()
        ranked = sorted(zip([pid for pid, _ in candidates], scores),
                        key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Cross-Encoder Reranker Evaluation")
    print("=" * 60)

    # Load stage 1 results
    stage1_path = os.path.join(args.stage1_results, "per_query_metrics.parquet")
    if not os.path.exists(stage1_path):
        raise FileNotFoundError(
            f"Stage 1 results not found: {stage1_path}\n"
            f"Run evaluate_hybrid.py first."
        )

    print(f"\nLoading stage 1 results from: {stage1_path}")
    stage1_df = pd.read_parquet(stage1_path)
    print(f"  {len(stage1_df):,} queries from stage 1")

    # Load corpus
    print("\nLoading corpus lookup...")
    corpus_lookup = load_corpus_lookup(args.data_dir)
    print(f"  Corpus: {len(corpus_lookup):,} products")

    # Load reranker
    print(f"\nLoading reranker: {args.reranker_model}...")
    reranker, mode = load_reranker(args.reranker_model, device)
    print(f"  Reranker loaded (mode={mode}, device={device})")

    # Evaluate
    print(f"\nReranking {len(stage1_df):,} queries...")
    per_query_metrics = []
    hits_list         = []
    start = time.time()

    for i, row in tqdm(stage1_df.iterrows(), total=len(stage1_df)):
        query_text  = row["query_text"]
        product_id  = row["product_id"]

        # Parse stage 1 retrieved IDs
        if "retrieved_ids" in row and isinstance(row["retrieved_ids"], str):
            stage1_ids = [pid.strip() for pid in row["retrieved_ids"].split(",") if pid.strip()]
        else:
            stage1_ids = []

        if not stage1_ids:
            # No stage 1 candidates → use empty retrieval
            reranked_ids = []
        else:
            # Rerank stage 1 candidates
            stage1_ids   = stage1_ids[:args.top_k_stage1]
            reranked_ids = rerank_with_crossencoder(
                query_text, stage1_ids, corpus_lookup,
                reranker, mode, device
            )

        metrics = compute_metrics(reranked_ids, product_id, k_values=tuple(args.k_values))
        per_query_metrics.append(metrics)

        hits_list.append({
            "query_text":    query_text,
            "product_id":    product_id,
            "hits@10":       hits_at_k(reranked_ids, product_id, k=10),
            "retrieved_ids": ",".join(reranked_ids[:10]),
            **metrics,
        })

    elapsed = time.time() - start
    agg = aggregate(per_query_metrics)

    print(f"  Done in {elapsed:.1f}s ({elapsed/len(stage1_df)*1000:.1f}ms/query)")
    print_metrics_table(agg, title="Hybrid + Cross-Encoder Reranker")

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"label": "Hybrid+Reranker", **agg}, f, indent=2)
    pd.DataFrame(hits_list).to_parquet(
        os.path.join(args.output_dir, "per_query_metrics.parquet"), index=False
    )

    print(f"\nResults saved to: {args.output_dir}")
    print("[DONE] Reranker evaluation complete.")

    return agg


if __name__ == "__main__":
    main()
