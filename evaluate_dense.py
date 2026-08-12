"""
evaluate_dense.py
=================
Evaluate a dense retrieval model (bi-encoder or dual-encoder) on the test set.

Handles two modes:
  1. Zero-shot: load bert-base-uncased without fine-tuning (--checkpoint not set)
  2. Fine-tuned: load from a trained checkpoint (--checkpoint path)

Usage:
    # Zero-shot (no fine-tuning)
    python evaluate_dense.py --model-type biencoder --output-dir results/zeroshot/

    # Fine-tuned bi-encoder
    python evaluate_dense.py --model-type biencoder \
        --checkpoint artifacts/models/biencoder_seed42/best_model \
        --output-dir results/biencoder/

    # Fine-tuned dual encoder
    python evaluate_dense.py --model-type dual \
        --checkpoint artifacts/models/dual_seed42/best_model \
        --output-dir results/dual/
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

sys.path.insert(0, ".")

from src.encoder import BiEncoder, DualEncoder, build_encoder, load_encoder
from src.dense_retriever import DenseRetriever
from src.metrics import compute_metrics, aggregate, print_metrics_table, hits_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval model")
    parser.add_argument("--data-dir",     type=str, default="data/",
                        help="Directory with parquet files")
    parser.add_argument("--output-dir",   type=str, default="results/dense/",
                        help="Directory to save metrics")
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help="Path to saved model checkpoint (None = zero-shot)")
    parser.add_argument("--model-type",   type=str, default="biencoder",
                        choices=["biencoder", "dual"],
                        help="Model type (needed when checkpoint is None)")
    parser.add_argument("--bert-model",   type=str, default="bert-base-uncased",
                        help="BERT backbone (used when checkpoint is None)")
    parser.add_argument("--pooling",      type=str, default="mean",
                        choices=["mean", "cls"],
                        help="Pooling strategy (used when checkpoint is None)")
    parser.add_argument("--max-len",      type=int, default=128)
    parser.add_argument("--encode-batch", type=int, default=64,
                        help="Batch size for corpus encoding (ALWAYS 8 for BERT on T4)")
    parser.add_argument("--query-batch",  type=int, default=128,
                        help="Batch size for query encoding")
    parser.add_argument("--k-values",     type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--save-top-k",   type=int, default=10,
                        help="Number of retrieved IDs saved per query (for failure analysis)")
    return parser.parse_args()


def load_data(data_dir: str):
    corpus_path = os.path.join(data_dir, "corpus.parquet")
    test_path   = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run build_dataset.py.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found: {test_path}. Run build_dataset.py.")

    corpus_df = pd.read_parquet(corpus_path)
    test_df   = pd.read_parquet(test_path)

    print(f"  Corpus size : {len(corpus_df):,} unique products")
    print(f"  Test pairs  : {len(test_df):,}")

    return corpus_df, test_df


def load_model(args):
    """Load fine-tuned model from checkpoint or build zero-shot encoder."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        print(f"  Loading fine-tuned model from: {args.checkpoint}")
        model = load_encoder(args.checkpoint)
        model = model.to(device)
        label = f"Fine-tuned {args.model_type}"
    else:
        print(f"  Building ZERO-SHOT encoder ({args.bert_model}, {args.model_type})")
        model = build_encoder(
            model_type=args.model_type,
            model_name=args.bert_model,
            max_len=args.max_len,
            pooling=args.pooling,
        )
        model = model.to(device)
        label = f"Zero-shot {args.model_type}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {label}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {device}")

    return model, device, label


def encode_corpus(model, corpus_docs: list, encode_batch: int = 8) -> np.ndarray:
    """
    Encode the full product corpus.
    batch_size=8 ALWAYS for BERT corpus encoding (T4 GPU constraint).
    """
    print(f"\nEncoding corpus ({len(corpus_docs):,} products, batch_size={encode_batch})...")
    start = time.time()

    # Determine encoding function based on model type
    if hasattr(model, "encode_docs"):
        corpus_embs = model.encode_docs(corpus_docs, batch_size=encode_batch)
    else:
        corpus_embs = model.encode(corpus_docs, batch_size=encode_batch)

    elapsed = time.time() - start
    print(f"  Corpus encoded in {elapsed:.1f}s | shape: {corpus_embs.shape}")
    return corpus_embs


def encode_queries(model, query_texts: list, query_batch: int = 32) -> np.ndarray:
    """Encode test queries."""
    print(f"\nEncoding {len(query_texts):,} test queries (batch_size={query_batch})...")
    start = time.time()

    if hasattr(model, "encode_queries"):
        query_embs = model.encode_queries(query_texts, batch_size=query_batch)
    else:
        query_embs = model.encode(query_texts, batch_size=query_batch)

    elapsed = time.time() - start
    print(f"  Queries encoded in {elapsed:.1f}s | shape: {query_embs.shape}")
    return query_embs


def evaluate(
    retriever: DenseRetriever,
    test_df: pd.DataFrame,
    query_embs: np.ndarray,
    k_values: list,
    save_top_k: int = 10,
) -> tuple:
    """
    Batch evaluate using pre-encoded query embeddings.

    Returns:
        (hits_list, aggregated_metrics)
    """
    print(f"\nRetrieving and computing metrics for {len(test_df):,} queries...")
    start = time.time()

    # Batch retrieve: much faster than per-query
    all_results = retriever.batch_retrieve(query_embs, k=max(k_values))

    per_query_metrics = []
    hits_list = []

    for i, (row_idx, row) in enumerate(test_df.iterrows()):
        product_id    = row["product_id"]
        retrieved_ids = [pid for pid, _ in all_results[i]]

        metrics = compute_metrics(retrieved_ids, product_id, k_values=tuple(k_values))
        per_query_metrics.append(metrics)

        hits_list.append({
            "query_text":    row["review_text"],
            "product_id":    product_id,
            "hits@10":       hits_at_k(retrieved_ids, product_id, k=10),
            "retrieved_ids": ",".join(retrieved_ids[:save_top_k]),
            **metrics,
        })

    elapsed = time.time() - start
    latency_ms = (elapsed / len(test_df)) * 1000
    print(f"  Done in {elapsed:.1f}s ({latency_ms:.1f}ms/query)")

    agg = aggregate(per_query_metrics)
    return hits_list, agg, latency_ms


def main():
    args = parse_args()

    # Enforce corpus encoding batch size constraint
    if args.encode_batch > 8:
        print(f"[WARNING] encode_batch {args.encode_batch} > 8, capping at 8 for T4 GPU")
        args.encode_batch = 8

    print("=" * 60)
    print("Dense Retrieval Evaluation")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint or 'NONE (zero-shot)'}")
    print(f"  Model type  : {args.model_type}")
    print(f"  Pooling     : {args.pooling}")

    # Load data
    print("\nLoading data...")
    corpus_df, test_df = load_data(args.data_dir)

    # Load model
    print("\nLoading model...")
    model, device, label = load_model(args)

    # Encode corpus
    corpus_ids  = corpus_df["product_id"].tolist()
    corpus_docs = corpus_df["product_doc"].tolist()
    corpus_embs = encode_corpus(model, corpus_docs, encode_batch=args.encode_batch)

    # Build FAISS index
    print("\nBuilding FAISS index...")
    retriever = DenseRetriever(corpus_ids, corpus_embs)

    # Encode test queries
    query_texts = test_df["review_text"].tolist()
    query_embs  = encode_queries(model, query_texts, query_batch=args.query_batch)

    # Evaluate
    hits_list, agg_metrics, latency_ms = evaluate(retriever, test_df, query_embs, args.k_values, save_top_k=args.save_top_k)

    # Print results
    print_metrics_table(agg_metrics, title=f"Dense Retrieval: {label}")

    # Save outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "label":       label,
            "n_queries":   int(len(test_df)),
            "corpus_size": int(len(corpus_df)),
            "latency_ms":  round(latency_ms, 3),
            **agg_metrics,
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    pq_df = pd.DataFrame(hits_list)
    pq_path = os.path.join(args.output_dir, "per_query_metrics.parquet")
    pq_df.to_parquet(pq_path, index=False)
    print(f"Per-query metrics saved to: {pq_path}")

    # Save corpus embeddings for hybrid retrieval
    emb_path = os.path.join(args.output_dir, "corpus_embs.npy")
    import numpy as np
    np.save(emb_path, corpus_embs)
    print(f"Corpus embeddings saved to: {emb_path}")

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS — {label}")
    print("=" * 60)
    for k, v in sorted(agg_metrics.items()):
        print(f"  {k:<20} : {v:.4f}")
    print("=" * 60)
    print("\n[DONE] Dense evaluation complete.")

    return agg_metrics


if __name__ == "__main__":
    main()
