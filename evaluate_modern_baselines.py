"""
evaluate_modern_baselines.py
============================
Evaluate 2026 SOTA sentence encoders zero-shot as external baselines:
  - intfloat/e5-base-v2         (E5, contrastive pretrained)
  - BAAI/bge-base-en-v1.5       (BGE, top MTEB retriever)
  - facebook/contriever         (Contriever, unsupervised dense retrieval)

These are the baselines a Principal Applied Scientist expects in 2026,
not BM25 (2009) or vanilla BERT. This directly answers:
"Why didn't you compare against modern retrievers?"

Each model uses its REQUIRED prompt format:
  - E5:    "query: {text}" and "passage: {text}"
  - BGE:   query gets an instruction prefix; passages raw
  - Contriever: raw text, mean pooling

Usage:
    pip install sentence-transformers
    python evaluate_modern_baselines.py --model e5
    python evaluate_modern_baselines.py --model bge
    python evaluate_modern_baselines.py --model contriever
    python evaluate_modern_baselines.py --model all
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


MODEL_CONFIGS = {
    "e5": {
        "hf_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "output_dir": "results/e5",
        "label": "E5-base-v2",
    },
    "bge": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "output_dir": "results/bge",
        "label": "BGE-base-en-v1.5",
    },
    "contriever": {
        "hf_name": "facebook/contriever",
        "query_prefix": "",
        "doc_prefix": "",
        "output_dir": "results/contriever",
        "label": "Contriever",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="all",
                   choices=["e5", "bge", "contriever", "all"])
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def evaluate_one(model_key, data_dir, batch_size):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {cfg['label']} ({cfg['hf_name']})")
    print(f"{'='*60}")

    test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    corpus = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))

    # ── Contriever needs custom mean-pooling; E5/BGE use sentence-transformers ──
    if model_key == "contriever":
        embs_q, embs_d = _encode_contriever(
            cfg, test["review_text"].tolist(),
            corpus["product_doc"].tolist(), batch_size
        )
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["hf_name"])
        print(f"  Encoding {len(corpus):,} products...")
        embs_d = model.encode(
            [cfg["doc_prefix"] + d for d in corpus["product_doc"].tolist()],
            batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  Encoding {len(test):,} queries...")
        embs_q = model.encode(
            [cfg["query_prefix"] + q for q in test["review_text"].tolist()],
            batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )

    # ── FAISS retrieval (identical pipeline to all other systems) ──
    retriever = DenseRetriever(corpus["product_id"].tolist(),
                               embs_d.astype(np.float32))
    t0 = time.time()
    all_results = retriever.batch_retrieve(embs_q.astype(np.float32), k=10)
    latency = (time.time() - t0) / len(test) * 1000

    ndcgs, recalls, mrrs, r1s, hits = [], [], [], [], []
    for i, (_, row) in enumerate(test.iterrows()):
        retrieved_ids = [r[0] for r in all_results[i]]
        true_id = row["product_id"]
        ndcgs.append(ndcg_at_k(retrieved_ids, true_id, k=10))
        recalls.append(recall_at_k(retrieved_ids, true_id, k=10))
        mrrs.append(compute_mrr(retrieved_ids, true_id))
        r1s.append(recall_at_k(retrieved_ids, true_id, k=1))
        hits.append(hits_at_k(retrieved_ids, true_id, k=10))

    metrics = {
        "ndcg@10": float(np.mean(ndcgs)),
        "recall@10": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "recall@1": float(np.mean(r1s)),
        "latency_ms": float(latency),
        "model": cfg["hf_name"],
    }

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    per_query = pd.DataFrame({
        "query_text": test["review_text"].values,
        "product_id": test["product_id"].values,
        "hits@10": hits, "ndcg@10": ndcgs,
        "recall@1": r1s, "recall@10": recalls, "mrr": mrrs,
    })
    per_query.to_parquet(
        os.path.join(cfg["output_dir"], "per_query_metrics.parquet"), index=False)

    print(f"\n  {cfg['label']} RESULTS:")
    print(f"    NDCG@10   : {metrics['ndcg@10']:.4f}")
    print(f"    Recall@10 : {metrics['recall@10']:.4f}")
    print(f"    MRR       : {metrics['mrr']:.4f}")
    print(f"    Recall@1  : {metrics['recall@1']:.4f}")
    print(f"  vs BiEncoder: 0.0693 NDCG (your fine-tuned model)")
    print(f"  Saved to {cfg['output_dir']}")
    return metrics


def _encode_contriever(cfg, queries, docs, batch_size):
    """Contriever uses mean pooling over token embeddings, raw text."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModel.from_pretrained(cfg["hf_name"]).to(device).eval()

    def mean_pool(token_embs, mask):
        mask = mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(texts):
        out_all = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out_all.append(emb.cpu().numpy())
        return np.vstack(out_all)

    print(f"  Encoding {len(docs):,} products (Contriever)...")
    embs_d = encode(docs)
    print(f"  Encoding {len(queries):,} queries (Contriever)...")
    embs_q = encode(queries)
    return embs_q, embs_d


def main():
    args = parse_args()
    keys = ["e5", "bge", "contriever"] if args.model == "all" else [args.model]
    all_metrics = {}
    for key in keys:
        all_metrics[key] = evaluate_one(key, args.data_dir, args.batch_size)

    print(f"\n{'='*60}")
    print("  MODERN BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'NDCG@10':<10} {'Recall@10':<10}")
    print(f"  {'-'*42}")
    print(f"  {'BM25 (2009)':<22} {'0.0208':<10} {'0.0349':<10}")
    print(f"  {'SBERT MiniLM':<22} {'0.0660':<10} {'0.1117':<10}")
    for key in keys:
        m = all_metrics[key]
        print(f"  {MODEL_CONFIGS[key]['label']:<22} "
              f"{m['ndcg@10']:<10.4f} {m['recall@10']:<10.4f}")
    print(f"  {'BiEncoder (ours) ★':<22} {'0.0693':<10} {'0.1226':<10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
