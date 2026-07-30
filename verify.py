"""
verify.py
=========
Smoke-test every module in the BLaIR dual encoder project.
No GPU, no data files, no heavy downloads required — runs in ~30 seconds on CPU.

Usage:
    cd /path/to/blair
    python verify.py

Passes:  PASS [module_name]
Fails:   FAIL [module_name] — error message
Summary: N/M tests passed
"""

import sys
import os
import traceback
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
results = []

def test(name, fn):
    global PASS, FAIL
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  PASS [{name}] ({elapsed:.2f}s)")
        PASS += 1
        results.append((True, name))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL [{name}] ({elapsed:.2f}s)")
        print(f"       {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        FAIL += 1
        results.append((False, name))


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1/7] Import checks")

def test_imports():
    import torch
    import transformers
    import numpy as np
    import pandas as pd

test("core_imports", test_imports)

def test_src_imports():
    from src.encoder import BiEncoder, DualEncoder, MeanPooling, CLSPooling, build_encoder, load_encoder
    from src.loss import infonce_loss, infonce_loss_with_hard_negatives
    from src.dataset import RetrievalDataset, collate_fn
    from src.metrics import compute_metrics, aggregate, ndcg_at_k, recall_at_k, mrr, hits_at_k
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever
    from src.hybrid_retriever import HybridRetriever
    from src.trainer import Trainer

test("src_imports", test_src_imports)


# ── 2. Encoder shapes ─────────────────────────────────────────────────────────
print("\n[2/7] Encoder forward pass (CPU, tiny BERT)")

def _make_tiny_bert():
    """Create a tiny BertModel from config (no download needed)."""
    from transformers import BertConfig, BertModel
    cfg = BertConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=1000,
    )
    return BertModel(cfg)


def test_biencoder_forward():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.encoder import MeanPooling

    bert = _make_tiny_bert()
    H = bert.config.hidden_size

    class _TinyBiEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert   = bert
            self.pooler = MeanPooling()
        def forward(self, q_ids, q_mask, p_ids, p_mask):
            q_out = self.bert(input_ids=q_ids, attention_mask=q_mask)
            p_out = self.bert(input_ids=p_ids, attention_mask=p_mask)
            q = F.normalize(self.pooler(q_out.last_hidden_state, q_mask), dim=-1)
            p = F.normalize(self.pooler(p_out.last_hidden_state, p_mask), dim=-1)
            return q, p

    model = _TinyBiEncoder()
    model.eval()
    B, L = 2, 16
    ids  = torch.ones(B, L, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.long)

    with torch.no_grad():
        q_emb, p_emb = model(ids, mask, ids, mask)

    assert q_emb.shape == (B, H), f"Expected ({B}, {H}), got {q_emb.shape}"
    assert abs(q_emb[0].norm().item() - 1.0) < 1e-4, "Embeddings not normalized"

test("biencoder_forward", test_biencoder_forward)


def test_dualencoder_forward():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.encoder import MeanPooling
    from transformers import BertModel

    cfg_t = _make_tiny_bert().config
    cfg_i = _make_tiny_bert().config

    class _TinyDualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import BertConfig
            self.text_encoder = BertModel(cfg_t)
            self.item_encoder = BertModel(cfg_i)
            self.query_pooler = MeanPooling()
            self.doc_pooler   = MeanPooling()
        def forward(self, q_ids, q_mask, p_ids, p_mask):
            q_out = self.text_encoder(input_ids=q_ids, attention_mask=q_mask)
            p_out = self.item_encoder(input_ids=p_ids, attention_mask=p_mask)
            q = F.normalize(self.query_pooler(q_out.last_hidden_state, q_mask), dim=-1)
            p = F.normalize(self.doc_pooler  (p_out.last_hidden_state, p_mask), dim=-1)
            return q, p

    model = _TinyDualEncoder()
    model.eval()
    B, L = 2, 16
    ids  = torch.ones(B, L, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.long)

    with torch.no_grad():
        q_emb, p_emb = model(ids, mask, ids, mask)

    assert q_emb.shape[0] == B
    # Parameter independence
    text_param_ids = {id(p) for p in model.text_encoder.parameters()}
    item_param_ids = {id(p) for p in model.item_encoder.parameters()}
    assert len(text_param_ids & item_param_ids) == 0, \
        "Encoders share parameters — should be independent"

test("dualencoder_forward", test_dualencoder_forward)


# ── 3. Loss function ──────────────────────────────────────────────────────────
print("\n[3/7] Loss function")

def test_infonce():
    import torch
    import torch.nn.functional as F
    from src.loss import infonce_loss, infonce_loss_with_hard_negatives

    B, D = 4, 32
    q = F.normalize(torch.randn(B, D), dim=-1)
    p = F.normalize(torch.randn(B, D), dim=-1)

    loss = infonce_loss(q, p, temperature=0.05)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    # Perfect retrieval should give ~0 loss
    perfect_loss = infonce_loss(q, q.clone(), temperature=0.05)
    assert perfect_loss.item() < loss.item(), \
        "Perfect alignment should have lower loss than random"

test("infonce_loss", test_infonce)

def test_infonce_hardneg():
    import torch
    import torch.nn.functional as F
    from src.loss import infonce_loss_with_hard_negatives

    B, D = 4, 32
    q   = F.normalize(torch.randn(B, D), dim=-1)
    pos = F.normalize(torch.randn(B, D), dim=-1)
    neg = F.normalize(torch.randn(B, D), dim=-1)

    loss = infonce_loss_with_hard_negatives(q, pos, neg, temperature=0.05)
    assert loss.item() > 0
    assert not torch.isnan(loss)

test("infonce_hardneg", test_infonce_hardneg)


# ── 4. Metrics ────────────────────────────────────────────────────────────────
print("\n[4/7] Metrics")

def test_metrics_basic():
    from src.metrics import ndcg_at_k, recall_at_k, mrr, hits_at_k, compute_metrics, aggregate

    # Hit at position 1
    assert ndcg_at_k(['A', 'B', 'C'], 'A', 10) == 1.0
    assert ndcg_at_k(['A', 'B', 'C'], 'B', 10) < 1.0
    assert ndcg_at_k(['X', 'Y', 'Z'], 'A', 10) == 0.0

    assert recall_at_k(['A', 'B'], 'A', k=5) == 1.0
    assert recall_at_k(['A', 'B'], 'C', k=5) == 0.0

    assert mrr(['A', 'B', 'C'], 'A') == 1.0
    assert abs(mrr(['X', 'A', 'B'], 'A') - 0.5) < 1e-6
    assert mrr(['X', 'Y', 'Z'], 'A') == 0.0

    assert hits_at_k(['A', 'B'], 'A', k=10) == 1
    assert hits_at_k(['X', 'Y'], 'A', k=10) == 0

    m = compute_metrics(['A', 'B', 'C', 'D'], 'A', k_values=(1, 5, 10))
    assert m['ndcg@10'] == 1.0
    assert m['recall@1'] == 1.0
    assert m['recall@10'] == 1.0

test("metrics_basic", test_metrics_basic)

def test_metrics_aggregate():
    from src.metrics import aggregate
    per_q = [
        {'ndcg@10': 1.0, 'recall@10': 1.0, 'mrr': 1.0, 'recall@1': 1.0},
        {'ndcg@10': 0.0, 'recall@10': 0.0, 'mrr': 0.0, 'recall@1': 0.0},
    ]
    agg = aggregate(per_q)
    assert abs(agg['ndcg@10'] - 0.5) < 1e-6
    assert abs(agg['mrr']     - 0.5) < 1e-6

test("metrics_aggregate", test_metrics_aggregate)


# ── 5. BM25 retriever ─────────────────────────────────────────────────────────
print("\n[5/7] BM25 retriever")

def test_bm25():
    from src.bm25_retriever import BM25Retriever

    ids  = ['p1', 'p2', 'p3', 'p4']
    docs = [
        'wireless bluetooth headphones noise cancelling',
        'usb-c laptop charger 65 watt fast charge',
        'mechanical keyboard cherry mx red switches backlit',
        'portable battery bank 20000mah usb power',
    ]

    bm25 = BM25Retriever(ids, docs)

    # "wireless" should retrieve p1
    results = bm25.retrieve('wireless headphones', k=2)
    assert len(results) == 2
    assert results[0][0] == 'p1', f"Expected p1, got {results[0][0]}"

    # Batch retrieve
    batch = bm25.batch_retrieve(['laptop charger', 'keyboard'], k=2)
    assert len(batch) == 2
    assert batch[0][0][0] == 'p2', f"Expected p2, got {batch[0][0][0]}"

test("bm25_retriever", test_bm25)


# ── 6. FAISS dense retriever ──────────────────────────────────────────────────
print("\n[6/7] FAISS dense retriever")

def test_dense_retriever():
    import numpy as np
    from src.dense_retriever import DenseRetriever

    D = 16
    ids  = ['a', 'b', 'c', 'd', 'e']
    embs = np.random.randn(len(ids), D).astype('float32')
    # Normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    retriever = DenseRetriever(ids, embs)

    # Exact query = emb[0] → should retrieve 'a' first
    q = embs[0:1]
    results = retriever.retrieve(q, k=3)
    assert len(results) == 3
    assert results[0][0] == 'a', f"Expected 'a', got {results[0][0]}"

    # Batch
    batch = retriever.batch_retrieve(embs[:2], k=3)
    assert len(batch) == 2

test("dense_retriever", test_dense_retriever)


# ── 7. Hybrid retriever ───────────────────────────────────────────────────────
print("\n[7/7] Hybrid retriever (RRF)")

def test_hybrid():
    import numpy as np
    from src.hybrid_retriever import HybridRetriever
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever

    ids  = ['p1', 'p2', 'p3', 'p4', 'p5']
    docs = [
        'bluetooth speaker waterproof outdoor',
        'hdmi cable 4k 60hz high speed',
        'gaming mouse rgb 16000 dpi',
        'smart plug wifi alexa google home',
        'led desk lamp usb charging port',
    ]

    D = 16
    embs = np.random.randn(len(ids), D).astype('float32')
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    bm25    = BM25Retriever(ids, docs)
    dense   = DenseRetriever(ids, embs)
    hybrid  = HybridRetriever(bm25, dense, rrf_k=60)

    q_emb = embs[0:1]
    results = hybrid.retrieve('bluetooth speaker', q_emb, k=3)
    assert len(results) == 3
    retrieved_ids = [r[0] for r in results]
    assert 'p1' in retrieved_ids, "BM25 keyword match should be in top-3"

test("hybrid_retriever", test_hybrid)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
total = PASS + FAIL
print(f"Results: {PASS}/{total} tests passed")

if FAIL > 0:
    print("\nFailed tests:")
    for ok, name in results:
        if not ok:
            print(f"  ✗ {name}")
    sys.exit(1)
else:
    print("All tests passed ✓")
    sys.exit(0)
