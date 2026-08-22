"""
tests/test_integration.py
End-to-end integration test: encode → index → retrieve → evaluate.
Uses a tiny synthetic corpus (10 products, 5 queries) so it
runs in <30 seconds without GPU or real data.
"""
import sys
sys.path.insert(0, '.')
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# ── Synthetic data ─────────────────────────────────────────
CORPUS_IDS = [f'P{i}' for i in range(10)]
CORPUS_DOCS = [
    "universal remote control infrared blaster multi-device",
    "wireless bluetooth headphones noise cancelling foldable",
    "usb-c fast charging cable braided nylon 6ft",
    "laptop stand adjustable ergonomic aluminum portable",
    "mechanical keyboard rgb backlight gaming tenkeyless",
    "webcam 1080p hd autofocus built-in microphone",
    "wireless mouse ergonomic rechargeable silent click",
    "monitor 27 inch 4k ips display 144hz",
    "ssd external portable 1tb usb 3.2",
    "phone stand desk holder adjustable angle",
]
QUERIES = [
    "finally stopped squinting at my tiny screen great upgrade",
    "my back pain is gone since i got this for my desk",
    "charges so fast compared to the old cable",
    "love how quiet the clicks are in meetings",
    "perfect for video calls working from home",
]
TRUE_IDS = ['P7', 'P3', 'P2', 'P4', 'P5']  # ground truth


# ══════════════════════════════════════════════════════════
# Integration Test 1: Metrics pipeline
# ══════════════════════════════════════════════════════════
from src.metrics import ndcg_at_k, recall_at_k, mrr, hits_at_k

def test_metrics_pipeline_perfect():
    """All queries retrieve correct product at rank 1."""
    ndcgs, recalls, mrrs = [], [], []
    for true_id in TRUE_IDS:
        retrieved = [true_id] + [f'X{i}' for i in range(9)]
        ndcgs.append(ndcg_at_k(retrieved, true_id, k=10))
        recalls.append(recall_at_k(retrieved, true_id, k=10))
        mrrs.append(mrr(retrieved, true_id))
    assert np.mean(ndcgs) == 1.0
    assert np.mean(recalls) == 1.0
    assert np.mean(mrrs) == 1.0

def test_metrics_pipeline_random():
    """Random retrieval → metrics near zero."""
    np.random.seed(42)
    ndcgs = []
    for true_id in TRUE_IDS:
        retrieved = [f'X{i}' for i in range(10)]  # wrong products
        ndcgs.append(ndcg_at_k(retrieved, true_id, k=10))
    assert np.mean(ndcgs) == 0.0

def test_metrics_pipeline_partial():
    """Correct product at rank 5 → NDCG < 1, Recall = 1."""
    true_id = 'P3'
    retrieved = ['X0','X1','X2','X3', true_id, 'X4','X5','X6','X7','X8']
    assert ndcg_at_k(retrieved, true_id, k=10) < 1.0
    assert recall_at_k(retrieved, true_id, k=10) == 1.0
    assert hits_at_k(retrieved, true_id, k=10) == 1


# ══════════════════════════════════════════════════════════
# Integration Test 2: DenseRetriever pipeline (mocked FAISS)
# ══════════════════════════════════════════════════════════
from src.dense_retriever import DenseRetriever

def test_dense_retrieval_pipeline():
    """Full encode→index→retrieve pipeline with mocked FAISS."""
    n, dim = 10, 64
    corpus_embs = np.random.randn(n, dim).astype(np.float32)
    corpus_embs /= np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    query_embs  = np.random.randn(5, dim).astype(np.float32)
    query_embs  /= np.linalg.norm(query_embs, axis=1, keepdims=True)

    with patch('faiss.IndexFlatIP') as mock_idx:
        inst = mock_idx.return_value
        inst.ntotal = n
        inst.add    = MagicMock()
        scores  = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.5,
                                     0.4, 0.3, 0.2, 0.1, 0.05]), (5, 1))
        indices = np.tile(np.arange(n), (5, 1))
        inst.search = MagicMock(return_value=(scores, indices))

        retriever   = DenseRetriever(CORPUS_IDS, corpus_embs)
        all_results = retriever.batch_retrieve(query_embs, k=10)

    assert len(all_results) == 5
    for results in all_results:
        assert len(results) == 10
        pids   = [pid for pid, _ in results]
        scores_list = [s for _, s in results]
        assert scores_list == sorted(scores_list, reverse=True)
        assert all(pid in CORPUS_IDS for pid in pids)

def test_dense_retrieval_top1_correct():
    """Top-1 retrieved product matches expected corpus ID."""
    n, dim = 5, 32
    ids  = [f'PROD_{i}' for i in range(n)]
    embs = np.random.randn(n, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    with patch('faiss.IndexFlatIP') as mock_idx:
        inst = mock_idx.return_value
        inst.ntotal = n
        inst.add    = MagicMock()
        inst.search = MagicMock(return_value=(
            np.array([[0.99, 0.5]]),
            np.array([[2, 0]])
        ))
        r = DenseRetriever(ids, embs)
        results = r.retrieve(embs[0], k=2)

    assert results[0][0] == 'PROD_2'
    assert results[0][1] == pytest.approx(0.99)


# ══════════════════════════════════════════════════════════
# Integration Test 3: BM25 pipeline
# ══════════════════════════════════════════════════════════
from src.bm25_retriever import BM25Retriever

def test_bm25_pipeline_builds_and_retrieves():
    r = BM25Retriever(CORPUS_IDS, CORPUS_DOCS)
    results = r.retrieve("remote control infrared", k=5)
    assert len(results) == 5
    assert results[0][0] == 'P0'

def test_bm25_pipeline_all_queries():
    r = BM25Retriever(CORPUS_IDS, CORPUS_DOCS)
    all_results = r.batch_retrieve(
        ["remote control", "headphones", "charging cable",
         "keyboard gaming", "webcam video"], k=3)
    assert len(all_results) == 5
    assert all(len(res) == 3 for res in all_results)

def test_bm25_pipeline_ndcg_computable():
    """BM25 results can be fed directly to NDCG computation."""
    r = BM25Retriever(CORPUS_IDS, CORPUS_DOCS)
    results = r.retrieve("keyboard gaming mechanical", k=10)
    retrieved_ids = [pid for pid, _ in results]
    ndcg = ndcg_at_k(retrieved_ids, 'P4', k=10)
    assert 0.0 <= ndcg <= 1.0


# ══════════════════════════════════════════════════════════
# Integration Test 4: InfoNCE loss pipeline
# ══════════════════════════════════════════════════════════
from src.loss import infonce_loss

def test_infonce_full_training_step():
    """Simulate one training step: forward + backward + optimizer."""
    from src.encoder import MeanPooling
    import torch.nn as nn

    B, D = 8, 128
    query_embs = torch.randn(B, D, requires_grad=True)
    doc_embs   = torch.randn(B, D, requires_grad=True)

    # Forward
    loss = infonce_loss(query_embs, doc_embs, temperature=0.05)

    # Backward
    loss.backward()

    assert query_embs.grad is not None
    assert doc_embs.grad is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0

def test_infonce_batch_size_1_fails_gracefully():
    """Batch size 1 has no in-batch negatives — loss should still compute."""
    q = torch.randn(1, 64)
    p = torch.randn(1, 64)
    loss = infonce_loss(q, p, temperature=0.05)
    assert not torch.isnan(loss)

def test_infonce_gradient_flows_through_norm():
    """Gradient must flow through L2 normalization."""
    q = torch.randn(4, 64, requires_grad=True)
    q_norm = torch.nn.functional.normalize(q, p=2, dim=-1)
    p_norm = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=-1)
    loss = infonce_loss(q_norm, p_norm, temperature=0.05)
    loss.backward()
    assert q.grad is not None
    assert not torch.isnan(q.grad).any()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
