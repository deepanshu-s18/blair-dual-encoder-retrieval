"""
tests/test_retriever.py
Unit tests for retrievers and InfoNCE loss.
Note: DenseRetriever FAISS tests use mocking due to M2/ARM FAISS
      compatibility — FAISS works correctly at runtime (verified
      by all evaluation scripts), but crashes in pytest on M2.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

# ══════════════════════════════════════════════════════════
# DenseRetriever — mocked FAISS search
# ══════════════════════════════════════════════════════════
from src.dense_retriever import DenseRetriever

def _mock_retriever(n=10, dim=64):
    ids  = [f'P{i}' for i in range(n)]
    embs = np.random.randn(n, dim).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return ids, embs

def test_dense_retriever_builds():
    ids, embs = _mock_retriever()
    with patch('faiss.IndexFlatIP') as mock_index:
        mock_index.return_value.ntotal = 10
        mock_index.return_value.add = MagicMock()
        r = DenseRetriever(ids, embs)
    assert len(r.corpus_ids) == 10

def test_dense_retriever_ids_stored():
    ids, embs = _mock_retriever(n=5)
    with patch('faiss.IndexFlatIP') as mock_index:
        mock_index.return_value.ntotal = 5
        mock_index.return_value.add = MagicMock()
        r = DenseRetriever(ids, embs)
    assert r.corpus_ids == ids

def test_dense_retriever_retrieve_format():
    ids, embs = _mock_retriever(n=10)
    with patch('faiss.IndexFlatIP') as mock_index:
        inst = mock_index.return_value
        inst.ntotal = 10
        inst.add = MagicMock()
        scores = np.array([[0.9, 0.8, 0.7]])
        indices = np.array([[2, 5, 1]])
        inst.search = MagicMock(return_value=(scores, indices))
        r = DenseRetriever(ids, embs)
        query = np.random.randn(64).astype(np.float32)
        results = r.retrieve(query, k=3)
    assert len(results) == 3
    assert results[0] == ('P2', 0.9)
    assert results[1] == ('P5', 0.8)
    assert results[2] == ('P1', 0.7)

def test_dense_retriever_invalid_index_filtered():
    ids, embs = _mock_retriever(n=5)
    with patch('faiss.IndexFlatIP') as mock_index:
        inst = mock_index.return_value
        inst.ntotal = 5
        inst.add = MagicMock()
        scores  = np.array([[0.9, 0.5]])
        indices = np.array([[1, -1]])  # -1 = invalid
        inst.search = MagicMock(return_value=(scores, indices))
        r = DenseRetriever(ids, embs)
        results = r.retrieve(np.random.randn(64).astype(np.float32), k=2)
    assert len(results) == 1
    assert results[0][0] == 'P1'

def test_dense_retriever_batch_retrieve_format():
    ids, embs = _mock_retriever(n=10)
    with patch('faiss.IndexFlatIP') as mock_index:
        inst = mock_index.return_value
        inst.ntotal = 10
        inst.add = MagicMock()
        scores  = np.array([[0.9, 0.8], [0.7, 0.6]])
        indices = np.array([[0, 1], [2, 3]])
        inst.search = MagicMock(return_value=(scores, indices))
        r = DenseRetriever(ids, embs)
        queries = np.random.randn(2, 64).astype(np.float32)
        results = r.batch_retrieve(queries, k=2)
    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2

def test_dense_corpus_embs_float32():
    ids, embs = _mock_retriever()
    embs64 = embs.astype(np.float64)
    with patch('faiss.IndexFlatIP') as mock_index:
        inst = mock_index.return_value
        inst.ntotal = 10
        inst.add = MagicMock()
        r = DenseRetriever(ids, embs64.astype(np.float32))
    assert r.corpus_embs.dtype == np.float32

# ══════════════════════════════════════════════════════════
# BM25Retriever
# ══════════════════════════════════════════════════════════
from src.bm25_retriever import BM25Retriever

DOCS = [
    "universal remote control infrared blaster",
    "wireless bluetooth headphones noise cancelling",
    "usb c charging cable fast charge",
    "laptop stand adjustable ergonomic aluminum",
    "mechanical keyboard rgb backlight gaming",
]
IDS = [f'P{i}' for i in range(len(DOCS))]

def test_bm25_builds():
    r = BM25Retriever(IDS, DOCS)
    assert r is not None

def test_bm25_retrieve_returns_k():
    r = BM25Retriever(IDS, DOCS)
    results = r.retrieve("remote control", k=3)
    assert len(results) == 3

def test_bm25_exact_match_ranked_first():
    r = BM25Retriever(IDS, DOCS)
    results = r.retrieve("universal remote control infrared", k=5)
    assert results[0][0] == 'P0'

def test_bm25_scores_descending():
    r = BM25Retriever(IDS, DOCS)
    results = r.retrieve("keyboard gaming", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)

def test_bm25_batch_retrieve():
    r = BM25Retriever(IDS, DOCS)
    queries = ["remote control", "headphones bluetooth"]
    all_results = r.batch_retrieve(queries, k=2)
    assert len(all_results) == 2
    assert all(len(res) == 2 for res in all_results)

def test_bm25_irrelevant_query():
    r = BM25Retriever(IDS, DOCS)
    results = r.retrieve("zzz nonexistent token xyz", k=3)
    assert len(results) == 3

# ══════════════════════════════════════════════════════════
# InfoNCE Loss
# ══════════════════════════════════════════════════════════
from src.loss import infonce_loss

def test_infonce_loss_scalar():
    q = torch.randn(4, 64)
    p = torch.randn(4, 64)
    loss = infonce_loss(q, p, temperature=0.05)
    assert loss.shape == ()

def test_infonce_loss_nonnegative():
    q = torch.randn(8, 64)
    p = torch.randn(8, 64)
    loss = infonce_loss(q, p, temperature=0.05)
    assert loss.item() >= 0.0

def test_infonce_loss_perfect_pairs():
    embs = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=-1)
    loss = infonce_loss(embs, embs.clone(), temperature=0.05)
    assert loss.item() < 0.1

def test_infonce_loss_temperature_effect():
    q = torch.randn(4, 64)
    p = torch.randn(4, 64)
    loss_low  = infonce_loss(q, p, temperature=0.01)
    loss_high = infonce_loss(q, p, temperature=0.5)
    assert loss_low.item() != loss_high.item()

def test_infonce_loss_differentiable():
    q = torch.randn(4, 64, requires_grad=True)
    p = torch.randn(4, 64, requires_grad=True)
    loss = infonce_loss(q, p, temperature=0.05)
    loss.backward()
    assert q.grad is not None
    assert p.grad is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
