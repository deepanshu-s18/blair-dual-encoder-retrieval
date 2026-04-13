"""
tests/test_metrics.py
Unit tests for src/metrics.py
"""
import sys
sys.path.insert(0, '.')
import math
import pytest
from src.metrics import ndcg_at_k, recall_at_k, mrr, hits_at_k

def test_ndcg_perfect():
    assert ndcg_at_k(['A','B','C'], 'A', k=10) == 1.0

def test_ndcg_rank2():
    expected = 1.0 / math.log2(3)
    assert abs(ndcg_at_k(['B','A','C'], 'A', k=10) - expected) < 1e-6

def test_ndcg_not_found():
    assert ndcg_at_k(['B','C','D'], 'A', k=10) == 0.0

def test_ndcg_beyond_k():
    assert ndcg_at_k(['B','C','D','E','F','G','H','I','J','K','A'], 'A', k=10) == 0.0

def test_recall_hit():
    assert recall_at_k(['A','B','C'], 'A', k=3) == 1.0

def test_recall_miss():
    assert recall_at_k(['B','C','D'], 'A', k=3) == 0.0

def test_recall_at_k_boundary():
    assert recall_at_k(['B','C','A'], 'A', k=3) == 1.0

def test_recall_beyond_k():
    assert recall_at_k(['B','C','D','A'], 'A', k=3) == 0.0

def test_mrr_rank1():
    assert mrr(['A','B','C'], 'A') == 1.0

def test_mrr_rank2():
    assert abs(mrr(['B','A','C'], 'A') - 0.5) < 1e-6

def test_mrr_rank3():
    assert abs(mrr(['B','C','A'], 'A') - 1/3) < 1e-6

def test_mrr_not_found():
    assert mrr(['B','C','D'], 'A') == 0.0

def test_hits_found():
    assert hits_at_k(['A','B','C'], 'A', k=3) == 1

def test_hits_not_found():
    assert hits_at_k(['B','C','D'], 'A', k=3) == 0

def test_hits_beyond_k():
    assert hits_at_k(['B','C','D','A'], 'A', k=3) == 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
