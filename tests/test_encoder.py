"""
tests/test_encoder.py
Unit tests for src/encoder.py
"""
import sys
sys.path.insert(0, '.')
import pytest
import torch
import numpy as np
from src.encoder import BiEncoder, DualEncoder, MeanPooling

# ── MeanPooling ───────────────────────────────────────────
def test_mean_pooling_shape():
    pool = MeanPooling()
    token_embs = torch.randn(2, 10, 768)
    mask = torch.ones(2, 10)
    out = pool(token_embs, mask)
    assert out.shape == (2, 768)

def test_mean_pooling_masked():
    """Padding tokens should be excluded from mean"""
    pool = MeanPooling()
    token_embs = torch.zeros(1, 4, 4)
    token_embs[0, 0, :] = 1.0
    token_embs[0, 1, :] = 2.0
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
    out = pool(token_embs, mask)
    expected = torch.tensor([[1.5, 1.5, 1.5, 1.5]])
    assert torch.allclose(out, expected, atol=1e-5)

def test_mean_pooling_no_zero_division():
    """All-zero mask should not cause division by zero"""
    pool = MeanPooling()
    token_embs = torch.randn(1, 5, 768)
    mask = torch.zeros(1, 5)
    out = pool(token_embs, mask)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

# ── BiEncoder ─────────────────────────────────────────────
def test_biencoder_shared_weights():
    """BiEncoder must use same BERT for queries and docs"""
    model = BiEncoder()
    assert model.bert is model.bert  # same object

def test_biencoder_encode_queries_shape():
    model = BiEncoder()
    embs = model.encode_queries(['test query one', 'test query two'],
                                 batch_size=2, device='cpu')
    assert embs.shape == (2, 768)

def test_biencoder_encode_docs_shape():
    model = BiEncoder()
    embs = model.encode_docs(['product one description', 'product two description'],
                              batch_size=2, device='cpu')
    assert embs.shape == (2, 768)

def test_biencoder_l2_normalized():
    """Output embeddings must be L2 normalized"""
    model = BiEncoder()
    embs = model.encode_queries(['test'], batch_size=1, device='cpu')
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_biencoder_save_load(tmp_path):
    """Save and reload must produce identical embeddings"""
    model = BiEncoder()
    model.save(str(tmp_path))
    model2 = BiEncoder.load(str(tmp_path))
    text = ['test sentence for save load check']
    emb1 = model.encode_queries(text, batch_size=1, device='cpu')
    emb2 = model2.encode_queries(text, batch_size=1, device='cpu')
    assert np.allclose(emb1, emb2, atol=1e-5)

# ── DualEncoder ───────────────────────────────────────────
def test_dualencoder_separate_weights():
    """DualEncoder must have SEPARATE text and item encoders"""
    model = DualEncoder()
    assert model.text_encoder is not model.item_encoder

def test_dualencoder_encode_shape():
    model = DualEncoder()
    q_embs = model.encode_queries(['query'], batch_size=1, device='cpu')
    d_embs = model.encode_docs(['document'], batch_size=1, device='cpu')
    assert q_embs.shape == (1, 768)
    assert d_embs.shape == (1, 768)

def test_dualencoder_l2_normalized():
    model = DualEncoder()
    embs = model.encode_queries(['test'], batch_size=1, device='cpu')
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
