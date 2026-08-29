import numpy as np
import pytest
from src.neural_scratch import (
    softmax_stable, cross_entropy_loss, forward_pass, backward,
    infonce_numpy, mean_pooling_numpy, knn_predict, mse_loss_3d,
    dropout, sigmoid, f_beta, binomial_prob, binomial_at_least,
    gini_index, entropy, kmeans, pca, l2_normalize
)

# ── softmax ───────────────────────────────────────────────
def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(softmax_stable(x).sum() - 1.0) < 1e-6

def test_softmax_numerically_stable():
    x = np.array([1000.0, 1001.0, 1002.0])
    s = softmax_stable(x)
    assert not np.isnan(s).any()
    assert abs(s.sum() - 1.0) < 1e-6

def test_softmax_batch():
    x = np.random.randn(4, 8)
    s = softmax_stable(x)
    assert s.shape == (4, 8)
    assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-6)

# ── cross_entropy_loss ────────────────────────────────────
def test_cross_entropy_perfect():
    y_pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    y_true = np.array([0, 1])
    loss = cross_entropy_loss(y_pred, y_true)
    assert loss > 0
    assert loss < 0.2

def test_cross_entropy_shape():
    y_pred = softmax_stable(np.random.randn(8, 4))
    y_true = np.random.randint(0, 4, size=8)
    loss = cross_entropy_loss(y_pred, y_true)
    assert loss.shape == ()

# ── forward_pass ──────────────────────────────────────────
def test_forward_pass_shape():
    X  = np.random.randn(4, 8)
    W1 = np.random.randn(8, 16)
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 3)
    b2 = np.zeros(3)
    a2, cache = forward_pass(X, W1, b1, W2, b2)
    assert a2.shape == (4, 3)

def test_forward_pass_probabilities():
    X  = np.random.randn(4, 8)
    W1 = np.random.randn(8, 16); b1 = np.zeros(16)
    W2 = np.random.randn(16, 3); b2 = np.zeros(3)
    a2, _ = forward_pass(X, W1, b1, W2, b2)
    assert np.allclose(a2.sum(axis=-1), 1.0, atol=1e-6)
    assert (a2 >= 0).all()

# ── backward ──────────────────────────────────────────────
def test_backward_reduces_loss():
    np.random.seed(42)
    X  = np.random.randn(8, 4)
    W1 = np.random.randn(4, 8);  b1 = np.zeros(8)
    W2 = np.random.randn(8, 3);  b2 = np.zeros(3)
    y_true = np.random.randint(0, 3, 8)
    a2, cache = forward_pass(X, W1, b1, W2, b2)
    loss_before = cross_entropy_loss(a2, y_true)
    W1n, b1n, W2n, b2n = backward(y_true, cache, lr=0.1)
    a2n, _ = forward_pass(X, W1n, b1n, W2n, b2n)
    loss_after = cross_entropy_loss(a2n, y_true)
    assert loss_after < loss_before

# ── infonce_numpy ─────────────────────────────────────────
def test_infonce_positive():
    np.random.seed(0)
    q = l2_normalize(np.random.randn(4, 8))
    p = l2_normalize(np.random.randn(4, 8))
    loss = infonce_numpy(q, p)
    assert loss > 0

def test_infonce_perfect_pairs_lower():
    np.random.seed(1)
    embs = l2_normalize(np.random.randn(4, 8))
    loss_perfect = infonce_numpy(embs, embs.copy(), temperature=0.05)
    p_rand = l2_normalize(np.random.randn(4, 8))
    loss_random = infonce_numpy(embs, p_rand, temperature=0.05)
    assert loss_perfect < loss_random

def test_infonce_temperature_effect():
    np.random.seed(2)
    q = l2_normalize(np.random.randn(4, 8))
    p = l2_normalize(np.random.randn(4, 8))
    loss_low  = infonce_numpy(q, p, temperature=0.01)
    loss_high = infonce_numpy(q, p, temperature=1.0)
    assert loss_low != loss_high

# ── mean_pooling_numpy ────────────────────────────────────
def test_mean_pooling_shape():
    embs = np.random.randn(3, 10, 16)
    mask = np.ones((3, 10))
    out = mean_pooling_numpy(embs, mask)
    assert out.shape == (3, 16)

def test_mean_pooling_ignores_padding():
    embs = np.ones((1, 4, 4))
    embs[0, 2, :] = 100.0  # padding token with large value
    embs[0, 3, :] = 100.0
    mask = np.array([[1, 1, 0, 0]])
    out = mean_pooling_numpy(embs, mask)
    assert np.allclose(out, 1.0, atol=1e-5)

# ── knn_predict ───────────────────────────────────────────
def test_knn_correct_class():
    X_tr = np.array([[0,0],[1,0],[0,1],[5,5],[6,5],[5,6]], dtype=float)
    y_tr = np.array([0,0,0,1,1,1])
    assert knn_predict(X_tr, y_tr, np.array([0.5, 0.5]), k=3) == 0
    assert knn_predict(X_tr, y_tr, np.array([5.5, 5.5]), k=3) == 1

def test_knn_k1():
    X_tr = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_tr = np.array([0, 1])
    assert knn_predict(X_tr, y_tr, np.array([0.1, 0.1]), k=1) == 0

# ── mse_loss_3d ───────────────────────────────────────────
def test_mse_zero():
    x = np.random.randn(2, 3, 4)
    assert mse_loss_3d(x, x) == 0.0

def test_mse_positive():
    y_pred = np.ones((2, 3, 4))
    y_true = np.zeros((2, 3, 4))
    assert abs(mse_loss_3d(y_pred, y_true) - 1.0) < 1e-6

# ── dropout ───────────────────────────────────────────────
def test_dropout_training_zeros():
    np.random.seed(0)
    x = np.ones((1000,))
    out = dropout(x, p=0.5, training=True)
    zero_frac = (out == 0).mean()
    assert 0.4 < zero_frac < 0.6

def test_dropout_inference_unchanged():
    x = np.random.randn(100)
    out = dropout(x, p=0.5, training=False)
    assert np.allclose(out, x)

# ── sigmoid ───────────────────────────────────────────────
def test_sigmoid_range():
    x = np.array([-100.0, 0.0, 100.0])
    s = sigmoid(x)
    assert s[0] > 0 and s[0] < 0.01
    assert abs(s[1] - 0.5) < 1e-6
    assert s[2] > 0.99

def test_sigmoid_no_nan():
    x = np.array([-1000.0, 1000.0])
    s = sigmoid(x)
    assert not np.isnan(s).any()

# ── f_beta ────────────────────────────────────────────────
def test_f1_balanced():
    score = f_beta(0.8, 0.8, beta=1.0)
    assert abs(score - 0.8) < 1e-6

def test_f2_weights_recall():
    f1  = f_beta(0.9, 0.5, beta=1.0)
    f2  = f_beta(0.9, 0.5, beta=2.0)
    assert f2 < f1  # F2 penalizes low recall more

# ── binomial ──────────────────────────────────────────────
def test_binomial_7coins():
    p5   = binomial_prob(7, 5, 0.9)
    pge5 = binomial_at_least(7, 5, 0.9)
    assert abs(p5   - 0.1240) < 0.001
    assert abs(pge5 - 0.9743) < 0.001

def test_binomial_sums_to_one():
    total = sum(binomial_prob(10, k, 0.3) for k in range(11))
    assert abs(total - 1.0) < 1e-6

# ── gini and entropy ──────────────────────────────────────
def test_gini_pure():
    assert gini_index([10, 0]) == 0.0

def test_gini_impure():
    assert abs(gini_index([7, 3]) - 0.42) < 0.01

def test_entropy_pure():
    assert entropy([10, 0]) == 0.0

def test_entropy_impure():
    assert abs(entropy([7, 3]) - 0.8813) < 0.001

# ── kmeans ────────────────────────────────────────────────
def test_kmeans_separates_clusters():
    np.random.seed(42)
    c1 = np.random.randn(20, 2)
    c2 = np.random.randn(20, 2) + 10
    X = np.vstack([c1, c2])
    labels, centroids = kmeans(X, k=2)
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[20]

# ── pca ───────────────────────────────────────────────────
def test_pca_shape():
    X = np.random.randn(50, 10)
    X_r, comps, evr = pca(X, n_components=3)
    assert X_r.shape == (50, 3)
    assert comps.shape == (3, 10)
    assert evr.shape == (3,)

def test_pca_variance_ratio():
    X = np.random.randn(100, 5)
    _, _, evr = pca(X, n_components=5)
    assert abs(evr.sum() - 1.0) < 1e-6

# ── l2_normalize ──────────────────────────────────────────
def test_l2_normalize_unit_norm():
    x = np.random.randn(8, 16)
    xn = l2_normalize(x)
    norms = np.linalg.norm(xn, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_l2_normalize_dot_equals_cosine():
    a = l2_normalize(np.random.randn(4))
    b = l2_normalize(np.random.randn(4))
    dot = np.dot(a, b)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(dot - cos) < 1e-6
