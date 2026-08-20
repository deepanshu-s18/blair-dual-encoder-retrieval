import numpy as np
from math import comb

def softmax_stable(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    correct_probs = y_pred[np.arange(batch_size), y_true]
    return -np.log(correct_probs + 1e-9).mean()

def forward_pass(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    a2 = softmax_stable(z2)
    cache = (X, z1, a1, z2, a2, W1, W2, b1, b2)
    return a2, cache

def backward(y_true, cache, lr=0.01):
    X, z1, a1, z2, a2, W1, W2, b1, b2 = cache
    B = X.shape[0]
    dz2 = a2.copy()
    dz2[np.arange(B), y_true] -= 1
    dz2 /= B
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return W1 - lr*dW1, b1 - lr*db1, W2 - lr*dW2, b2 - lr*db2

def infonce_numpy(q_embs, p_embs, temperature=0.05):
    sim = q_embs @ p_embs.T / temperature
    labels = np.arange(len(q_embs))
    return cross_entropy_loss(softmax_stable(sim), labels)

def mean_pooling_numpy(token_embs, attention_mask):
    mask = attention_mask[:, :, np.newaxis].astype(float)
    summed = (token_embs * mask).sum(axis=1)
    count = mask.sum(axis=1).clip(min=1e-9)
    return summed / count
