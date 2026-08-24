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

def knn_predict(X_train, y_train, x_new, k=5):
    distances = np.linalg.norm(X_train - x_new, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    values, counts = np.unique(k_labels, return_counts=True)
    return values[np.argmax(counts)]

def mse_loss_3d(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-np.clip(x, -500, 500))), np.exp(np.clip(x, -500, 500))/(1+np.exp(np.clip(x, -500, 500))))

def f_beta(precision, recall, beta=1.0):
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom
