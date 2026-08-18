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
