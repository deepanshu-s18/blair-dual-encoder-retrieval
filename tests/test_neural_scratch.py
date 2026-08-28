"""Tests for neural_scratch implementations."""
import numpy as np
import pytest
from src.neural_scratch import softmax_stable

def test_softmax():
    assert np.allclose(softmax_stable(np.array([1.0, 2.0])).sum(), 1.0)

def test_infonce_numpy():
    pass
