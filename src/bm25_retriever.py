"""BM25 Okapi baseline implementation."""
import math
from collections import Counter

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
