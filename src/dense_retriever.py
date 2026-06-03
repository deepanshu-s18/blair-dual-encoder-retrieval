"""Dense vector retriever using FAISS."""
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, dim=768):
        self.index = faiss.IndexFlatIP(dim)
    def search(self, q_emb, k=10):
        pass
