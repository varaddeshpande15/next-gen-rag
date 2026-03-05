import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dim: int = 768, index_file: str = "faiss_index.bin", meta_file: str = "faiss_metadata.pkl"):
        self.dim = dim
        self.index_file = index_file
        self.meta_file = meta_file
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self.load()

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        if not embeddings:
            return
        vector = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.metadata.extend(chunks)
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[float, Dict]]:
        if self.index.ntotal == 0:
            return []
        vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i != -1 and i < len(self.metadata):
                results.append((float(score), self.metadata[i]))
        return results

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, 'rb') as f:
                self.metadata = pickle.load(f)
