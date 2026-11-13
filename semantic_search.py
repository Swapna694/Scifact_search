"""Semantic search: sentence embeddings + cosine similarity.

Each document and the query are turned into dense vectors by a
sentence-transformer. Texts with similar meaning end up close together,
so cosine similarity (angle between vectors) ranks documents by meaning,
not by shared words.

  cosine(a, b) = (a . b) / (|a| * |b|)
"""
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, MODEL_NAME


def cosine_similarity(query_vec, doc_matrix):
    """Cosine similarity between one query vector and every row of doc_matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
    return d @ q


class SemanticSearch:
    def fit(self, texts):
        self.model = SentenceTransformer(MODEL_NAME)
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache = os.path.join(CACHE_DIR, "corpus_embeddings.npy")

        if os.path.exists(cache):
            self.embeddings = np.load(cache)
        else:
            print("Encoding corpus (one time only, may take a few minutes)...")
            self.embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
            np.save(cache, self.embeddings)
        return self

    def search(self, query, k=3):
        q_vec = self.model.encode([query])[0]
        scores = cosine_similarity(q_vec, self.embeddings)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]