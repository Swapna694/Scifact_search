"""Two-stage search: retrieve candidates, then rerank with a cross-encoder.

Stage 1: BM25 and semantic search each return their top RERANK_POOL documents;
         the two lists are merged (duplicates removed).
Stage 2: a cross-encoder reads the query and each candidate TOGETHER and gives
         a new relevance score. Candidates are re-sorted by that score.
"""
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL, RERANK_POOL


class RerankedSearch:
    def __init__(self, docs, bm25, semantic):
        self.docs = docs
        self.bm25 = bm25
        self.semantic = semantic
        print("Loading reranker model...")
        self.model = CrossEncoder(RERANK_MODEL, max_length=512)

    def search(self, query, k=3):
        # Stage 1: merge candidates from both retrievers (keeps order, no duplicates)
        candidates = {}
        for engine in (self.bm25, self.semantic):
            for i, _ in engine.search(query, RERANK_POOL):
                candidates[i] = None
        indexes = list(candidates)

        # Stage 2: cross-encoder scores each (query, document) pair
        pairs = [(query, f"{self.docs[i]['title']}. {self.docs[i]['text']}") for i in indexes]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(indexes, scores), key=lambda x: x[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked[:k]]