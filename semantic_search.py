"""Semantic search with sentence embeddings stored in a Qdrant vector database.

Each abstract is turned into a vector by a sentence-transformer and stored in
Qdrant (local mode, in the qdrant_db folder). At query time the query is
encoded the same way and Qdrant returns the nearest vectors by cosine
similarity.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, QDRANT_COLLECTION, QDRANT_PATH

BATCH = 256


class SemanticSearch:
    def fit(self, texts):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient(path=QDRANT_PATH)
        name = QDRANT_COLLECTION

        # Reuse the stored vectors if the collection already holds this corpus
        if self.client.collection_exists(name):
            if self.client.count(name).count == len(texts):
                return self
            self.client.delete_collection(name)  # corpus changed: rebuild

        print("Encoding corpus and storing vectors in Qdrant (one time only)...")
        dim = self.model.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        for start in range(0, len(texts), BATCH):
            chunk = texts[start:start + BATCH]
            vectors = self.model.encode(chunk, show_progress_bar=False)
            points = [
                PointStruct(id=start + j, vector=v.tolist())
                for j, v in enumerate(vectors)
            ]
            self.client.upsert(collection_name=name, points=points)
            print(f"  stored {min(start + BATCH, len(texts))}/{len(texts)}")
        return self

    def search(self, query, k=3):
        q_vec = self.model.encode([query])[0].tolist()
        res = self.client.query_points(
            collection_name=QDRANT_COLLECTION, query=q_vec, limit=k
        )
        # point id == position of the document in the corpus list
        return [(int(p.id), float(p.score)) for p in res.points]