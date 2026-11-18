"""Semantic search with sentence embeddings stored in a Qdrant vector database.

Each abstract is turned into a vector by a sentence-transformer and stored in
Qdrant (local mode, in the qdrant_db folder). At query time the query is
encoded the same way and Qdrant returns the nearest vectors by cosine
similarity.

If a previous run was interrupted, encoding resumes where it stopped.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL

BATCH = 256

_client = None


def _get_client():
    """One Qdrant client per Python process.

    Uses Qdrant Cloud when QDRANT_URL is set, otherwise the local folder
    (local mode allows only one process at a time)."""
    global _client
    if _client is None:
        if QDRANT_URL:
            print("Connecting to Qdrant Cloud...")
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        else:
            _client = QdrantClient(path=QDRANT_PATH)
    return _client


class SemanticSearch:
    def fit(self, texts):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = _get_client()
        name = QDRANT_COLLECTION
        total = len(texts)

        start_from = 0
        if self.client.collection_exists(name):
            stored = self.client.count(name).count
            if stored == total:
                print(f"Using {stored} vectors already stored in Qdrant.")
                return self
            if stored > total:                       # corpus shrank: rebuild
                self.client.delete_collection(name)
            else:                                    # interrupted run: resume
                print(f"Resuming: {stored}/{total} vectors already stored.")
                start_from = stored

        if start_from == 0 and not self.client.collection_exists(name):
            get_dim = (getattr(self.model, "get_embedding_dimension", None)
                       or self.model.get_sentence_embedding_dimension)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=get_dim(), distance=Distance.COSINE),
            )

        print("Encoding corpus and storing vectors in Qdrant...")
        for start in range(start_from, total, BATCH):
            chunk = texts[start:start + BATCH]
            vectors = self.model.encode(chunk, show_progress_bar=False)
            points = [
                PointStruct(id=start + j, vector=v.tolist())
                for j, v in enumerate(vectors)
            ]
            self.client.upsert(collection_name=name, points=points)
            print(f"  stored {min(start + BATCH, total)}/{total}")
        return self

    def search(self, query, k=3):
        q_vec = self.model.encode([query])[0].tolist()
        res = self.client.query_points(
            collection_name=QDRANT_COLLECTION, query=q_vec, limit=k
        )
        # point id == position of the document in the corpus list
        return [(int(p.id), float(p.score)) for p in res.points]