import os

from dotenv import load_dotenv

load_dotenv()  # reads QDRANT_URL / QDRANT_API_KEY from a .env file if present

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MODEL_NAME = "all-MiniLM-L6-v2"   # embedding model for semantic search

TOP_K = 3          # results shown per method
EVAL_K = 10        # depth used when computing Recall@k / MRR

# BM25 hyper-parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Reranker (cross-encoder)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 10   # N: candidates taken from EACH retriever (BM25 and semantic) before reranking
RERANK_TOP_N = 3   # how many reranked results to show

# Qdrant vector database (local mode: stored in a folder, no server needed)
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
QDRANT_COLLECTION = "scifact"

# Qdrant Cloud: if QDRANT_URL is set, the cluster is used instead of the local folder
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# LLM for the RAG chatbot (Google Gemini API)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
RAG_TOP_N = 5
MAX_ABSTRACT_CHARS = 1500
