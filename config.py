import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MODEL_NAME = "all-MiniLM-L6-v2"   # embedding model for semantic search

TOP_K = 1         # results shown per method
EVAL_K = 10        # depth used when computing Recall@k / MRR

# BM25 hyper-parameters
BM25_K1 = 1.5
BM25_B = 0.75