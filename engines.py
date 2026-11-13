"""Builds all three search engines on the same corpus."""
from bm25 import BM25Search
from preprocessing import tokenize
from semantic_search import SemanticSearch
from tfidf import TFIDFSearch


def build_engines(docs):
    texts = [f"{d['title']}. {d['text']}" for d in docs]
    tokens = [tokenize(t) for t in texts]

    print("Building TF-IDF index...")
    tfidf = TFIDFSearch().fit(tokens)
    print("Building BM25 index...")
    bm25 = BM25Search().fit(tokens)
    print("Building semantic index...")
    semantic = SemanticSearch().fit(texts)

    return {"tfidf": tfidf, "bm25": bm25, "semantic": semantic}