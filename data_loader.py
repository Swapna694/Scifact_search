"""Download and load the SciFact dataset (BEIR format)."""
import csv
import io
import json
import os
import urllib.request
import zipfile

from config import DATA_DIR, DATA_URL


def _dataset_root():
    root = os.path.join(DATA_DIR, "scifact")
    if not os.path.exists(os.path.join(root, "corpus.jsonl")):
        print("Downloading SciFact dataset (one time only)...")
        os.makedirs(DATA_DIR, exist_ok=True)
        with urllib.request.urlopen(DATA_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    return root


def load_corpus():
    """Returns a list of {'id', 'title', 'text'}."""
    docs = []
    with open(os.path.join(_dataset_root(), "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append({"id": d["_id"], "title": d.get("title", ""), "text": d.get("text", "")})
    return docs


def load_qrels():
    """Returns {query_id: set(relevant_doc_ids)} for the test split."""
    qrels = {}
    path = os.path.join(_dataset_root(), "qrels", "test.tsv")
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def load_test_queries():
    """Returns [{'id', 'text'}] for queries that have relevance labels."""
    qrels = load_qrels()
    queries = []
    with open(os.path.join(_dataset_root(), "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries.append({"id": q["_id"], "text": q["text"]})
    return queries