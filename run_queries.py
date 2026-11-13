"""Run the SAME set of queries against TF-IDF, BM25 and semantic search.

Uses SciFact test queries (which have relevance labels), prints the top
results for each method, then reports average Recall@k and MRR.

Usage:  python run_queries.py            # first 10 queries
        python run_queries.py --n 50
"""
import argparse

from config import EVAL_K, TOP_K
from data_loader import load_corpus, load_qrels, load_test_queries
from display import print_results
from engines import build_engines

GROUPS = [
    ("keyword_search", [("tfidf", "tfidf"), ("bm25", "bm25")]),
    ("semantic_search", [("semantic", "semantic")]),
]
METHODS = [(k, l) for _, members in GROUPS for k, l in members]


def recall_at_k(ranked_ids, relevant, k):
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids, relevant):
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of queries to run")
    args = parser.parse_args()

    docs = load_corpus()
    qrels = load_qrels()
    queries = load_test_queries()[: args.n]
    engines = build_engines(docs)

    totals = {key: {"recall": 0.0, "mrr": 0.0} for key, _ in METHODS}

    for q in queries:
        relevant = qrels[q["id"]]
        print("\n" + "=" * 90)
        print(f"query: {q['text']}")
        for group, members in GROUPS:
            print(f"\n{group}:")
            for key, label in members:
                results = engines[key].search(q["text"], EVAL_K)
                ranked_ids = [docs[i]["id"] for i, _ in results]
                totals[key]["recall"] += recall_at_k(ranked_ids, relevant, EVAL_K)
                totals[key]["mrr"] += reciprocal_rank(ranked_ids, relevant)
                if group == "keyword_search":
                    print(f"  {label}:")
                    print_results(docs, results[:TOP_K], relevant, indent=4)
                else:
                    print_results(docs, results[:TOP_K], relevant, indent=2)

    n = len(queries)
    print("\n" + "=" * 90)
    print(f"SUMMARY over {n} queries")
    print(f"{'Method':<10} {'Recall@' + str(EVAL_K):<12} {'MRR':<8}")
    for key, label in METHODS:
        print(f"{label:<10} {totals[key]['recall'] / n:<12.3f} {totals[key]['mrr'] / n:<8.3f}")


if __name__ == "__main__":
    main()