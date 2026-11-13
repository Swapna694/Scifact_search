"""Interactive CLI chatbot: keyword search (TF-IDF + BM25) and semantic search."""
from config import TOP_K
from data_loader import load_corpus
from display import print_results
from engines import build_engines


def main():
    docs = load_corpus()
    engines = build_engines(docs)
    print(f"\nReady. {len(docs)} documents indexed. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        print("\nkeyword_search:")
        print("  tfidf:")
        print_results(docs, engines["tfidf"].search(query, TOP_K), indent=4)
        print("  bm25:")
        print_results(docs, engines["bm25"].search(query, TOP_K), indent=4)

        print("\nsemantic_search:")
        print_results(docs, engines["semantic"].search(query, TOP_K), indent=2)
        print()


if __name__ == "__main__":
    main()