"""Terminal RAG chatbot.   Run:  python chat_cli.py"""
from data_loader import load_corpus
from engines import build_engines
from rag import RAGChatbot


def main():
    docs = load_corpus()
    engines = build_engines(docs)
    bot = RAGChatbot(docs, engines)
    print("\nRAG chatbot ready. Type 'reset' to clear memory, 'exit' to quit.\n")

    while True:
        try:
            q = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q.lower() == "reset":
            bot.reset()
            print("(memory cleared)\n")
            continue

        try:
            r = bot.ask(q)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        if r["standalone"] != q:
            print(f"(searched for: {r['standalone']})")
        print(f"\nbot: {r['answer']}\n")
        t = r["timings"]
        print(f"(time: rewrite {t['rewrite']:.1f}s | search+rerank {t['search']:.1f}s | LLM answer {t['llm_answer']:.1f}s)")
        print("sources:")
        for d, _ in r["sources"]:
            print(f"  [{d['id']}] {d['title']}")
        print()


if __name__ == "__main__":
    main()