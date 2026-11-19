"""RAG logic: retrieve papers -> build a prompt -> ask the LLM -> return answer + sources."""
import time

from config import MAX_ABSTRACT_CHARS, RAG_TOP_N
from llm import ask_llm

SYSTEM_PROMPT = """You are a research assistant answering questions about scientific papers.
Use ONLY the numbered sources provided in the message. Do not use outside knowledge.
Cite sources inline with their ids in square brackets, for example [18810195].
If the sources do not contain enough evidence, say: "The retrieved papers do not contain enough evidence to answer this."
If the user states a claim, begin with exactly one of: SUPPORTED, CONTRADICTED, or NOT ENOUGH EVIDENCE, then explain in 2-4 sentences.
Otherwise answer in 2-5 sentences. Be concise and never invent citations."""

REWRITE_SYSTEM = """Rewrite the user's latest message as a standalone search question,
using the conversation for context. If it is already standalone, return it unchanged.
Output only the question, nothing else."""


class RAGChatbot:
    def __init__(self, docs, engines, top_n=RAG_TOP_N):
        self.docs = docs
        self.engines = engines
        self.top_n = top_n
        self.history = []          # list of (question, answer)

    def reset(self):
        self.history = []

    def _standalone(self, question):
        """Turn a follow-up like 'what about women?' into a self-contained question."""
        if not self.history:
            return question
        convo = "\n".join(f"User: {q}\nAssistant: {a[:300]}" for q, a in self.history[-3:])
        prompt = f"Conversation so far:\n{convo}\n\nNew message: {question}\n\nStandalone question:"
        return ask_llm(prompt, system=REWRITE_SYSTEM, temperature=0) or question

    def _format_sources(self, results):
        blocks = []
        for i, _ in results:
            d = self.docs[i]
            blocks.append(f"[{d['id']}] {d['title']}\n{d['text'][:MAX_ABSTRACT_CHARS]}")
        return "\n\n".join(blocks)

    def ask(self, question):
        t0 = time.time()
        standalone = self._standalone(question)
        t1 = time.time()
        results = self.engines["reranked"].search(standalone, self.top_n)
        t2 = time.time()

        prompt = f"Sources:\n\n{self._format_sources(results)}\n\nQuestion: {standalone}"
        answer = ask_llm(prompt, system=SYSTEM_PROMPT) or (
            "Sorry, I could not generate an answer. Please try again."
        )
        t3 = time.time()

        self.history.append((question, answer))
        return {
            "answer": answer,
            "standalone": standalone,
            "sources": [(self.docs[i], score) for i, score in results],
            "timings": {"rewrite": t1 - t0, "search": t2 - t1, "llm_answer": t3 - t2},
        }
