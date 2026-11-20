"""Streamlit RAG chat page.   Run:  streamlit run chat_app.py"""
import streamlit as st

from data_loader import load_corpus
from engines import build_engines
from rag import RAGChatbot

st.set_page_config(page_title="SciScout")


@st.cache_resource(show_spinner="Loading data and building indexes (the first run can take a few minutes)...")
def load():
    docs = load_corpus()
    return docs, build_engines(docs)


docs, engines = load()

if "bot" not in st.session_state:
    st.session_state.bot = RAGChatbot(docs, engines)
    st.session_state.messages = []

st.title("SciScout")
st.caption("Answers are written by an LLM using only the retrieved papers. Always check the sources.")

if st.sidebar.button("New chat"):
    st.session_state.bot.reset()
    st.session_state.messages = []
    st.rerun()


def show_sources(sources):
    with st.expander("Sources"):
        for d, _ in sources:
            st.markdown(f"**[{d['id']}] {d['title']}**")
            st.write(d["text"][:400] + "...")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask about the papers, or state a claim to check...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and writing an answer..."):
            try:
                r = st.session_state.bot.ask(question)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(r["answer"])
        t = r["timings"]
        st.caption(f"Time: rewrite {t['rewrite']:.1f}s | search + rerank {t['search']:.1f}s | LLM answer {t['llm_answer']:.1f}s")
        show_sources(r["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": r["answer"], "sources": r["sources"]}
    )