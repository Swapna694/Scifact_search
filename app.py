"""Simple web UI for the SciFact search project (Streamlit).

Run:  streamlit run app.py
"""
import streamlit as st

from config import TOP_K
from data_loader import load_corpus
from display import matched_words
from engines import build_engines

st.set_page_config(page_title="SciFact Search", layout="wide")


@st.cache_resource(show_spinner="Loading data and building indexes (the first run can take a few minutes)...")
def load():
    docs = load_corpus()
    return docs, build_engines(docs)


docs, engines = load()


def show(results, query=None):
    """Render a ranked result list. Pass `query` to also show matched words."""
    if not results:
        st.info("No results.")
        return
    for rank, (i, score) in enumerate(results, 1):
        d = docs[i]
        st.markdown(f"**{rank}. {d['title']}**")
        caption = f"score {score:.3f}  |  id {d['id']}"
        if query is not None:
            words, total = matched_words(query, d)
            caption += f"  |  matched words ({len(words)}/{total}): {', '.join(words) or '-'}"
        st.caption(caption)
        with st.expander("Abstract"):
            st.write(d["text"])


st.title("SciFact Search")
st.write("Compare keyword search (TF-IDF, BM25), semantic search and reranking.")

k = st.sidebar.slider("Results per method", 1, 10, TOP_K)
query = st.text_input("Enter a scientific query",
                      placeholder="e.g. Aspirin reduces the risk of colorectal cancer")

if query.strip():
    tab_kw, tab_sem, tab_rr = st.tabs(["Keyword search", "Semantic search", "Reranked"])

    with tab_kw:
        left, right = st.columns(2)
        with left:
            st.subheader("TF-IDF")
            show(engines["tfidf"].search(query, k), query=query)
        with right:
            st.subheader("BM25")
            show(engines["bm25"].search(query, k), query=query)

    with tab_sem:
        show(engines["semantic"].search(query, k))

    with tab_rr:
        st.caption("Top BM25 + semantic candidates, re-scored by a cross-encoder.")
        show(engines["reranked"].search(query, k))