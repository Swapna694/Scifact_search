from preprocessing import tokenize


def matched_words(query, doc):
    """Query words (after preprocessing) that actually appear in the document."""
    q_tokens = list(dict.fromkeys(tokenize(query)))      # unique, keep order
    doc_tokens = set(tokenize(f"{doc['title']} {doc['text']}"))
    return [t for t in q_tokens if t in doc_tokens], len(q_tokens)


def print_results(docs, results, relevant=None, indent=2, query=None):
    """If `query` is given (keyword search), also show which query words matched."""
    pad = " " * indent
    if not results:
        print(f"{pad}(no results)")
    for rank, (i, score) in enumerate(results, 1):
        d = docs[i]
        mark = "  <-- RELEVANT" if relevant and d["id"] in relevant else ""
        print(f"{pad}{rank}. [{score:.3f}] {d['title']} (id: {d['id']}){mark}")
        if query is not None:
            words, total = matched_words(query, d)
            print(f"{pad}   matched words ({len(words)}/{total}): {', '.join(words) if words else '-'}")
        print(f"{pad}   {d['text'][:150].replace(chr(10), ' ')}...")