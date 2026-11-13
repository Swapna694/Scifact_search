def print_results(docs, results, relevant=None, indent=2):
    pad = " " * indent
    if not results:
        print(f"{pad}(no results)")
    for rank, (i, score) in enumerate(results, 1):
        d = docs[i]
        mark = "  <-- RELEVANT" if relevant and d["id"] in relevant else ""
        print(f"{pad}{rank}. [{score:.3f}] {d['title']} (id: {d['id']}){mark}")
        print(f"{pad}   {d['text'][:150].replace(chr(10), ' ')}...")