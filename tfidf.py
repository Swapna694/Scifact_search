"""TF-IDF keyword search, implemented from scratch.

  tf(t, d)  = 1 + log(count of t in d)          (sublinear tf)
  idf(t)    = log((1 + N) / (1 + df(t))) + 1
  weight    = tf * idf
  score     = cosine(query vector, document vector)
"""
import math
from collections import Counter, defaultdict

import numpy as np

from preprocessing import tokenize


class TFIDFSearch:
    def fit(self, tokenized_docs):
        self.n_docs = len(tokenized_docs)

        df = Counter()
        for tokens in tokenized_docs:
            df.update(set(tokens))
        self.idf = {t: math.log((1 + self.n_docs) / (1 + d)) + 1 for t, d in df.items()}

        # inverted index: term -> [(doc_index, L2-normalised tf-idf weight)]
        self.index = defaultdict(list)
        for i, tokens in enumerate(tokenized_docs):
            weights = {t: (1 + math.log(c)) * self.idf[t] for t, c in Counter(tokens).items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for t, w in weights.items():
                self.index[t].append((i, w / norm))
        return self

    def search(self, query, k=3):
        counts = Counter(t for t in tokenize(query) if t in self.idf)
        if not counts:
            return []
        q_weights = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

        scores = np.zeros(self.n_docs)
        for t, qw in q_weights.items():
            for i, dw in self.index[t]:
                scores[i] += qw * dw
        scores /= q_norm  # now equals cosine similarity

        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]