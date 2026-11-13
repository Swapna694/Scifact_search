"""BM25 keyword search, implemented from scratch.

  idf(t)   = log(1 + (N - df + 0.5) / (df + 0.5))
  score    = sum over query terms of
             idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

Unlike TF-IDF, BM25 saturates term frequency (k1) and normalises by
document length (b).
"""
import math
from collections import Counter, defaultdict

import numpy as np

from config import BM25_B, BM25_K1
from preprocessing import tokenize


class BM25Search:
    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b

    def fit(self, tokenized_docs):
        self.n_docs = len(tokenized_docs)
        self.doc_len = np.array([len(t) for t in tokenized_docs], dtype=float)
        self.avgdl = self.doc_len.mean()

        df = Counter()
        self.index = defaultdict(list)  # term -> [(doc_index, tf)]
        for i, tokens in enumerate(tokenized_docs):
            for t, c in Counter(tokens).items():
                df[t] += 1
                self.index[t].append((i, c))

        self.idf = {
            t: math.log(1 + (self.n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }
        return self

    def search(self, query, k=3):
        scores = np.zeros(self.n_docs)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i, tf in self.index[t]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += self.idf[t] * tf * (self.k1 + 1) / norm

        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]