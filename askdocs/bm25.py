from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict


class BM25:
    """Okapi BM25 over an inverted index (implemented from scratch).

    Indexing is O(total tokens). A query touches only the posting lists of its
    terms, so search is O(sum of posting lengths) plus O(m log k) for top-k.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.avgdl = 0.0

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.postings = defaultdict(list)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / len(docs)) if docs else 0.0
        for i, toks in enumerate(docs):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        return self

    def search(self, query_tokens: list[str], k: int = 5) -> list[tuple[int, float]]:
        n_docs = len(self.doc_len)
        if n_docs == 0 or self.avgdl == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
