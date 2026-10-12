from __future__ import annotations

import numpy as np


class VectorIndex:
    """Inner-product nearest-neighbour index (FAISS, with a NumPy fallback).

    Vectors are L2-normalised, so inner product equals cosine similarity.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._n = 0
        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self._mat = None
        except ImportError:  # pragma: no cover - exercised only without faiss
            self._index = None
            self._mat = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._n

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) array, got {vecs.shape}")
        if self._index is not None:
            self._index.add(vecs)
        else:  # pragma: no cover
            self._mat = np.vstack([self._mat, vecs])
        self._n += len(vecs)

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._n == 0:
            return []
        k = min(k, self._n)
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self._mat @ q[0]  # pragma: no cover
        top = np.argsort(-sims)[:k]  # pragma: no cover
        return [(int(i), float(sims[i])) for i in top]  # pragma: no cover
