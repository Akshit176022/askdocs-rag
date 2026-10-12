from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class HashingEmbedder:
    """Offline, dependency-light embedder (hashed uni/bi-gram TF, L2-normalised).

    Lexical rather than semantic: used for tests, CI and offline demos. Use
    ``SentenceTransformerEmbedder`` for real semantic search.
    """

    def __init__(self, dim: int = 2048) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    """Hugging Face sentence-transformers model (default: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "sbert":
        return SentenceTransformerEmbedder(settings.sbert_model)
    raise ValueError(f"Unknown embedder: {settings.embedder!r}")
