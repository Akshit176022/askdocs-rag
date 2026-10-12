from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .text import tokenize
from .vectorindex import VectorIndex

RRF_K = 60  # standard constant for Reciprocal Rank Fusion

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL UNIQUE,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    source: str
    position: int
    text: str
    score: float


class KnowledgeBase:
    """Document store + hybrid (dense + BM25) retriever.

    SQLite is the source of truth (chunk text and embeddings). FAISS and BM25
    are in-memory structures rebuilt from SQLite on start-up and after
    ingestion, so no document is ever re-embedded just to restore the index.
    """

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(Path(settings.storage_dir) / "askdocs.db"), check_same_thread=False
        )
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = 256
        self._check_embedder_compat()
        self._chunks: list[Hit] = []
        self._index = VectorIndex(embedder.dim)
        self._bm25 = BM25()
        self._rebuild()

    # ---- bookkeeping -----------------------------------------------------
    def _check_embedder_compat(self) -> None:
        sig = f"{self.settings.embedder}:{self.settings.sbert_model}:{self.embedder.dim}"
        row = self._db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO meta VALUES ('embedder', ?)", (sig,))
            self._db.commit()
        elif row[0] != sig:
            raise RuntimeError(
                f"Index was built with embedder '{row[0]}' but current config is '{sig}'. "
                "Use a different ASKDOCS_STORAGE_DIR or delete the old storage."
            )

    def _rebuild(self) -> None:
        rows = self._db.execute(
            "SELECT c.id, d.source, c.position, c.text, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
        ).fetchall()
        self._chunks = [Hit(r[0], r[1], r[2], r[3], 0.0) for r in rows]
        self._index = VectorIndex(self.embedder.dim)
        if rows:
            vecs = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
            self._index.add(vecs)
        self._bm25 = BM25().fit([tokenize(h.text) for h in self._chunks])

    # ---- ingestion ---------------------------------------------------------
    def ingest_many(self, docs: list[tuple[str, str]]) -> dict[str, int]:
        """Ingest (source, text) pairs; re-ingesting a source replaces it."""
        counts: dict[str, int] = {}
        with self._lock:
            for source, text in docs:
                chunks = chunk_text(
                    text, self.settings.chunk_words, self.settings.chunk_overlap
                )
                counts[source] = len(chunks)
                self._db.execute("DELETE FROM documents WHERE source = ?", (source,))
                if not chunks:
                    continue
                vecs = self.embedder.embed(chunks)
                cur = self._db.execute("INSERT INTO documents(source) VALUES (?)", (source,))
                doc_id = cur.lastrowid
                self._db.executemany(
                    "INSERT INTO chunks(doc_id, position, text, embedding) VALUES (?,?,?,?)",
                    [
                        (doc_id, i, c, vecs[i].astype(np.float32).tobytes())
                        for i, c in enumerate(chunks)
                    ],
                )
            self._db.commit()
            self._rebuild()
        return counts

    def ingest(self, source: str, text: str) -> int:
        return self.ingest_many([(source, text)])[source]

    def delete(self, source: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM documents WHERE source = ?", (source,))
            self._db.commit()
            self._rebuild()
            return cur.rowcount > 0

    def stats(self) -> dict:
        docs = self._db.execute("SELECT source FROM documents ORDER BY source").fetchall()
        return {
            "documents": [d[0] for d in docs],
            "n_documents": len(docs),
            "n_chunks": len(self._chunks),
            "embedder": self.settings.embedder,
        }

    # ---- retrieval ---------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached
        vec = self.embedder.embed([query])[0]
        self._query_cache[query] = vec
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)
        return vec

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        """Retrieve the top-k chunks.

        mode: 'dense' (FAISS cosine), 'bm25' (lexical) or 'hybrid'
        (Reciprocal Rank Fusion of the two rankings).
        """
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        with self._lock:
            if not self._chunks:
                return []
            depth = max(k * 4, 20)
            dense = self._index.search(self._embed_query(query), depth) if mode != "bm25" else []
            lexical = self._bm25.search(tokenize(query), depth) if mode != "dense" else []

            if mode == "dense":
                ranked = dense
            elif mode == "bm25":
                ranked = lexical
            else:
                fused: dict[int, float] = {}
                for ranking in (dense, lexical):
                    for rank, (idx, _) in enumerate(ranking):
                        fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
                ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

            out = []
            for idx, score in ranked[:k]:
                h = self._chunks[idx]
                out.append(Hit(h.chunk_id, h.source, h.position, h.text, float(score)))
            return out
