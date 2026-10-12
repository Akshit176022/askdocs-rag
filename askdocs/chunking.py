from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Split into sentence-like units; markdown heading markers are stripped."""
    parts = []
    for p in _SENT_SPLIT.split(text):
        p = re.sub(r"^#+\s*", "", p.strip())
        p = " ".join(p.split())
        if p:
            parts.append(p)
    return parts


def chunk_text(text: str, max_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Sentence-aware sliding-window chunker.

    Sentences are packed into a chunk until adding the next one would exceed
    ``max_words``. The tail of the previous chunk (up to ``overlap_words``) is
    repeated at the start of the next chunk so that facts split across a
    boundary stay retrievable. Runs in O(n) over the number of sentences.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            t = 0
            for prev in reversed(cur):
                pn = len(prev.split())
                if t + pn > overlap_words:
                    break
                tail.insert(0, prev)
                t += pn
            cur, cur_len = tail, t
        cur.append(sent)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks
