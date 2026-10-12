import pytest

from askdocs.bm25 import BM25
from askdocs.chunking import chunk_text
from askdocs.text import tokenize


def test_chunks_respect_max_words_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    chunks = chunk_text(text, max_words=30, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 30 for c in chunks)
    # the last sentence of one chunk is repeated at the start of the next
    first_tail = chunks[0].split(". ")[-1].rstrip(".")
    assert first_tail.split(" has")[0] in chunks[1]


def test_chunk_text_validates_arguments():
    with pytest.raises(ValueError):
        chunk_text("hello", max_words=10, overlap_words=10)
    with pytest.raises(ValueError):
        chunk_text("hello", max_words=0)


def test_chunk_text_empty_and_short():
    assert chunk_text("") == []
    assert chunk_text("One short sentence.") == ["One short sentence."]


def test_bm25_ranks_matching_document_first():
    docs = [
        "faiss performs similarity search over vectors",
        "docker packages applications into containers",
        "overfitting memorises training data",
    ]
    bm = BM25().fit([tokenize(d) for d in docs])
    top = bm.search(tokenize("what does docker do"), k=2)
    assert top[0][0] == 1
    assert bm.search(tokenize("zzzz unknown"), k=3) == []


def test_bm25_empty_index():
    assert BM25().fit([]).search(["a"], k=3) == []
