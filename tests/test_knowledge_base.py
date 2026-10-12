import pytest

from askdocs.config import Settings
from askdocs.embeddings import HashingEmbedder
from askdocs.knowledge_base import KnowledgeBase


def test_ingest_and_search_modes(loaded_kb):
    for mode in ("dense", "bm25", "hybrid"):
        hits = loaded_kb.search("What is the F1 score?", k=3, mode=mode)
        assert hits and hits[0].source == "model_evaluation.md"


def test_unknown_mode_rejected(loaded_kb):
    with pytest.raises(ValueError):
        loaded_kb.search("x", mode="magic")


def test_reingest_replaces_document(kb):
    kb.ingest("a.md", "Cats are small mammals. They purr when content.")
    kb.ingest("a.md", "Dogs are loyal animals. They bark at strangers.")
    stats = kb.stats()
    assert stats["n_documents"] == 1
    hits = kb.search("dogs bark", k=1)
    assert "Dogs" in hits[0].text


def test_delete_document(loaded_kb):
    assert loaded_kb.delete("rag_overview.md") is True
    assert loaded_kb.delete("rag_overview.md") is False
    assert "rag_overview.md" not in loaded_kb.stats()["documents"]


def test_persistence_roundtrip(settings):
    first = KnowledgeBase(settings, HashingEmbedder())
    first.ingest("a.md", "Quantization stores weights in lower precision.")
    second = KnowledgeBase(settings, HashingEmbedder())  # reload from SQLite
    assert second.stats()["n_chunks"] == 1
    assert second.search("quantization", k=1)[0].source == "a.md"


def test_embedder_mismatch_is_detected(settings):
    KnowledgeBase(settings, HashingEmbedder()).ingest("a.md", "Some text here.")
    other = Settings(storage_dir=settings.storage_dir, embedder="hashing", sbert_model="other-model")
    with pytest.raises(RuntimeError):
        KnowledgeBase(other, HashingEmbedder())


def test_empty_kb_returns_no_hits(kb):
    assert kb.search("anything") == []
