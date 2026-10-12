import os
from pathlib import Path

import pytest

os.environ.setdefault("ASKDOCS_EMBEDDER", "hashing")

from askdocs.config import Settings
from askdocs.embeddings import HashingEmbedder
from askdocs.generator import ExtractiveGenerator
from askdocs.knowledge_base import KnowledgeBase
from askdocs.loaders import load_directory
from askdocs.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_dir=str(tmp_path / "store"), embedder="hashing")


@pytest.fixture
def kb(settings):
    return KnowledgeBase(settings, HashingEmbedder())


@pytest.fixture
def loaded_kb(kb):
    kb.ingest_many(load_directory(ROOT / "data" / "sample_docs"))
    return kb


@pytest.fixture
def pipeline(loaded_kb):
    return RAGPipeline(loaded_kb, ExtractiveGenerator())
