from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from environment variables."""

    storage_dir: str = "./storage"
    embedder: str = "sbert"  # sbert | hashing
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator: str = "extractive"  # extractive | anthropic | hf
    anthropic_model: str = "claude-sonnet-5-5"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    chunk_words: int = 120
    chunk_overlap: int = 30
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            storage_dir=os.getenv("ASKDOCS_STORAGE_DIR", d.storage_dir),
            embedder=os.getenv("ASKDOCS_EMBEDDER", d.embedder),
            sbert_model=os.getenv("ASKDOCS_SBERT_MODEL", d.sbert_model),
            generator=os.getenv("ASKDOCS_GENERATOR", d.generator),
            anthropic_model=os.getenv("ASKDOCS_ANTHROPIC_MODEL", d.anthropic_model),
            hf_model=os.getenv("ASKDOCS_HF_MODEL", d.hf_model),
            chunk_words=int(os.getenv("ASKDOCS_CHUNK_WORDS", d.chunk_words)),
            chunk_overlap=int(os.getenv("ASKDOCS_CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(os.getenv("ASKDOCS_TOP_K", d.top_k)),
        )
