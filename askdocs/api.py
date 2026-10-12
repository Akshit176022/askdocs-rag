from __future__ import annotations

import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .embeddings import get_embedder
from .generator import get_generator
from .knowledge_base import KnowledgeBase
from .loaders import read_bytes
from .pipeline import RAGPipeline


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|bm25)$")


class SourceOut(BaseModel):
    source: str
    position: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[SourceOut]
    timings_ms: dict[str, float]


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="AskDocs RAG API", version=__version__)
    lock = threading.Lock()
    state: dict[str, RAGPipeline] = {}

    def pipeline() -> RAGPipeline:
        # Lazy init: the embedding model is loaded on first use, not at import.
        with lock:
            if "p" not in state:
                s = settings or Settings.from_env()
                kb = KnowledgeBase(s, get_embedder(s))
                state["p"] = RAGPipeline(kb, get_generator(s))
            return state["p"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/stats")
    def stats() -> dict:
        return pipeline().kb.stats()

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)) -> dict:
        docs = []
        for f in files:
            try:
                docs.append((f.filename or "upload", read_bytes(f.filename or "", await f.read())))
            except ValueError as e:
                raise HTTPException(status_code=415, detail=str(e)) from e
        counts = pipeline().kb.ingest_many(docs)
        return {"ingested": counts}

    @app.delete("/documents/{source:path}")
    def delete_document(source: str) -> dict:
        if not pipeline().kb.delete(source):
            raise HTTPException(status_code=404, detail="document not found")
        return {"deleted": source}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        res = pipeline().answer(req.question, k=req.k, mode=req.mode)
        return QueryResponse(
            answer=res.answer,
            abstained=res.abstained,
            sources=[
                SourceOut(source=h.source, position=h.position, score=h.score, text=h.text)
                for h in res.hits
            ],
            timings_ms=res.timings_ms,
        )

    return app


app = create_app()
