from __future__ import annotations

import time
from dataclasses import dataclass, field

from .generator import NO_ANSWER, Generator
from .knowledge_base import Hit, KnowledgeBase


@dataclass
class RAGResult:
    question: str
    answer: str
    abstained: bool
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """retrieve -> build prompt -> generate, with per-stage latency tracking."""

    def __init__(self, kb: KnowledgeBase, generator: Generator) -> None:
        self.kb = kb
        self.generator = generator

    def answer(self, question: str, k: int = 4, mode: str = "hybrid") -> RAGResult:
        t0 = time.perf_counter()
        hits = self.kb.search(question, k=k, mode=mode)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, hits) if hits else NO_ANSWER
        t2 = time.perf_counter()
        return RAGResult(
            question=question,
            answer=answer,
            abstained=answer.strip().startswith(NO_ANSWER[:20]),
            hits=hits,
            timings_ms={
                "retrieval": (t1 - t0) * 1000,
                "generation": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        )
