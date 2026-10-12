from __future__ import annotations

from typing import Protocol

from .config import Settings
from .knowledge_base import Hit
from .text import tokenize

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use like [1] or [2]. Keep the answer concise. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER}"
)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i}] ({h.source}) {h.text}" for i, h in enumerate(hits, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> str: ...


class ExtractiveGenerator:
    """No-LLM baseline: returns the best-matching context sentences, or abstains.

    Lets the whole pipeline (and its evaluation) run offline and deterministically.
    """

    name = "extractive"

    def __init__(self, min_overlap: float = 0.5, max_sentences: int = 2) -> None:
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        from .chunking import split_sentences

        q = set(tokenize(question))
        if not q or not hits:
            return NO_ANSWER
        scored = []
        for ref, h in enumerate(hits, 1):
            for sent in split_sentences(h.text):
                overlap = len(q & set(tokenize(sent))) / len(q)
                scored.append((overlap, -ref, sent, ref))
        scored.sort(reverse=True)
        best = [s for s in scored if s[0] >= self.min_overlap][: self.max_sentences]
        if not best:
            return NO_ANSWER
        return " ".join(f"{s[2]} [{s[3]}]" for s in best)


class AnthropicGenerator:
    """Generates answers with the Anthropic API (needs ANTHROPIC_API_KEY)."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 400) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[Hit]) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, hits)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


class HFGenerator:
    """Local small language model via Hugging Face transformers (experimental)."""

    name = "hf"

    def __init__(self, model_name: str, max_new_tokens: int = 256) -> None:
        from transformers import pipeline

        self.pipe = pipeline("text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, hits: list[Hit]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits)},
        ]
        out = self.pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


def get_generator(settings: Settings) -> Generator:
    if settings.generator == "extractive":
        return ExtractiveGenerator()
    if settings.generator == "anthropic":
        return AnthropicGenerator(settings.anthropic_model)
    if settings.generator == "hf":
        return HFGenerator(settings.hf_model)
    raise ValueError(f"Unknown generator: {settings.generator!r}")
