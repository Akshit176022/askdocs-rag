from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .pipeline import RAGPipeline
from .text import tokenize


@dataclass
class EvalItem:
    question: str
    answerable: bool
    gold_source: str | None
    gold_keywords: list[str]


def load_eval_set(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(
                EvalItem(
                    d["question"],
                    bool(d["answerable"]),
                    d.get("gold_source"),
                    d.get("gold_keywords", []),
                )
            )
    return items


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content tokens that appear in the retrieved context.

    A cheap, deterministic hallucination proxy: 1.0 means every content word in
    the answer is supported by the context; low values flag unsupported text.
    Citation markers like [1] are ignored.
    """
    import re

    toks = tokenize(re.sub(r"\[\d+\]", " ", answer))
    if not toks:
        return 1.0
    ctx = set(tokenize(context))
    return sum(t in ctx for t in toks) / len(toks)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def evaluate(pipeline: RAGPipeline, items: list[EvalItem], k: int = 4, mode: str = "hybrid") -> dict:
    """Run the pipeline on a labelled set and compute retrieval + answer metrics."""
    ans = [i for i in items if i.answerable]
    unans = [i for i in items if not i.answerable]

    hit1 = hitk = rr = 0.0
    kw_scores: list[float] = []
    grounded: list[float] = []
    false_abstain = 0
    correct_abstain = 0
    latencies: list[float] = []

    for item in ans:
        res = pipeline.answer(item.question, k=k, mode=mode)
        latencies.append(res.timings_ms["total"])
        sources = [h.source for h in res.hits]
        if sources[:1] == [item.gold_source]:
            hit1 += 1
        if item.gold_source in sources:
            hitk += 1
            rr += 1.0 / (sources.index(item.gold_source) + 1)
        if res.abstained:
            false_abstain += 1
            kw_scores.append(0.0)
            continue
        low = res.answer.lower()
        kws = item.gold_keywords
        kw_scores.append(sum(kw.lower() in low for kw in kws) / len(kws) if kws else 1.0)
        grounded.append(groundedness(res.answer, " ".join(h.text for h in res.hits)))

    for item in unans:
        res = pipeline.answer(item.question, k=k, mode=mode)
        latencies.append(res.timings_ms["total"])
        correct_abstain += int(res.abstained)

    n = max(len(ans), 1)
    return {
        "mode": mode,
        "k": k,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "hit@1": hit1 / n,
        f"hit@{k}": hitk / n,
        "mrr": rr / n,
        "keyword_accuracy": statistics.mean(kw_scores) if kw_scores else 0.0,
        "groundedness": statistics.mean(grounded) if grounded else 0.0,
        "abstention_accuracy": (correct_abstain / len(unans)) if unans else None,
        "false_abstain_rate": false_abstain / n,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


def format_table(results: list[dict]) -> str:
    if not results:
        return ""
    k = results[0]["k"]
    cols = [
        ("mode", "mode"),
        ("hit@1", "hit@1"),
        (f"hit@{k}", f"hit@{k}"),
        ("mrr", "MRR"),
        ("keyword_accuracy", "kw-acc"),
        ("groundedness", "grounded"),
        ("abstention_accuracy", "abstain-acc"),
        ("false_abstain_rate", "false-abs"),
        ("latency_p50_ms", "p50 ms"),
        ("latency_p95_ms", "p95 ms"),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for r in results:
        cells = []
        for key, _ in cols:
            v = r[key]
            cells.append(v if isinstance(v, str) else ("n/a" if v is None else f"{v:.2f}"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])
