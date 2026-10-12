from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .embeddings import get_embedder
from .evaluation import evaluate, format_table, load_eval_set
from .generator import get_generator
from .knowledge_base import KnowledgeBase
from .loaders import load_directory
from .pipeline import RAGPipeline


def _build(settings: Settings) -> RAGPipeline:
    kb = KnowledgeBase(settings, get_embedder(settings))
    return RAGPipeline(kb, get_generator(settings))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="askdocs", description="Ask questions over your documents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index a directory of .txt/.md/.pdf files")
    s.add_argument("path")

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=None)
    s.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")

    s = sub.add_parser("eval", help="evaluate retrieval and answers on a labelled set")
    s.add_argument("--eval-set", default="data/eval/qa.jsonl")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--json", action="store_true", help="print raw JSON instead of a table")

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("askdocs.api:app", host=args.host, port=args.port)
        return 0

    pipe = _build(settings)

    if args.cmd == "ingest":
        counts = pipe.kb.ingest_many(load_directory(args.path))
        for src, n in counts.items():
            print(f"{src}: {n} chunks")
        print(json.dumps(pipe.kb.stats(), indent=2))
    elif args.cmd == "ask":
        res = pipe.answer(args.question, k=args.k or settings.top_k, mode=args.mode)
        print(res.answer)
        print("\nSources:")
        for i, h in enumerate(res.hits, 1):
            print(f"  [{i}] {h.source} (chunk {h.position}, score {h.score:.3f})")
        print(f"\nlatency: {res.timings_ms['total']:.1f} ms")
    elif args.cmd == "eval":
        items = load_eval_set(args.eval_set)
        results = [evaluate(pipe, items, k=args.k, mode=m) for m in ("dense", "bm25", "hybrid")]
        print(json.dumps(results, indent=2) if args.json else format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
