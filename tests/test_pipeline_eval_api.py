from pathlib import Path

from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.evaluation import evaluate, groundedness, load_eval_set, percentile
from askdocs.generator import NO_ANSWER

ROOT = Path(__file__).resolve().parents[1]


def test_answers_with_citation(pipeline):
    res = pipeline.answer("What is quantization?", k=3)
    assert not res.abstained
    assert "lower precision" in res.answer.lower()
    assert "[" in res.answer
    assert set(res.timings_ms) == {"retrieval", "generation", "total"}


def test_abstains_on_out_of_scope_question(pipeline):
    res = pipeline.answer("How do I bake sourdough bread?", k=3)
    assert res.abstained and res.answer == NO_ANSWER


def test_groundedness_scores():
    assert groundedness("FAISS searches vectors [1]", "faiss searches dense vectors") == 1.0
    assert groundedness("Paris is lovely", "faiss searches vectors") == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 95) == 0.0


def test_evaluation_meets_quality_floor(pipeline):
    items = load_eval_set(ROOT / "data" / "eval" / "qa.jsonl")
    res = evaluate(pipeline, items, k=3, mode="hybrid")
    assert res["hit@3"] >= 0.85
    assert res["mrr"] >= 0.7
    assert res["abstention_accuracy"] == 1.0
    assert res["groundedness"] >= 0.9


def test_api_end_to_end(settings):
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["status"] == "ok"

    files = [
        ("files", ("notes.md", b"Docker packages an app with its dependencies into an image.", "text/markdown"))
    ]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200 and r.json()["ingested"]["notes.md"] == 1

    r = client.post("/query", json={"question": "What does Docker package?", "k": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["sources"][0]["source"] == "notes.md"
    assert body["abstained"] is False

    assert client.get("/stats").json()["n_documents"] == 1
    assert client.delete("/documents/notes.md").status_code == 200
    assert client.delete("/documents/notes.md").status_code == 404


def test_api_validation_and_bad_upload(settings):
    client = TestClient(create_app(settings))
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "hi", "mode": "bad"}).status_code == 422
    bad = [("files", ("x.exe", b"abc", "application/octet-stream"))]
    assert client.post("/ingest", files=bad).status_code == 415
