"""Exercises the FastAPI wiring (routes, request/response schema, error
handling) with the real app but a fake embedder + in-memory Chroma, and the
LLM call monkeypatched - so this proves the HTTP layer is correct without
needing real network access or API keys.
"""
import chromadb
from fastapi.testclient import TestClient

import agents.core as core
from api.main import app, initialize_app_state
from tests.fakes import fake_embed_fn, FakeMessage, make_scripted_completion_fn


def make_client(monkeypatch, script):
    core.SESSIONS.clear()
    client = chromadb.Client()
    initialize_app_state(embed_fn=fake_embed_fn, chroma_client=client)
    monkeypatch.setattr(core, "run_turn", core.run_turn)  # no-op, keeps real loop
    monkeypatch.setattr("litellm.completion", make_scripted_completion_fn(script), raising=False)
    return TestClient(app)


def test_health_endpoint_reports_kb_loaded(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["kb_loaded"] is True


def test_chat_endpoint_end_to_end_with_fake_llm(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="Try a short walk daily.", tool_calls=None)])
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "oss", "message": "Any exercise tips?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Try a short walk daily."
    assert isinstance(body["tool_calls"], list)
    assert body["latency_ms"] >= 0


def test_chat_endpoint_rejects_unknown_agent(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "not-a-real-agent", "message": "hi"})
    assert resp.status_code == 400
