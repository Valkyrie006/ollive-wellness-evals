"""Exercises the HTTP layer - routes, schemas, validation, error handling,
rate limiting and debug gating - against the real FastAPI app, with a fake
embedder + in-memory Chroma and the LLM call monkeypatched. No network, no
API keys, real code paths.
"""
import chromadb
import pytest
from fastapi.testclient import TestClient

import agents.core as core
from api.main import app, initialize_app_state, rate_limiter
from settings import settings
from tests.fakes import FakeMessage, fake_embed_fn, make_scripted_completion_fn


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Each test starts with empty sessions, a clear rate-limit window, and
    a key present so /chat isn't rejected before it reaches the loop."""
    core.SESSIONS.clear()
    rate_limiter.reset()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_not_real")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key_not_real")
    yield
    rate_limiter.reset()


def make_client(monkeypatch, script):
    client = chromadb.Client()
    initialize_app_state(embed_fn=fake_embed_fn, chroma_client=client)
    monkeypatch.setattr("litellm.completion", make_scripted_completion_fn(script), raising=False)
    return TestClient(app)


def test_health_is_dependency_free(monkeypatch):
    """Liveness must not depend on the KB - an orchestrator shouldn't kill a
    pod because an upstream is slow."""
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_kb_loaded(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["kb_chunks"] > 0


def test_agents_endpoint_exposes_configured_models(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    body = tc.get("/agents").json()
    names = {a["name"] for a in body["agents"]}
    assert names == {"oss", "frontier"}
    assert all(a["model"] for a in body["agents"])


def test_chat_endpoint_end_to_end_with_fake_llm(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="Try a short walk daily.", tool_calls=None)])
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "oss", "message": "Any exercise tips?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Try a short walk daily."
    assert isinstance(body["tool_calls"], list)
    assert body["latency_ms"] >= 0
    assert resp.headers.get("X-Request-ID")


def test_chat_rejects_unknown_agent(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "nope", "message": "hi"})
    assert resp.status_code == 400


def test_chat_rejects_oversized_message(monkeypatch):
    """An unbounded message field on an endpoint backed by a paid upstream is
    a cost and abuse vector."""
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    huge = "x" * (settings.max_message_chars + 1)
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "oss", "message": huge})
    assert resp.status_code == 422


def test_chat_rejects_blank_message(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "oss", "message": "   "})
    assert resp.status_code == 422


def test_chat_returns_503_when_key_missing(monkeypatch):
    """A missing key should be a clear server-side 503, not a confusing
    upstream auth error surfaced as a 502.

    The key name is read from the agent's config rather than hardcoded:
    which provider the OSS agent sits on is a .env override away, and a
    test that pins one provider fails for the wrong reason when it moves.
    """
    from agents.config import AGENTS

    key_env = AGENTS["oss"]["api_key_env"]
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    monkeypatch.delenv(key_env, raising=False)
    resp = tc.post("/chat", json={"session_id": "t1", "agent": "oss", "message": "hi"})
    assert resp.status_code == 503
    assert key_env in resp.json()["detail"]


def test_security_headers_are_applied(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    headers = tc.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_rate_limit_blocks_a_burst(monkeypatch):
    """Past the window budget the endpoint must return 429 with Retry-After
    rather than forwarding every request to a paid upstream."""
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    payload = {"session_id": "burst", "agent": "oss", "message": "hi"}

    statuses = [tc.post("/chat", json=payload).status_code
                for _ in range(settings.rate_limit_requests + 3)]

    assert 429 in statuses, "expected the burst to be rate limited"
    limited = tc.post("/chat", json=payload)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_reset_clears_session_memory(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="Hi there.", tool_calls=None)])
    tc.post("/chat", json={"session_id": "s-reset", "agent": "oss", "message": "My name is Sam."})
    assert core.get_history("s-reset")
    assert tc.post("/reset", json={"session_id": "s-reset"}).status_code == 200
    assert core.get_history("s-reset") == []


def test_public_config_tells_ui_whether_debug_is_available(monkeypatch):
    tc = make_client(monkeypatch, [FakeMessage(content="ok", tool_calls=None)])
    body = tc.get("/config").json()
    assert "debug_endpoints" in body
    assert body["max_message_chars"] == settings.max_message_chars
