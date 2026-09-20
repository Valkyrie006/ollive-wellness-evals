"""Lightweight FastAPI backend (plan.md Section 3). One /chat route serves
both agents by `agent` field ("oss" or "frontier") - both call the exact
same agents.core.run_turn with only the model config swapped.
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.config import AGENTS, api_key_for
from agents.core import run_turn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("wellness")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state()
    yield


app = FastAPI(title="Wellness Assistant POC", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    agent: str  # "oss" | "frontier"
    message: str


class ChatResponse(BaseModel):
    response: str
    tool_calls: list
    latency_ms: float


class ResetRequest(BaseModel):
    session_id: str


def initialize_app_state(embed_fn=None, chroma_client=None, source_dir=None):
    """Builds the KB collection + embedder once at startup and stores them on
    app.state. Split out from the @app.on_event so tests can call it with
    fakes instead of downloading real model weights / hitting real Chroma.
    """
    from kb.ingest import build_kb, DEFAULT_SOURCE_DIR

    kwargs = {}
    if embed_fn is not None:
        kwargs["embed_fn"] = embed_fn
    if chroma_client is not None:
        kwargs["chroma_client"] = chroma_client
    kwargs["source_dir"] = source_dir or DEFAULT_SOURCE_DIR

    client, coll = build_kb(**kwargs)
    app.state.chroma_client = client
    app.state.kb_collection = coll
    app.state.embed_fn = embed_fn or _default_embed_fn_singleton()


def _default_embed_fn_singleton():
    from kb.ingest import default_embed_fn
    return default_embed_fn


@app.get("/health")
def health():
    ready = hasattr(app.state, "kb_collection")
    return {"status": "ok", "kb_loaded": ready}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"agent must be one of {list(AGENTS)}")
    if not hasattr(app.state, "kb_collection"):
        raise HTTPException(status_code=503, detail="KB not loaded yet")

    model_config = AGENTS[req.agent]
    api_key = api_key_for(model_config)

    start = time.perf_counter()
    try:
        result = run_turn(
            session_id=req.session_id,
            user_message=req.message,
            model_config=model_config,
            api_key=api_key,
            coll=app.state.kb_collection,
            embed_fn=app.state.embed_fn,
        )
    except Exception as e:
        # Surface the real cause (bad/missing API key, rate limit, network
        # issue, provider outage) instead of a bare 500 - this is the one
        # thing most likely to go wrong on a first real run. Full traceback
        # also goes to the server log so it's visible in the uvicorn terminal.
        logger.exception("chat failed for agent=%s model=%s", req.agent, model_config["model"])
        raise HTTPException(
            status_code=502,
            detail=f"Upstream call to '{model_config['model']}' failed: {type(e).__name__}: {e}",
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(response=result["response"], tool_calls=result["tool_calls"], latency_ms=latency_ms)


def _fetch_available_models(provider: str, key: str) -> list[str]:
    """Ask the provider what it actually serves for this key. Providers retire
    model IDs on their own schedule, so this is the authoritative answer to
    'what should I put in config' - no guessing from docs that may be stale.
    """
    import requests
    from agents.config import MODEL_LIST_ENDPOINTS

    url = MODEL_LIST_ENDPOINTS.get(provider)
    if not url:
        return []
    if provider == "groq":
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    if provider == "gemini":
        r = requests.get(url, params={"key": key, "pageSize": 200}, timeout=20)
        r.raise_for_status()
        names = []
        for m in r.json().get("models", []):
            if "generateContent" in (m.get("supportedGenerationMethods") or ["generateContent"]):
                names.append(m["name"].replace("models/", ""))
        return sorted(names)
    return []


@app.get("/available-models")
def available_models():
    """Live model list per provider, for filling in OSS_MODEL / FRONTIER_MODEL
    / JUDGE_MODEL in .env when a provider retires an ID.
    """
    out = {}
    for provider in {cfg["provider"] for cfg in AGENTS.values()}:
        cfg = next(c for c in AGENTS.values() if c["provider"] == provider)
        key = api_key_for(cfg)
        if not key:
            out[provider] = {"status": "missing_key", "models": []}
            continue
        try:
            out[provider] = {"status": "ok", "models": _fetch_available_models(provider, key)}
        except Exception as e:
            out[provider] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:300]}", "models": []}
    return out


@app.get("/diagnostics")
def diagnostics():
    """One-shot self-check of every external dependency, so a failing /chat
    can be diagnosed without reading server logs: is each provider key
    present, is it actually accepted by the provider, is the KB loaded, does
    the real embedder work, and does web search work.
    """
    import litellm

    report = {"agents": [], "kb": {}, "embedder": {}, "web_search": {}}

    for name, cfg in AGENTS.items():
        entry = {"agent": name, "model": cfg["model"], "key_env": cfg["api_key_env"]}
        key = api_key_for(cfg)
        if not key:
            entry.update(
                status="missing_key",
                detail=f"{cfg['api_key_env']} is not set. Add it to .env and restart the server.",
            )
            report["agents"].append(entry)
            continue
        entry["key_prefix"] = key[:6] + "..."
        entry["key_length"] = len(key)
        try:
            resp = litellm.completion(
                model=cfg["model"],
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                api_key=key,
                max_tokens=8,
            )
            entry.update(status="ok", sample=(resp.choices[0].message.content or "").strip()[:60])
        except Exception as e:
            logger.exception("diagnostics: agent %s failed", name)
            entry.update(status="error", error_type=type(e).__name__, detail=str(e)[:600])
            # A retired model ID is the single most likely failure here, and
            # it's only actionable if you know what IS available - so fetch
            # the live list and attach it rather than making the user guess.
            if "not_found" in str(e).lower() or "no longer available" in str(e).lower():
                try:
                    entry["available_models"] = _fetch_available_models(cfg["provider"], key)
                except Exception as le:
                    entry["available_models_error"] = f"{type(le).__name__}: {str(le)[:200]}"
        report["agents"].append(entry)

    try:
        report["kb"] = {"status": "ok", "chunks": app.state.kb_collection.count()}
    except Exception as e:
        report["kb"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    try:
        vec = app.state.embed_fn(["healthy sleep habits"])
        report["embedder"] = {"status": "ok", "dimensions": len(vec[0])}
    except Exception as e:
        report["embedder"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    try:
        from agents.tools import search_web
        results = search_web("wellness habits", max_results=2)
        report["web_search"] = {"status": "ok", "results": len(results)}
    except Exception as e:
        report["web_search"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    return report


@app.post("/reset")
def reset(req: ResetRequest):
    """Clears a session's short-term memory (used by the UI's New chat button)."""
    from agents.core import reset_session
    reset_session(req.session_id)
    return {"status": "ok"}


# Serve the minimal chat UI at /
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
