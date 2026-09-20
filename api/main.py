"""HTTP layer for the wellness assistant.

One /chat route serves both agents, selected by the `agent` field - both
call the exact same agents.core.run_turn with only the model config
swapped, which is what keeps the architecture "fixed" between them.

Everything beyond that route exists because this is meant to be deployable
in public: request limits, rate limiting, request-scoped logging, and debug
endpoints that are off by default outside development.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agents.config import AGENTS, api_key_for
from agents.core import run_turn
from api.ratelimit import RateLimiter
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("wellness.api")

rate_limiter = RateLimiter(
    limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A KB failure is fatal to the app's purpose, but crashing the process on
    # boot makes it invisible behind a restart loop. Start anyway, mark the
    # app not-ready, and let /ready report why.
    try:
        initialize_app_state()
        app.state.startup_error = None
        logger.info("startup complete: env=%s debug_endpoints=%s",
                    settings.env, settings.enable_debug_endpoints)
    except Exception as e:  # noqa: BLE001 - reported through /ready
        app.state.startup_error = f"{type(e).__name__}: {e}"
        logger.exception("startup failed - serving in not-ready state")
    yield


app = FastAPI(
    title="Wellness Assistant",
    description="Two wellness agents on one fixed architecture, plus an evals harness.",
    version="1.0.0",
    lifespan=lifespan,
    # Interactive API docs are a debug surface too.
    docs_url="/docs" if settings.enable_debug_endpoints else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.enable_debug_endpoints else None,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


# --------------------------------------------------------------------------
# middleware
# --------------------------------------------------------------------------

@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tags every request with an id, logs its outcome, and applies rate
    limiting to the endpoints that cost money.

    The id is returned in the response so a user-visible error can be traced
    to a specific server-side log line without exposing the error itself.
    """
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()

    if request.url.path in RATE_LIMITED_PATHS:
        allowed, retry_after = rate_limiter.check(_client_key(request))
        if not allowed:
            logger.warning("rate limited request_id=%s path=%s", request_id, request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a moment and try again.",
                         "request_id": request_id},
                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
            )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    # Static, conservative headers. The UI is same-origin and uses no CDN,
    # so a strict CSP costs nothing here.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'"
    )
    if request.url.path not in {"/health"}:
        logger.info("request_id=%s %s %s -> %d (%.0fms)",
                    request_id, request.method, request.url.path,
                    response.status_code, elapsed_ms)
    return response


def _client_key(request: Request) -> str:
    """Rate-limit bucket key.

    Behind a reverse proxy the socket peer is the proxy, so the first hop in
    X-Forwarded-For is used when present. That header is client-controlled
    and therefore spoofable - acceptable for basic abuse damping, but real
    protection belongs at the proxy or WAF. See docs/DESIGN.md.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


RATE_LIMITED_PATHS = {"/chat", "/diagnostics", "/available-models"}


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=settings.max_session_id_chars)
    agent: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=settings.max_message_chars)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v


class ChatResponse(BaseModel):
    response: str
    tool_calls: list
    latency_ms: float


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=settings.max_session_id_chars)


# --------------------------------------------------------------------------
# startup
# --------------------------------------------------------------------------

def initialize_app_state(embed_fn=None, chroma_client=None, source_dir=None):
    """Builds the KB collection + embedder once at startup and stores them on
    app.state. Split out from the lifespan hook so tests can call it with
    fakes instead of downloading real model weights.
    """
    from kb.ingest import DEFAULT_SOURCE_DIR, build_kb

    kwargs = {"source_dir": source_dir or DEFAULT_SOURCE_DIR}
    if embed_fn is not None:
        kwargs["embed_fn"] = embed_fn
    if chroma_client is not None:
        kwargs["chroma_client"] = chroma_client

    client, coll = build_kb(**kwargs)
    app.state.chroma_client = client
    app.state.kb_collection = coll
    app.state.embed_fn = embed_fn or _default_embed_fn_singleton()
    app.state.startup_error = None


def _default_embed_fn_singleton():
    from kb.ingest import default_embed_fn
    return default_embed_fn


def _require_debug_endpoints():
    if not settings.enable_debug_endpoints:
        # 404 rather than 403: don't confirm the endpoint exists.
        raise HTTPException(status_code=404, detail="Not found")


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness: is the process up. Deliberately dependency-free so a
    container orchestrator doesn't kill a pod over a slow upstream."""
    return {"status": "ok", "env": settings.env, "version": app.version}


@app.get("/ready")
def ready():
    """Readiness: can this instance actually serve a request."""
    if getattr(app.state, "startup_error", None):
        raise HTTPException(status_code=503, detail=f"Startup failed: {app.state.startup_error}")
    if not hasattr(app.state, "kb_collection"):
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    return {"status": "ready", "kb_chunks": app.state.kb_collection.count()}


@app.get("/agents")
def list_agents():
    """What each agent is currently wired to. The UI labels itself from this
    so the displayed model can't drift from the one being called."""
    return {
        "agents": [
            {"name": name, "model": cfg["model"], "provider": cfg["provider"]}
            for name, cfg in AGENTS.items()
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    request_id = getattr(request.state, "request_id", "-")

    if req.agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"agent must be one of {list(AGENTS)}")
    if not hasattr(app.state, "kb_collection"):
        raise HTTPException(status_code=503, detail="Knowledge base not loaded yet")

    model_config = AGENTS[req.agent]
    api_key = api_key_for(model_config)
    if not api_key:
        logger.error("request_id=%s missing %s", request_id, model_config["api_key_env"])
        raise HTTPException(
            status_code=503,
            detail=f"Server is missing {model_config['api_key_env']}.",
        )

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
    except Exception as e:  # noqa: BLE001
        logger.exception("request_id=%s chat failed agent=%s model=%s",
                         request_id, req.agent, model_config["model"])
        # Upstream error text can carry provider internals and, in the worst
        # case, fragments of a request. It's invaluable while developing and
        # a liability in production, so it's only returned outside prod -
        # where it's suppressed, the request id ties the user's error to the
        # full server-side traceback.
        if settings.is_production:
            detail = f"Upstream model call failed. Reference: {request_id}"
        else:
            detail = (f"Upstream call to '{model_config['model']}' failed: "
                      f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=detail) from e

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("request_id=%s agent=%s tools=%s latency_ms=%.0f",
                request_id, req.agent,
                [t["name"] for t in result["tool_calls"]], latency_ms)
    return ChatResponse(
        response=result["response"],
        tool_calls=result["tool_calls"],
        latency_ms=latency_ms,
    )


@app.post("/reset")
def reset(req: ResetRequest):
    """Clears a session's short-term memory (the UI's New chat button)."""
    from agents.core import reset_session
    reset_session(req.session_id)
    return {"status": "ok"}


# --------------------------------------------------------------------------
# debug endpoints - disabled outside development unless explicitly enabled
# --------------------------------------------------------------------------

def _fetch_available_models(provider: str, key: str) -> list[str]:
    """Ask the provider what it actually serves for this key. Providers
    retire model IDs on their own schedule, so this is the authoritative
    answer rather than a guess from possibly-stale docs.
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
        return sorted(
            m["name"].replace("models/", "")
            for m in r.json().get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or ["generateContent"])
        )
    return []


@app.get("/available-models")
def available_models():
    """Live model list per provider, for filling in OSS_MODEL /
    FRONTIER_MODEL / JUDGE_MODEL when a provider retires an ID."""
    _require_debug_endpoints()
    out = {}
    for provider in {cfg["provider"] for cfg in AGENTS.values()}:
        cfg = next(c for c in AGENTS.values() if c["provider"] == provider)
        key = api_key_for(cfg)
        if not key:
            out[provider] = {"status": "missing_key", "models": []}
            continue
        try:
            out[provider] = {"status": "ok", "models": _fetch_available_models(provider, key)}
        except Exception as e:  # noqa: BLE001
            out[provider] = {"status": "error",
                             "detail": f"{type(e).__name__}: {str(e)[:300]}",
                             "models": []}
    return out


@app.get("/diagnostics")
def diagnostics():
    """One-shot self-check of every external dependency, so a failing /chat
    can be diagnosed without reading server logs: is each provider key
    present and accepted, is the KB loaded, does the real embedder work, and
    does web search work.
    """
    _require_debug_endpoints()
    import litellm

    report = {"agents": [], "kb": {}, "embedder": {}, "web_search": {},
              "sessions_in_memory": len(__import__("agents.core", fromlist=["SESSIONS"]).SESSIONS)}

    for name, cfg in AGENTS.items():
        entry = {"agent": name, "model": cfg["model"], "key_env": cfg["api_key_env"]}
        key = api_key_for(cfg)
        if not key:
            entry.update(status="missing_key",
                         detail=f"{cfg['api_key_env']} is not set. Add it to .env and restart.")
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
        except Exception as e:  # noqa: BLE001
            logger.exception("diagnostics: agent %s failed", name)
            entry.update(status="error", error_type=type(e).__name__, detail=str(e)[:600])
            # A retired model ID is the likeliest failure here and is only
            # actionable if you know what IS available, so attach the live
            # list rather than making the caller go hunting.
            if "not_found" in str(e).lower() or "no longer available" in str(e).lower():
                try:
                    entry["available_models"] = _fetch_available_models(cfg["provider"], key)
                except Exception as le:  # noqa: BLE001
                    entry["available_models_error"] = f"{type(le).__name__}: {str(le)[:200]}"
        report["agents"].append(entry)

    try:
        report["kb"] = {"status": "ok", "chunks": app.state.kb_collection.count()}
    except Exception as e:  # noqa: BLE001
        report["kb"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    try:
        vec = app.state.embed_fn(["healthy sleep habits"])
        report["embedder"] = {"status": "ok", "dimensions": len(vec[0])}
    except Exception as e:  # noqa: BLE001
        report["embedder"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    try:
        from agents.tools import search_web
        report["web_search"] = {"status": "ok", "results": len(search_web("wellness habits", max_results=2))}
    except Exception as e:  # noqa: BLE001
        report["web_search"] = {"status": "error", "detail": f"{type(e).__name__}: {str(e)[:400]}"}

    return report


@app.get("/config")
def public_config():
    """What the UI needs to render itself correctly - notably whether the
    debug panel is available, so it can hide the button instead of offering
    one that 404s."""
    return {
        "env": settings.env,
        "debug_endpoints": settings.enable_debug_endpoints,
        "max_message_chars": settings.max_message_chars,
    }


# The UI is mounted last so it never shadows an API route.
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
