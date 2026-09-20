"""Lightweight FastAPI backend (plan.md Section 3). One /chat route serves
both agents by `agent` field ("oss" or "frontier") - both call the exact
same agents.core.run_turn with only the model config swapped.
"""
from __future__ import annotations
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.config import AGENTS, api_key_for
from agents.core import run_turn

load_dotenv()


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
    result = run_turn(
        session_id=req.session_id,
        user_message=req.message,
        model_config=model_config,
        api_key=api_key,
        coll=app.state.kb_collection,
        embed_fn=app.state.embed_fn,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(response=result["response"], tool_calls=result["tool_calls"], latency_ms=latency_ms)


# Serve the minimal chat UI at /
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
