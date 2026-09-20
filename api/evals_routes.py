"""Evaluation control endpoints.

A full run takes many minutes and hundreds of upstream calls, so these are
fire-and-poll rather than request/response: POST starts a background
thread, GET reports progress. That also means a dropped connection can't
lose a run that has already cost real quota.

These are debug-gated with the rest. An unauthenticated endpoint that
spends hundreds of provider calls on demand has no business being reachable
on a public deployment.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("wellness.evals.api")

router = APIRouter(prefix="/evals", tags=["evals"])

# One run at a time, guarded - concurrent runs would interleave writes to
# results/ and race each other through the same rate limits.
_state: dict = {"status": "idle", "phase": None, "done": 0, "total": 0,
                "started_at": None, "finished_at": None,
                "result": None, "error": None}
_lock = threading.Lock()


class RunRequest(BaseModel):
    base_url: str = Field(default="http://127.0.0.1:8000", max_length=200)
    agents: list = Field(default_factory=lambda: ["oss", "frontier"])
    include_judge_meta: bool = True
    prepare_datasets: bool = True
    # Optional[...] not `X | None` - pydantic evaluates these annotations
    # eagerly, so PEP 604 unions break on Python 3.9 regardless of the
    # __future__ import.
    axes: Optional[list] = None
    # None = leave the server's configured setting alone; True/False forces
    # it for the duration of this run so the same server can be A/B tested.
    guardrails: Optional[bool] = None
    out_name: str = "scorecard.json"
    raw_name: str = "raw.jsonl"
    label: Optional[str] = None


def _set(**kw):
    with _lock:
        _state.update(kw)


def _progress(done: int, total: int, label: str):
    _set(done=done, total=total, phase=label)


def _run(req: RunRequest):
    from agents import guardrails
    from evals import meta_check
    from evals.datasets import prepare
    from evals.runner import run_all

    try:
        summary = {}
        if req.guardrails is not None:
            guardrails.set_enabled(req.guardrails)
            summary["guardrails"] = req.guardrails

        if req.prepare_datasets:
            _set(phase="preparing datasets", done=0, total=0)
            summary["datasets"] = prepare.main()
            logger.info("datasets ready: %s", summary["datasets"])

        _set(phase="scoring agents")
        summary["scorecard"] = run_all(req.base_url, tuple(req.agents), progress=_progress,
                                       axes=req.axes, out_name=req.out_name,
                                       raw_name=req.raw_name, label=req.label,
                                       dataset_sources=summary.get("datasets"))

        if req.include_judge_meta:
            _set(phase="judging the judge")
            summary["judge_quality"] = meta_check.evaluate_judge(progress=_progress)
            summary["safety_cross_check"] = meta_check.safety_cross_check()

        _set(status="finished", phase="done", result=summary,
             finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        logger.info("eval run complete")
    except Exception as e:  # noqa: BLE001 - surfaced through GET /evals/status
        logger.exception("eval run failed")
        _set(status="failed", error=f"{type(e).__name__}: {e}",
             finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finally:
        # never leave a forced override behind - the next request would
        # silently inherit an eval run's setting
        guardrails.set_enabled(None)


def start_run(req: RunRequest) -> dict:
    with _lock:
        if _state["status"] == "running":
            raise HTTPException(status_code=409, detail="an eval run is already in progress")
        _state.update(status="running", phase="starting", done=0, total=0,
                      result=None, error=None, finished_at=None,
                      started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    threading.Thread(target=_run, args=(req,), daemon=True).start()
    return {"status": "started"}


def get_status() -> dict:
    with _lock:
        out = dict(_state)
    if out["total"]:
        out["percent"] = round(100 * out["done"] / out["total"], 1)
    return out
