"""THE fixed architecture: one tool-calling loop, imported unchanged by both
the OSS and frontier assistant. Only the `model_config` passed in differs
(see agents/config.py) - everything else (system prompt, tool schemas,
memory handling, retry logic) is identical, which is the whole point of the
"keep the architecture fixed" requirement.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Callable

from agents.tools import lookup_kb, search_web, TOOL_SCHEMAS
from agents.prompts import system_prompt

logger = logging.getLogger("wellness.core")

MAX_TOOL_ITERATIONS = 2
MEMORY_WINDOW = 6  # last N messages kept per session (short-term memory)

# Free tiers get capacity-rejected under load - Gemini returns 503 "high
# demand ... try again later" and Groq 429s on rate limits. Both are
# retryable and both are common enough that not retrying makes the app look
# broken when it isn't. Permanent errors (bad key, retired model ID) are
# re-raised immediately so they stay visible instead of being masked by
# three slow retries.
MAX_COMPLETION_RETRIES = 3
TRANSIENT_MARKERS = (
    "503", "unavailable", "overloaded", "high demand",
    "429", "rate limit", "rate_limit", "quota",
    "timeout", "timed out", "temporarily",
)


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def _completion_with_retry(completion_fn, **kwargs):
    delay = 1.0
    for attempt in range(MAX_COMPLETION_RETRIES):
        try:
            return completion_fn(**kwargs)
        except Exception as e:
            last_attempt = attempt == MAX_COMPLETION_RETRIES - 1
            if last_attempt or not _is_transient(e):
                raise
            logger.warning(
                "transient upstream error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_COMPLETION_RETRIES, delay, str(e)[:160],
            )
            time.sleep(delay)
            delay *= 2

# In-process session memory: session_id -> list[{"role", "content"}]
# Intentionally not persisted (plan.md Decision #6) - fine for a same-day POC.
SESSIONS: dict[str, list[dict]] = {}


def _safe_json_loads(raw: str) -> dict:
    """Tool-call arguments occasionally come back malformed on smaller models
    (plan.md Decision #3 flags this for Llama/Qwen-class models) - retry once
    with a couple of common repairs before giving up.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = raw.strip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        if not repaired.endswith("}") and repaired.startswith("{"):
            repaired = repaired + "}"
        return json.loads(repaired)  # let this raise if still broken; caller catches it


def get_history(session_id: str) -> list[dict]:
    return list(SESSIONS.get(session_id, []))


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


def run_turn(
    session_id: str,
    user_message: str,
    model_config: dict,
    api_key: str | None,
    coll,
    embed_fn: Callable[[list[str]], list[list[float]]],
    completion_fn: Callable | None = None,
) -> dict:
    """Runs one user turn through the shared tool-calling loop.

    `completion_fn` defaults to `litellm.completion` but is injectable so the
    control flow can be unit-tested without hitting a real model API (see
    tests/test_agent_core.py).
    """
    if completion_fn is None:
        import litellm
        completion_fn = litellm.completion

    history = get_history(session_id)
    messages = (
        [{"role": "system", "content": system_prompt()}]
        + history
        + [{"role": "user", "content": user_message}]
    )
    tool_calls_log = []
    final_text = None

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        resp = _completion_with_retry(
            completion_fn,
            model=model_config["model"],
            messages=messages,
            tools=TOOL_SCHEMAS,
            api_key=api_key,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_text = msg.content
            messages.append({"role": "assistant", "content": final_text})
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = _safe_json_loads(tc.function.arguments)
                if name == "lookup_kb":
                    result = lookup_kb(coll, embed_fn, **args)
                elif name == "search_web":
                    result = search_web(**args)
                else:
                    result = {"error": f"unknown tool {name}"}
            except Exception as e:  # noqa: BLE001 - deliberately broad, logged into tool_calls_log
                args = None
                result = {"error": str(e)}

            tool_calls_log.append({"name": name, "args": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
    else:
        # exhausted MAX_TOOL_ITERATIONS+1 loops without a final text answer
        final_text = "I wasn't able to finish looking that up - could you rephrase or ask again?"

    SESSIONS[session_id] = (
        history
        + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_text},
        ]
    )[-MEMORY_WINDOW:]

    return {"response": final_text, "tool_calls": tool_calls_log}
