"""THE fixed architecture: one tool-calling loop, imported unchanged by both
the OSS and frontier assistant. Only the `model_config` passed in differs
(see agents/config.py) - everything else (system prompt, tool schemas,
memory handling, retry logic) is identical, which is the whole point of the
"keep the architecture fixed" requirement.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from agents.prompts import system_prompt
from agents.tools import TOOL_SCHEMAS, lookup_kb, search_web
from settings import settings

logger = logging.getLogger("wellness.core")

MAX_TOOL_ITERATIONS = settings.max_tool_iterations
MEMORY_WINDOW = settings.memory_window  # messages kept per session

# Free tiers get capacity-rejected under load - Gemini returns 503 "high
# demand ... try again later" and Groq 429s on rate limits. Both are
# retryable and both are common enough that not retrying makes the app look
# broken when it isn't. Permanent errors (bad key, retired model ID) are
# re-raised immediately so they stay visible instead of being masked by
# three slow retries.
MAX_COMPLETION_RETRIES = settings.completion_retries
TRANSIENT_MARKERS = (
    "503", "unavailable", "overloaded", "high demand",
    "429", "rate limit", "rate_limit", "quota",
    "timeout", "timed out", "temporarily",
)


RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")

# Providers usually say exactly how long to wait. Gemini returns
# "Please retry in 33.1s" and a retryDelay field; guessing instead of
# reading it is what turned a 20-requests-per-minute limit into a run that
# lost most of its items.
_RETRY_AFTER_PATTERNS = (
    r"please retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
    r"retrydelay[\"\':\s]+([0-9]+(?:\.[0-9]+)?)s",
    r"retry[-_ ]after[\"\':\s]+([0-9]+(?:\.[0-9]+)?)",
)

# Groq words it differently again - "Please try again in 1m23.4s", or
# "in 2h14m30s" when a daily cap is hit. Matching only Gemini's phrasing
# meant Groq limits fell back to a blind 20s/40s/80s backoff that could
# never clear a multi-hour cap, and the error said only "rate limit"
# instead of "come back in two hours".
_GROQ_RETRY_PATTERN = re.compile(
    r"try again in\s+(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+(?:\.[0-9]+)?)s)?", re.I)


def _groq_retry_seconds(msg: str) -> float | None:
    m = _GROQ_RETRY_PATTERN.search(msg)
    if not m or not any(m.groups()):
        return None
    hours, minutes, seconds = m.groups()
    return (float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)) or None


def retry_after_seconds(err: Exception) -> float | None:
    """The provider's own instruction, when it gives one."""
    msg = str(err)
    for pattern in _RETRY_AFTER_PATTERNS:
        m = re.search(pattern, msg, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return _groq_retry_seconds(msg)

# A rate limit is a transient error with a completely different time
# constant. Backing off 1s then 2s against a per-minute quota just burns
# the retries and reports failure - which is exactly what happened on the
# first full eval run: 51 of 68 items died this way.
RATE_LIMIT_BASE_DELAY_S = 20.0
TRANSIENT_BASE_DELAY_S = 1.0

# Total seconds a single request may spend asleep across all its retries.
#
# Without this, honouring the provider's own retry hint is unbounded: a
# free tier that answers "retry in 90s" five times turns one request into a
# 7-minute hang. That is worse than failing - the caller (a user, or an
# eval item) has no signal, no error, and no way to tell a throttled
# provider from a wedged server. Past the budget the request fails with the
# provider's message intact, so the cause is still visible.
#
# The default (75s) is a deliberate compromise: it covers two rounds of the
# ~30s "retry in" hint a per-minute free tier gives, which is usually
# enough to get the item, without letting one request hang for minutes.
MAX_RETRY_SLEEP_TOTAL_S = settings.max_retry_sleep_total_s


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(marker in msg for marker in RATE_LIMIT_MARKERS)


def _completion_with_retry(completion_fn, **kwargs):
    delay = None
    slept = 0.0
    for attempt in range(MAX_COMPLETION_RETRIES):
        try:
            return completion_fn(**kwargs)
        except Exception as e:
            last_attempt = attempt == MAX_COMPLETION_RETRIES - 1
            if last_attempt or not _is_transient(e):
                raise

            # Prefer what the provider actually told us over our own guess.
            hinted = retry_after_seconds(e)
            if hinted is not None:
                wait = hinted + 1.0          # small buffer for clock skew
            else:
                if delay is None:
                    delay = (RATE_LIMIT_BASE_DELAY_S if _is_rate_limit(e)
                             else TRANSIENT_BASE_DELAY_S)
                wait = delay
                delay *= 2

            if slept + wait > MAX_RETRY_SLEEP_TOTAL_S:
                # Out of patience, not out of attempts. Raise the provider's
                # own error so the caller sees "rate limit, retry in Ns"
                # rather than a request that never came back.
                logger.warning(
                    "giving up after %.0fs of backoff; next wait %.0fs exceeds the "
                    "%.0fs budget: %s", slept, wait, MAX_RETRY_SLEEP_TOTAL_S, str(e)[:600])
                raise

            logger.warning(
                "transient upstream error (attempt %d/%d), retrying in %.1fs%s: %s",
                attempt + 1, MAX_COMPLETION_RETRIES, wait,
                " (provider-specified)" if hinted is not None else "",
                str(e)[:600],
            )
            time.sleep(wait)
            slept += wait

class SessionStore:
    """Bounded in-process short-term memory.

    A plain dict grows without limit: every unique session_id a caller
    invents costs memory forever, which on a public deployment is a trivial
    way to exhaust the process. This caps it on both axes - entries expire
    after a TTL, and the oldest are evicted once the store is full.

    In-process is a deliberate choice for this scope, and its consequence is
    explicit: memory is lost on restart and is not shared between replicas,
    so running more than one instance needs Redis behind this same
    interface. See docs/DESIGN.md.
    """

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()

    def _expired(self, stamped_at: float, now: float) -> bool:
        return now - stamped_at > self.ttl_seconds

    def get(self, session_id: str) -> list[dict]:
        now = time.time()
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return []
            stamped_at, messages = entry
            if self._expired(stamped_at, now):
                del self._data[session_id]
                return []
            self._data.move_to_end(session_id)  # mark as recently used
            return list(messages)

    def set(self, session_id: str, messages: list[dict]) -> None:
        now = time.time()
        with self._lock:
            self._data[session_id] = (now, list(messages))
            self._data.move_to_end(session_id)
            # drop anything stale, then trim to the cap (oldest first)
            for key in [k for k, (ts, _) in self._data.items() if self._expired(ts, now)]:
                del self._data[key]
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


SESSIONS = SessionStore(
    max_sessions=settings.max_sessions,
    ttl_seconds=settings.session_ttl_seconds,
)


def _safe_json_loads(raw: str) -> dict:
    """Tool-call arguments occasionally come back malformed on smaller models
    (docs/DESIGN.md records this for small open models) - retry once
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
    return SESSIONS.get(session_id)


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id)


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
    # tool name -> the error it raised, for the circuit breaker below
    failed_tools: dict = {}

    for iteration in range(MAX_TOOL_ITERATIONS + 1):
        if iteration == MAX_TOOL_ITERATIONS:
            # Last pass. A model that keeps choosing tools - which the
            # mandatory-grounding prompt actively encourages - would
            # otherwise burn every iteration and leave the user with the
            # "couldn't finish" fallback despite having gathered perfectly
            # good tool results.
            #
            # Nudge rather than withhold. Dropping `tools` (or setting
            # tool_choice="none") makes Groq reject the whole request with
            # tool_use_failed the moment the model tries a call anyway -
            # observed live. A plain instruction is provider-agnostic and
            # degrades gracefully if ignored.
            messages.append({
                "role": "user",
                "content": (
                    "Answer now using the information you have already gathered. "
                    "Do not call any more tools."
                ),
            })

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
                if name in failed_tools:
                    # Circuit breaker. A tool that has already failed this
                    # turn will almost certainly fail again - the causes are
                    # environmental (no network, a broken TLS stack, a
                    # blocked endpoint), not query-dependent. Without this,
                    # a model that keeps reaching for a dead tool burns
                    # every remaining iteration on identical timeouts: one
                    # broken web-search backend turned ~10s evaluation items
                    # into ~3-minute ones and silently poisoned the latency
                    # numbers.
                    result = {"error": failed_tools[name],
                              "note": "this tool already failed on this turn; "
                                      "answer from what you have instead of retrying it"}
                elif name == "lookup_kb":
                    result = lookup_kb(coll, embed_fn, **args)
                elif name == "search_web":
                    result = search_web(**args)
                else:
                    result = {"error": f"unknown tool {name}"}
            except Exception as e:  # noqa: BLE001 - deliberately broad, logged into tool_calls_log
                args = None
                result = {"error": str(e)}
                failed_tools[name] = str(e)[:200]
                logger.warning("tool %s failed: %s", name, e)

            tool_calls_log.append({"name": name, "args": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
    else:
        # exhausted MAX_TOOL_ITERATIONS+1 loops without a final text answer
        final_text = "I wasn't able to finish looking that up - could you rephrase or ask again?"

    SESSIONS.set(
        session_id,
        (
            history
            + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_text},
            ]
        )[-MEMORY_WINDOW:],
    )

    return {"response": final_text, "tool_calls": tool_calls_log}
