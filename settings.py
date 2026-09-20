"""Single source of truth for runtime configuration.

Everything tunable is read from the environment exactly once, here, rather
than scattered `os.getenv` calls across modules. That matters for two
reasons: a misconfigured deployment fails loudly at import rather than
mysteriously at the first request, and there is one place to read to find
out what a running instance will actually do.

Deliberately implemented with the standard library instead of
pydantic-settings. The dependency would be reasonable, but this project has
already lost time to dependency-resolution problems on a constrained
machine, and a frozen dataclass over `os.getenv` costs about forty lines
and adds nothing to install.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

TRUTHY = {"1", "true", "yes", "on"}


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in TRUTHY


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    # --- environment -------------------------------------------------
    env: str = "development"

    # --- models ------------------------------------------------------
    # Providers retire model IDs on their own schedule, so these are
    # overridable without a code change. See docs/DESIGN.md.
    oss_model: str = "groq/openai/gpt-oss-20b"
    frontier_model: str = "gemini/gemini-3.6-flash"
    judge_model: str = "groq/qwen/qwen3.8-27b"

    # --- request limits ----------------------------------------------
    # A chat endpoint backed by a paid upstream is a money faucet if left
    # unbounded, so both the size and the rate of requests are capped.
    max_message_chars: int = 4000
    max_session_id_chars: int = 64
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # --- session memory ----------------------------------------------
    # In-process short-term memory is bounded on both axes: a TTL so
    # abandoned conversations expire, and a hard cap so a burst of unique
    # session ids can't exhaust memory.
    session_ttl_seconds: int = 3600
    max_sessions: int = 1000

    # --- behaviour ---------------------------------------------------
    max_tool_iterations: int = 2
    memory_window: int = 6
    completion_retries: int = 3

    # --- exposure ----------------------------------------------------
    # /diagnostics and /available-models call upstream providers with the
    # server's own key and echo provider errors. Useful locally, not
    # something to leave reachable on a public deployment.
    enable_debug_endpoints: bool = True
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    # --- guardrails ---------------------------------------------------
    # Toggleable so the eval can measure the same agents with and without
    # them. A guardrail whose effect you cannot measure is a guess.
    enable_guardrails: bool = True

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"production", "prod"}

    @classmethod
    def from_env(cls) -> Settings:
        env = os.getenv("APP_ENV", "development")
        is_prod = env.lower() in {"production", "prod"}
        return cls(
            env=env,
            oss_model=os.getenv("OSS_MODEL", "groq/openai/gpt-oss-20b"),
            frontier_model=os.getenv("FRONTIER_MODEL", "gemini/gemini-3.6-flash"),
            judge_model=os.getenv("JUDGE_MODEL", "groq/qwen/qwen3.8-27b"),
            max_message_chars=_int("MAX_MESSAGE_CHARS", 4000),
            max_session_id_chars=_int("MAX_SESSION_ID_CHARS", 64),
            rate_limit_requests=_int("RATE_LIMIT_REQUESTS", 20),
            rate_limit_window_seconds=_int("RATE_LIMIT_WINDOW_SECONDS", 60),
            session_ttl_seconds=_int("SESSION_TTL_SECONDS", 3600),
            max_sessions=_int("MAX_SESSIONS", 1000),
            max_tool_iterations=_int("MAX_TOOL_ITERATIONS", 2),
            memory_window=_int("MEMORY_WINDOW", 6),
            completion_retries=_int("COMPLETION_RETRIES", 3),
            # Off by default in production; opt back in explicitly if you
            # want them on a deployed instance.
            enable_debug_endpoints=_bool("ENABLE_DEBUG_ENDPOINTS", not is_prod),
            cors_origins=_csv("CORS_ORIGINS", ()),
            enable_guardrails=_bool("ENABLE_GUARDRAILS", True),
        )


settings = Settings.from_env()
