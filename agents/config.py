"""Model configs for the two assistant deployments + the eval judge.
The ONLY thing that differs between the OSS and frontier assistant is this
config object - agents/core.py is otherwise identical for both, which is
what satisfies the "keep the architecture fixed" requirement.

All three are free-tier, no-card services (see plan.md, Decisions #3-#5):
- OSS assistant:  Llama-3.1-8B-Instant via Groq
- Frontier assistant: Gemini Flash via Google AI Studio
- Eval judge: gpt-oss-20b via Groq (different family from both assistants,
  so it doesn't share a family with either one and risk self-preference bias)
"""
from __future__ import annotations

import os

OSS_CONFIG = {
    "name": "oss",
    "model": "groq/llama-3.1-8b-instant",
    "api_key_env": "GROQ_API_KEY",
}

FRONTIER_CONFIG = {
    "name": "frontier",
    "model": "gemini/gemini-2.0-flash",
    "api_key_env": "GOOGLE_API_KEY",
}

JUDGE_CONFIG = {
    "name": "judge",
    "model": "groq/gpt-oss-20b",
    "api_key_env": "GROQ_API_KEY",
}

AGENTS = {"oss": OSS_CONFIG, "frontier": FRONTIER_CONFIG}


def api_key_for(config: dict) -> str | None:
    return os.environ.get(config["api_key_env"])
