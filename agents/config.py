"""Model configs for the two assistant deployments + the eval judge.
The ONLY thing that differs between the OSS and frontier assistant is this
config object - agents/core.py is otherwise identical for both, which is
what satisfies the "keep the architecture fixed" requirement.

All three are free-tier, no-card services (see plan.md, Decisions #3-#5):
- OSS assistant:  an open-weights Llama model via Groq
- Frontier assistant: Gemini Flash via Google AI Studio
- Eval judge: gpt-oss-20b via Groq (a different model family from both
  assistants, so the judge never scores its own family and risk
  self-preference bias)

Model IDs are read from the environment with defaults, because providers
retire model IDs on their own schedule and a retired ID is indistinguishable
from a broken app from the outside. When that happens, hit GET
/available-models (or the Diagnostics button in the UI) to see what the
provider currently serves, then set the ID in .env and restart - no code
change needed:

    OSS_MODEL=groq/llama-3.3-70b-versatile
    FRONTIER_MODEL=gemini/gemini-3.6-flash
    JUDGE_MODEL=groq/openai/gpt-oss-20b
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Loaded here rather than only in api/main.py: the model IDs below are read at
# import time, and this module is imported by the eval runner and scripts too,
# none of which should have to remember to load .env first.
load_dotenv()

# Live model-list endpoints, used by /available-models to report exactly what
# each provider currently serves for the caller's key.
MODEL_LIST_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

OSS_CONFIG = {
    "name": "oss",
    "model": os.getenv("OSS_MODEL", "groq/llama-3.3-70b-versatile"),
    "api_key_env": "GROQ_API_KEY",
    "provider": "groq",
}

FRONTIER_CONFIG = {
    "name": "frontier",
    "model": os.getenv("FRONTIER_MODEL", "gemini/gemini-3.6-flash"),
    "api_key_env": "GOOGLE_API_KEY",
    "provider": "gemini",
}

JUDGE_CONFIG = {
    "name": "judge",
    "model": os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-20b"),
    "api_key_env": "GROQ_API_KEY",
    "provider": "groq",
}

AGENTS = {"oss": OSS_CONFIG, "frontier": FRONTIER_CONFIG}


def api_key_for(config: dict) -> str | None:
    return os.environ.get(config["api_key_env"])
