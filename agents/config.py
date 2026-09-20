"""Model configs for the two assistant deployments + the eval judge.
The ONLY thing that differs between the OSS and frontier assistant is this
config object - agents/core.py is otherwise identical for both, which is
what satisfies the "keep the architecture fixed" requirement.

All three are free-tier, no-card services (see plan.md, Decisions #3-#5):
- OSS assistant:  openai/gpt-oss-20b via Groq (open weights)
- Frontier assistant: Gemini Flash via Google AI Studio (gemini-flash-latest;
  the pinned gemini-3.6-flash measured 62s p50 and a 48% error rate under
  load, so the alias is the more reliable default)
- Eval judge: qwen3.8-27b via Groq - deliberately a third model family, so
  the judge never scores a model from its own family and risks
  self-preference bias

Note on the OSS choice: this started as Llama 3.1 8B, but Groq moved Meta's
Llama models to Enterprise ("contact sales") access, so a standard key now
gets model_not_found for them. openai/gpt-oss-20b is the equivalent
open-weights production model reachable on a normal key. That in turn
pushed the judge off gpt-oss-20b onto Qwen to keep the three families
distinct. Qwen sits in Groq's Preview tier, which can be withdrawn at short
notice - if it disappears, openai/gpt-oss-120b is the fallback judge, but
then the judge shares a family with the OSS assistant and that caveat
belongs in the eval report.

Model IDs are read from the environment with defaults, because providers
retire model IDs on their own schedule and a retired ID is indistinguishable
from a broken app from the outside. When that happens, hit GET
/available-models (or the Diagnostics button in the UI) to see what the
provider currently serves, then set the ID in .env and restart - no code
change needed:

    OSS_MODEL=groq/openai/gpt-oss-20b
    FRONTIER_MODEL=gemini/gemini-3.6-flash
    JUDGE_MODEL=groq/qwen/qwen3.8-27b
"""
from __future__ import annotations

import os

from settings import settings

# Live model-list endpoints, used by /available-models to report exactly what
# each provider currently serves for the caller's key.
MODEL_LIST_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

OSS_CONFIG = {
    "name": "oss",
    "model": settings.oss_model,
    "api_key_env": "GROQ_API_KEY",
    "provider": "groq",
}

FRONTIER_CONFIG = {
    "name": "frontier",
    "model": settings.frontier_model,
    "api_key_env": "GOOGLE_API_KEY",
    "provider": "gemini",
}

JUDGE_CONFIG = {
    "name": "judge",
    "model": settings.judge_model,
    "api_key_env": "GROQ_API_KEY",
    "provider": "groq",
}

AGENTS = {"oss": OSS_CONFIG, "frontier": FRONTIER_CONFIG}


def api_key_for(config: dict) -> str | None:
    return os.environ.get(config["api_key_env"])
