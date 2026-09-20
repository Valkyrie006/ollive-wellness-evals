"""Model configs for the two assistant deployments + the eval judge.
The ONLY thing that differs between the OSS and frontier assistant is this
config object - agents/core.py is otherwise identical for both, which is
what satisfies the "keep the architecture fixed" requirement.

All three are free-tier, no-card services (see docs/DESIGN.md):
- OSS assistant:      gemma-4-26b-a4b-it via Google AI Studio (open weights)
- Frontier assistant: gemini-3.1-flash-lite via Google AI Studio
- Eval judge:         qwen3.8-27b via Groq - deliberately a third model
  family, so the judge never scores a model from its own family and risk
  self-preference bias.

Why the OSS model is Gemma and not GPT-OSS, in order of what actually
happened:

1. Llama 3.1 8B on Groq. Groq moved Meta's Llama models to Enterprise
   ("contact sales") access, so a standard key gets model_not_found.
2. openai/gpt-oss-20b on Groq. Worked, and produced the first complete set
   of results. Then Groq's free tier hit its per-DAY token cap: "Rate limit
   reached ... on tokens per day (TPD): Limit 200000, Used 199367". A cap
   measured in days cannot be waited out inside a run, and a full
   evaluation costs far more than what trickles back.
3. gemma-4-26b-a4b-it on Google AI Studio. Open weights, free, no card,
   supports the tool calling this architecture requires, and answers in
   ~20s. Its larger sibling gemma-4-31b-it also works but takes ~76s per
   turn, which is unusable across a 44-item run.

The honest consequence: both assistants now sit on Google AI Studio. That
removes provider infrastructure as a confound - same serving stack, same
API, so a latency difference is the model rather than the vendor - but it
does mean the open-vs-frontier comparison happens inside one vendor's
lineup. Set OSS_MODEL=groq/openai/gpt-oss-20b to run it across vendors
instead, on a key with daily tokens to spare.

The judge stays on Groq. It is the only Groq consumer now, and it spends a
few hundred tokens per call, so it fits in what the free tier refills.
Qwen sits in Groq's Preview tier and can be withdrawn at short notice - if
it disappears, openai/gpt-oss-120b is the fallback judge, but then the
judge shares a family with nothing here and that is fine, while
gemini-anything would share a family with BOTH agents and must not be used.

Model IDs are read from the environment with defaults, because providers
retire model IDs on their own schedule and a retired ID is indistinguishable
from a broken app from the outside. When that happens, hit GET
/available-models (or the Diagnostics button in the UI) to see what the
provider currently serves, then set the ID in .env and restart - no code
change needed:

    OSS_MODEL=gemini/gemma-4-26b-a4b-it
    FRONTIER_MODEL=gemini/gemini-3.1-flash-lite
    JUDGE_MODEL=groq/qwen/qwen3.8-27b

The provider and its API key are derived from the model ID prefix, so an
override like the above moves the key with it automatically.
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

# The model ID already names its provider ("groq/...", "gemini/..."), so the
# provider and its key are derived from it rather than pinned per agent.
#
# They used to be hardcoded, and that made the override in .env a trap:
# pointing OSS_MODEL at a Gemini-hosted model left the config still claiming
# provider "groq", so the app sent the Groq key to Google and failed with an
# auth error that pointed at the wrong thing entirely. One source of truth
# removes the whole class of mismatch.
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}
DEFAULT_PROVIDER = "groq"


def provider_of(model: str) -> str:
    prefix = model.split("/", 1)[0].lower() if "/" in model else ""
    return prefix if prefix in PROVIDER_KEY_ENV else DEFAULT_PROVIDER


def _agent_config(name: str, model: str) -> dict:
    provider = provider_of(model)
    return {
        "name": name,
        "model": model,
        "provider": provider,
        "api_key_env": PROVIDER_KEY_ENV[provider],
    }


OSS_CONFIG = _agent_config("oss", settings.oss_model)
FRONTIER_CONFIG = _agent_config("frontier", settings.frontier_model)
JUDGE_CONFIG = _agent_config("judge", settings.judge_model)

AGENTS = {"oss": OSS_CONFIG, "frontier": FRONTIER_CONFIG}


def api_key_for(config: dict) -> str | None:
    return os.environ.get(config["api_key_env"])
