# Wellness Assistant + Evals Platform (1-Day POC)

Two deployments of the same wellness-assistant architecture (an OSS model and
a frontier model, both free-tier), plus a lean evals platform that scores
both on hallucination, bias/harmful outputs, and content safety. Full
design rationale is in [`plan.md`](plan.md); the step-by-step build log is in
[`implementation_plan.md`](implementation_plan.md).

## Setup

1. **Get two free API keys (no card needed for either):**
   - Groq: https://console.groq.com → API Keys → create one → `GROQ_API_KEY`
   - Google AI Studio: https://aistudio.google.com/apikey → create one → `GOOGLE_API_KEY`
2. `cp .env.example .env` and fill in both keys.
3. `pip install -r requirements.txt`
4. Run the API (also serves the chat UI at `/`):
   ```
   uvicorn api.main:app --reload
   ```
   Open http://localhost:8000 in a browser, pick "OSS" or "Frontier" from the
   dropdown, and chat. Both hit the exact same `agents/core.py` tool-calling
   loop - only the model config differs (see `plan.md`, Section 2).
5. Prepare the eval datasets (one-time, needs internet + optionally a
   Hugging Face token for gated datasets):
   ```
   python -m evals.datasets.prepare
   ```
6. With the API still running in another terminal, run the evals:
   ```
   python -m evals.runner
   python -m evals.report
   ```
   This writes `results/raw.jsonl`, `results/scorecard.json`, and
   `results/chart.png`.
7. Judge meta-check (assessing the judge's own quality against a small
   hand-labeled set):
   ```
   python -m evals.meta_check sample
   # hand-label evals/gold_labels_template.jsonl, save as evals/gold_labels.jsonl
   ```

## Architecture decisions

See `plan.md` Section 2 for the full list with reasoning. Summary:
- No agent framework (LangGraph/LangChain) - a hand-rolled tool-calling loop
  is the entire "architecture," kept byte-identical across both models.
- `litellm` unifies the two providers' tool-call formats so the loop code
  never branches on which model is active.
- OSS assistant: **Llama-3.1-8B-Instant via Groq** (fully free, no card,
  perpetual free tier, reliable native tool-calling).
- Frontier assistant: **Gemini Flash via Google AI Studio** (fully free, no
  card).
- Eval judge: **`gpt-oss-20b` via Groq** - a third model family, distinct
  from both assistants, to avoid the judge favoring a response just because
  it shares a family with it.
- Knowledge base: the 9 provided wellness `.md` files, paragraph-chunked,
  embedded with `sentence-transformers` (`all-MiniLM-L6-v2`), held in an
  in-memory Chroma collection (rebuilt on every process start - no
  persistence needed for a same-day demo).

## Trade-offs made

- **~15 test items per axis**, not hundreds - a directional signal, not a
  statistically rigorous benchmark. Good enough to compare two assistants at
  POC depth, not to publish a leaderboard.
- **Bias/safety scoring reuses DeepEval's `BiasMetric`/`ToxicityMetric`**
  rather than purpose-built stereotype or unsafe-response classifiers - an
  accepted gap given the time budget (see `plan.md` Decision #7).
- **No persistence** - session memory and the KB vector store are in-process
  and reset when the server restarts.
- **Single labeler, no formal Cohen's kappa** for the judge meta-check - a
  simple % agreement instead.
- **No multi-turn adversarial jailbreaks** - the safety test set is
  single-shot prompts only.

## What we'd improve with more time

- A larger, independently-labeled gold set (2+ labelers) with real
  inter-rater and judge-vs-human kappa, not just % agreement.
- An ensemble judge (2-3 model families voting) instead of a single judge
  model, to further reduce idiosyncratic judge bias.
- Purpose-built classifiers for stereotype/toxicity detection as a second
  signal alongside the LLM judge.
- Multi-turn adversarial safety tests, not just single-shot jailbreak prompts.
- Guardrails added directly in response to the eval results (regex/refusal
  filter on whichever axis scores worst), with a before/after re-run to
  prove they work.
- A public deployment of the OSS assistant (e.g. a Hugging Face Space) and a
  recorded demo.

## Repo layout

```
agents/       shared tool-calling core, tools, prompt, model configs
kb/           knowledge-base source files + ingestion
api/          FastAPI backend
ui/           minimal HTML/JS chat page
evals/        datasets, judge, runner, refusal checker, meta-check, report
results/      eval run outputs (raw.jsonl, scorecard.json, chart.png)
tests/        offline-testable unit/integration tests (see below)
```

## Testing note

This POC was built and unit-tested in a sandboxed environment with no
general internet access (only package registries reachable) - so the tests
under `tests/` verify all the control-flow logic (KB chunking, the
tool-calling loop, retry-on-malformed-JSON, the FastAPI routes, eval
aggregation, refusal classification) using fakes/mocks in place of the real
Groq/Gemini/Hugging Face/DuckDuckGo network calls. See `tests/README.md` for
exactly what's covered by mocks vs. what still needs a real run with live
API keys and internet access.
