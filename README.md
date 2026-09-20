# Wellness Assistant + Evals Platform

Two AI wellness assistants — one open-weights, one frontier — running on a
**byte-identical architecture**, plus a harness that scores both on
hallucination, bias, and content safety, and then checks whether the judge
doing the scoring can be trusted.

The point of the project is the comparison. If the two agents differ in any
way other than the model ID, the evaluation measures the scaffolding rather
than the models — so the prompt, the tool schemas, the tool-calling loop,
the memory handling, and the retry logic are shared code, and
`agents/config.py` is the only file that knows they're different.

---

## What's in here

| Piece | Where | What it does |
|---|---|---|
| Fixed agent architecture | `agents/core.py` | One tool-calling loop, imported unchanged by both agents |
| Agent configs | `agents/config.py` | The *only* difference between the two agents |
| Tools | `agents/tools.py` | `lookup_kb` (Chroma over the wellness KB) and `search_web` (DuckDuckGo) |
| Knowledge base | `kb/` | 9 wellness guides, paragraph-chunked and embedded |
| HTTP API + UI | `api/`, `ui/` | FastAPI backend, single-file chat interface |
| Evals harness | `evals/` | Scores both agents on three axes, then meta-evaluates the judge |
| Guardrails | `agents/guardrails.py` | Input/output safety layer, derived from the eval results |

## Quick start

```bash
git clone https://github.com/Valkyrie006/ollive-wellness-evals.git
cd ollive-wellness-evals

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then add your two free API keys
uvicorn api.main:app --reload --port 8000
```

Open <http://localhost:8000>. First start takes an extra minute while the
embedding model downloads.

Both providers are **free tier, no card required**:
[Groq](https://console.groq.com/keys) and
[Google AI Studio](https://aistudio.google.com/apikey).

### Docker

```bash
export GROQ_API_KEY=... GOOGLE_API_KEY=...
docker compose up --build
```

The image bakes in the embedding model, so containers start without
reaching Hugging Face.

## Running the evaluation

With the server running, either drive it from the CLI:

```bash
python evals/datasets/prepare.py      # pulls MedHallu / BBQ / JailbreakBench
python -m evals.runner --base-url http://localhost:8000 --agents oss frontier
python -m evals.meta_check            # scores the judge against known labels
python evals/report.py                # charts
python evals/report_doc.py            # one-page HTML + PDF report
```

…or trigger the whole pipeline over HTTP and poll it (this is how the
committed results were produced):

```bash
curl -X POST localhost:8000/evals/run -H 'Content-Type: application/json' \
  -d '{"agents":["oss","frontier"],"guardrails":false,"label":"baseline"}'
curl localhost:8000/evals/status
```

A full run takes ~20 minutes and several hundred upstream calls — free
tiers rate-limit hard, so the runner paces itself per agent.

**Outputs** land in `results/` and are committed, so the numbers in the
report can be checked against the per-item evidence:

| File | What's in it |
|---|---|
| `evaluation_report.pdf` | The one-page report |
| `scorecard.json` | Aggregated rates, latency percentiles, provenance |
| `raw.jsonl` | Every item: prompt, response, verdict, the judge's reason, latency |
| `judge_quality.json` | Judge scored against dataset ground truth, per case |
| `comparison.png`, `judge_quality.png` | The infographics |

### How the judge is judged

The spec asks for an assessment of the judge itself. Rather than
hand-labelling — which, at this scale, would mean an LLM labelling for an
LLM judge — the judge is scored against ground truth the dataset already
carries. MedHallu ships a known-correct *and* a known-hallucinated answer
per question, so the judge's verdicts can be scored for accuracy,
precision, recall, and **Cohen's κ**, which corrects for the agreement a
coin flip would achieve. The deterministic refusal classifier and the LLM
judge also label the same safety responses independently, and their
disagreement rate is reported.

### Guardrails

`agents/guardrails.py` adds a deterministic input filter (prompt-injection
*shape*, not topic) and an output layer that redacts specific dosage
instructions and appends a medical disclaimer. They are toggleable at
runtime so the **same running server** can be scored with and without —
otherwise "the guardrails help" is an assertion rather than a measurement:

```bash
curl -X POST localhost:8000/evals/run -H 'Content-Type: application/json' \
  -d '{"axes":["safety"],"guardrails":true,"out_name":"scorecard_guardrails.json"}'
```

## Configuration

Everything is environment-driven and read once in `settings.py`.

| Variable | Default | Why you'd change it |
|---|---|---|
| `GROQ_API_KEY`, `GOOGLE_API_KEY` | — | Required |
| `APP_ENV` | `development` | `production` hides upstream error detail and disables debug endpoints |
| `OSS_MODEL` | `groq/openai/gpt-oss-20b` | Providers retire model IDs; swap without a code change |
| `FRONTIER_MODEL` | `gemini/gemini-3.6-flash` | Same |
| `JUDGE_MODEL` | `groq/qwen/qwen3.8-27b` | Must stay in a different model family from both agents |
| `RATE_LIMIT_REQUESTS` | `20` | Requests per minute per client; `0` disables |
| `MAX_MESSAGE_CHARS` | `4000` | Caps request size on a paid upstream |
| `SESSION_TTL_SECONDS` | `3600` | How long short-term memory survives |
| `ENABLE_DEBUG_ENDPOINTS` | on outside prod | `/diagnostics`, `/available-models`, `/docs` |
| `CORS_ORIGINS` | none | Comma-separated, only if the UI is served elsewhere |

## API

| Route | Purpose |
|---|---|
| `POST /chat` | `{session_id, agent, message}` → answer, tool calls, latency |
| `POST /reset` | Clears a session's short-term memory |
| `GET /health` | Liveness — dependency-free, safe for orchestrators |
| `GET /ready` | Readiness — fails if the KB didn't load |
| `GET /agents` | What each agent is currently wired to |
| `GET /config` | What the UI needs to render itself |
| `GET /diagnostics` | Per-dependency self-check *(debug only)* |
| `GET /available-models` | Live model list from each provider *(debug only)* |
| `POST /evals/run` | Starts a background evaluation *(debug only)* |
| `GET /evals/status` | Progress and results of the current run *(debug only)* |

`/diagnostics` is the one to reach for when something breaks: it pings both
providers, the KB, the embedder, and web search, and reports what each one
actually said. When a model ID has been retired it also attaches the
provider's current model list, so the fix is visible at the point of
failure.

## Tests

```bash
python -m pytest tests/ -q      # no network, no API keys
```

Every network dependency is faked, so the suite runs anywhere and CI needs
no secrets. `tests/README.md` records exactly what that does and doesn't
prove.

## Design decisions

Every non-obvious choice, the alternatives considered, and what each one
would cost is written up in **[docs/DESIGN.md](docs/DESIGN.md)** — including
the ones that are wrong for production and are deliberate scope decisions
for a one-day build.

## License

MIT — see [LICENSE](LICENSE).
