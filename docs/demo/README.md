# Demo

Two artefacts, because a demo has to choose between *reproducible* and
*real*, and this repo wanted both.

| Artefact | What it is | Trade |
|---|---|---|
| [`walkthrough.gif`](walkthrough.gif) + 8 stills | The real app, driven by Playwright, with provider responses **replayed** | Anyone can regenerate it with no API keys; latency chips read ~0.0s |
| [`live_session.md`](live_session.md) | An unedited transcript of a **live** conversation against the real providers | Real models, real tool calls, real latencies; needs your own keys |

![Walkthrough](walkthrough.gif)

## The replayed walkthrough

```bash
python scripts/make_demo.py
```

Everything is the real application — the real FastAPI app, UI, tool-calling
loop, knowledge base and guardrail layer. The agent switcher is labelled
from a live `GET /agents`. Server logs during capture show the tools
executing and `guardrails applied: ['disclaimer_appended']` firing on the
medication turn.

**One substitution:** the provider call replays scripted responses instead
of hitting the model APIs. That keeps the demo deterministic and
reproducible without keys, rather than depending on a free tier being
healthy the moment someone presses record — which, on this project, it
repeatedly was not.

| File | Shows |
|---|---|
| `01-empty-state.png` | Empty state, switcher labelled from `GET /agents` |
| `02-turn-1-oss.png` | KB-grounded answer, `lookup_kb()` chip, markdown |
| `03-turn-2-oss.png` | Multi-turn: user states a name and a symptom |
| `04-turn-3-oss.png` | **Short-term memory** — recalls "Sam", *no* tool call |
| `05-turn-4-frontier.png` | **Agent switch**, `search_web()` fires |
| `06-turn-5-oss.png` | **Safety**: declines medication, redirects to a professional |
| `07-tool-call-detail.png` | Expanded tool payload — what `lookup_kb` returned |
| `08-dark-theme.png` | Dark theme |

## The live transcript

[`live_session.md`](live_session.md) covers the same requirements against
real providers: KB grounding, multi-turn context, short-term memory, a
mid-conversation agent switch, web search, and a safety refusal that still
helps. Its latencies (16.2s / 20.0s / 29.5s open-source, 7.6s frontier) line
up with the percentiles in `results/scorecard.json`.
