# Demo

Two artefacts, because a demo has to choose between *reproducible* and
*real* and this repo wanted both:

| Artefact | What it is | Trade |
|---|---|---|
| [`walkthrough.gif`](walkthrough.gif) + the 8 stills | The real app, driven by Playwright, with provider responses **replayed** from a recorded run | Anyone can regenerate it with no API keys; latency chips read ~0.0s |
| [`live_session.md`](live_session.md) | An unedited transcript of a **live** conversation against the real providers | Real models, real tool calls, real latencies; needs your own keys to reproduce |

![Walkthrough](walkthrough.gif)

## The replayed walkthrough

```bash
python scripts/make_demo.py
```

Everything is the **real** application: the real FastAPI app, the real
single-file UI, the real tool-calling loop in `agents/core.py`, the real
knowledge base, the real guardrail layer. The agent switcher is labelled
from a live `GET /agents`, so the model names in the header are the ones the
app is actually configured with. The server logs during capture show the
tools executing and `guardrails applied: ['disclaimer_appended']` firing on
the medication turn.

**One thing is substituted:** the provider call itself replays scripted
responses instead of hitting the model APIs. That is deliberate — it makes
the demo deterministic and reproducible by anyone who clones this repo
without keys, rather than depending on a free tier being healthy at the
moment someone presses record, which on this project it repeatedly was not.
It also means the latency chips read ~0.0s, because no network call happens.

| File | What it shows |
|---|---|
| `01-empty-state.png` | Empty state, agent switcher labelled from `GET /agents` |
| `02-turn-1-oss.png` | KB-grounded answer, `lookup_kb()` chip, markdown rendering |
| `03-turn-2-oss.png` | Multi-turn: user states a name and a symptom |
| `04-turn-3-oss.png` | **Short-term memory** — recalls "Sam" with *no* tool call |
| `05-turn-4-frontier.png` | **Agent switch** to the frontier model, `search_web()` fires |
| `06-turn-5-oss.png` | **Safety**: declines to recommend medication, redirects to a professional |
| `07-tool-call-detail.png` | Expanded tool payload — what `lookup_kb` actually returned |
| `08-dark-theme.png` | Dark theme |

## The live transcript

[`live_session.md`](live_session.md) is the counterpart: a real five-turn
conversation, captured unedited, covering the same requirements —
KB grounding, multi-turn context, short-term memory, a mid-conversation
agent switch, web search, and a safety refusal that still helps. Its
latencies (16.2s / 20.0s / 29.5s on the open-source agent, 7.6s on the
frontier one) line up with the percentiles in `results/scorecard.json`.

For live behaviour against your own keys, run the app and use the
**Diagnostics** button in the header.
