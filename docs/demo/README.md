# Demo

![Walkthrough](walkthrough.gif)

A five-turn walkthrough of the running application, regenerable with:

```bash
python scripts/make_demo.py
```

## What's actually running

The **real** application: the real FastAPI app, the real single-file UI, the
real tool-calling loop in `agents/core.py`, the real knowledge base, and the
real guardrail layer. The agent switcher is labelled from a live
`GET /agents`, so the model names in the header are the ones the app is
actually configured with. The server logs during capture show the tools
executing and `guardrails applied: ['disclaimer_appended']` firing on the
medication turn.

**One thing is substituted:** the provider call itself replays scripted
responses instead of hitting Groq and Google live. That is deliberate —
it makes the demo deterministic and reproducible by anyone who clones this
repo without API keys, rather than depending on two free tiers being
healthy at the moment someone presses record — which, on this project, they
repeatedly were not: one provider exhausted its per-day token cap mid-build. It also means the latency
chips read ~0.0s, because no network call happens; real measured latencies
are in `results/scorecard.json`.

For live behaviour against the real providers, run the app with your own
keys and use the **Diagnostics** button in the header.

## Frames

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

Each of these maps to a requirement: multi-turn conversation, short-term
memory, both tool calls, and the two agents on one interface.
