# Design decisions

Every non-obvious choice, what the alternatives were, and why this one won.
Choices that are deliberately *wrong for production* are marked as such.

Diagrams: **[DIAGRAMS.md](DIAGRAMS.md)** · System shape and trajectory:
**[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## The constraint that shapes everything

The architecture must be **fixed** between the two agents. If anything but
the model differs, the evaluation measures the scaffolding instead of the
models. Most of the difficulty lives here, because the easy fixes during
development are exactly the ones that break it — tuning the prompt for
whichever model is underperforming, adding a workaround for one provider.

---

## The decisions

| # | Decision | Chosen | Alternatives rejected |
|---|---|---|---|
| 1 | **Fixed architecture** | One `run_turn()`, config injected | Base class + subclasses (*a subclass can override — the guarantee becomes a code-review promise*); LangChain/LangGraph (*identity becomes a claim about a framework's internals*) |
| 2 | **Retrieval** | Paragraph chunks (94), MiniLM, in-memory Chroma | Token-window + overlap (*these files are already paragraph-structured; overlap adds noise*); persistent store (*94 chunks rebuild in seconds*) |
| 3 | **Tool compliance** | One stronger shared prompt, mandatory grounding | `tool_choice="required"` (*forces a pointless lookup on "thanks", breaks recall*); per-model prompt tuning (***violates the fixed-architecture requirement — disqualifying***) |
| 4 | **Short-term memory** | Bounded in-process store: 6 messages, 1h TTL, 1000-session LRU | Plain dict (*unbounded — every invented session id costs memory forever*); Redis (*infra for a one-day build*) |
| 5 | **Rate limiting** | Hand-rolled sliding window, loopback exempt | A library (*one more dependency for ~40 lines*); no limit (*a chat endpoint on a paid upstream is a money faucet*) |
| 6 | **Error disclosure** | Upstream detail in dev, generic + request id in prod | Always verbose (*leaks provider internals*); always generic (*undebuggable locally*) |
| 7 | **Interface** | FastAPI + single-file web UI | Streamlit/Gradio (*faster to write, but the harness needed a real HTTP surface anyway — this way there is one path, not two*) |
| 8 | **Scoring** | Direct per-axis prompts, temp 0 | Regex (*can't judge whether a paraphrase is faithful*); DeepEval (*prompts hidden inside the library — can't show what the judge was asked*) |
| 9 | **Datasets** | MedHallu · BBQ · JailbreakBench, with committed fallbacks | Hand-written prompts (*a dataset I author is one I can unconsciously tune to*) |
| 10 | **Judge model** | `qwen3.8-27b` — a third family | Same family as either agent (*self-preference bias lands exactly on the comparison this project exists to make*) |
| 11 | **Judge quality** | Dataset ground truth + Cohen's κ | Hand-labelling (*at this scale, an LLM labelling for an LLM judge*); raw agreement (*flatters any judge on an unbalanced set*) |
| 12 | **Safety metric** | Two rates, never averaged | A single safety score (*improvable by refusing everything — the exact failure a wellness assistant must not have*) |
| 13 | **Guardrails** | Derived from observed failures; runtime-toggleable | Rules added on suspicion (*how an assistant ends up refusing to discuss sleep*); two separate servers for the A/B (*confounds the result with everything else that differs*) |
| 14 | **Web search** | 3 DuckDuckGo backends, GET, 8s timeout | Keyed API — Brave/Serper/Tavily (*needs a card; outside the free constraint*). **Weakest part of the system** |
| 15 | **Eval transport** | Over HTTP, in the background | In-process calls (*would test a different path than users hit*); synchronous (*a dropped connection would lose a run that cost real quota*) |
| 16 | **Retry** | Provider's own hint, capped at 75s total | Unbounded (*"try again in 2h14m" became a hang*); fixed backoff (*can never clear a per-day cap*) |
| 17 | **Provider config** | Derived from the model ID prefix | Hardcoded per agent (*made the documented `OSS_MODEL` override a trap — sent the Groq key to Google*) |
| 18 | **Logging** | Rotating file + console, 600 chars of provider error | Console only (*the hardest bug was invisible for an hour*); truncated at 160 (*cut off exactly the limit type*) |
| 19 | **Settings** | Frozen dataclass over `os.getenv` | pydantic-settings (*reasonable, but ~40 lines vs another dependency on a constrained machine*) |
| 20 | **Python floor** | 3.9 | 3.10+ (*3.9 ships with macOS; two 3.10-only idioms imported fine and failed at runtime here*) |
| 21 | **Tests** | Every provider call faked | Live-API tests (*CI would need secrets, and a provider outage would fail the build*) |
| 22 | **Report** | Findings **derived** from the scorecard | Hand-written conclusions (*"latency favours the open-source agent" survived into a run where it was false*) |
| 23 | **Demo** | Replayed GIF **and** a live transcript | Replay only (*~0.0s latencies*); live only (*not reproducible without keys*) |

---

## The four worth more than a table row

### Which models — and why they changed three times

**Now:** OSS `gemma-4-26b-a4b-it`, frontier `gemini-3.1-flash-lite`, both on
Google AI Studio; judge `qwen3.8-27b` on Groq. Read that as the *end state*
of forced moves, not a first choice.

1. **Llama 3.1 on Groq** → Groq moved Meta's models to Enterprise access, so
   a standard key gets `model_not_found` when the real meaning is *your key
   can't reach this*.
2. **`gpt-oss-20b` on Groq** → worked, produced a full set of results, then
   the free tier hit its **per-day** token cap:
   `tokens per day (TPD): Limit 200000, Used 199367`. A per-day cap refills
   as a trickle; no amount of patience inside a run clears it.
3. **`gemma-4-26b-a4b-it` on AI Studio** → open weights, free, no card, tool
   calling, ~20s/turn. Its larger sibling `gemma-4-31b-it` also works but
   takes ~76s/turn — unusable across 44 items, and worth recording because
   "it works" and "it's fast enough to measure" are different tests.

Frontier moved too: `gemini-3.6-flash` gave 62s p50 and a 48% error rate;
`gemini-flash-latest` exhausted its quota mid-run; the 2.5-series 404s
through LiteLLM.

**The honest cost:** both assistants now sit with one vendor. That removes
provider infrastructure as a confound — same serving stack, so a latency gap
is the model and not the vendor — but narrows open-vs-frontier to one
vendor's lineup. `OSS_MODEL=groq/openai/gpt-oss-20b` restores the
cross-vendor setup on a key with daily tokens to spare.

**On the judge:** with both agents on Google, the judge *cannot* be a Gemini
or Gemma model — which is precisely why it stayed on Groq after both agents
left. It works because the judge is now Groq's only consumer here and spends
a few hundred tokens per call.

> ⚠️ Qwen sits in Groq's *Preview* tier and can be withdrawn at short notice.
> Fallback: `openai/gpt-oss-120b` — a different family from both agents, so
> the self-preference concern doesn't apply. What must **not** be used is any
> Gemini or Gemma model, which would share a family with both subjects.

### Judging the judge without hand labels

MedHallu ships a known-correct **and** a known-hallucinated answer per
question, so the labels already exist. Feed the judge both; score its
verdicts for accuracy, precision, recall and **Cohen's κ** — κ rather than
raw agreement, because raw agreement flatters any judge on an unbalanced
set.

Result: **κ = 1.00 on 12 labelled cases** — and the report refuses to call
that "reliable." The pairs are unambiguous, so it shows the judge handles
the easy case, not that it generalises. To cover the gap, a deterministic
regex classifier labels the same safety responses independently: the two
agree only **65%** of the time. So the report states plainly that the judge
is calibrated for hallucination, *not* demonstrably for safety.

### Web search: the bug behind a silent zero

`/diagnostics` reported `returned 0 results` while the same query opened
fine in a browser. The cause: the code sent a **POST**, and DuckDuckGo
answers a POST from a non-browser client with a challenge page that parses
to zero results. It now GETs, with browser headers, against the `lite`
endpoint first (a flat table — one `result-link` and one `result-snippet`
per row, nothing nested for a regex to fall out of step with).

Why it mattered disproportionately: web search sits *inside* the tool loop,
so a 20s failure is multiplied by the iterations and again by every eval
item. ~10s items became ~3-minute ones, and the scorecard's latency column
was measuring a broken parser rather than the models. The fix has three
parts and only the first is the parser: GET instead of POST; timeouts cut to
8s; and a **per-tool circuit breaker** so a tool that already failed this
turn is not called again — the causes are environmental, so a retry buys
nothing and costs the full timeout again.

### Safety as two opposite failures

`attack_success_rate` (complied with a jailbreak) and `over_refusal_rate`
(refused a benign question about a sensitive topic) are reported separately,
with 4 benign-but-sensitive controls alongside 6 harmful prompts. Averaging
them produces a score a model improves by refusing everything. Without the
controls, the safest-looking agent is the most useless one.

---

## Known limitations

| Limitation | Impact | Fix |
|---|---|---|
| Small samples (6/6/10 per axis) | One item moves a rate by several points | Scale to 100+, report confidence intervals |
| Judge validated on hallucination only | Bias and safety verdicts are unchecked; 65% agreement with the rule classifier | Labelled subsets for the other two axes |
| Single-turn eval cases only | Jailbreaks and drift are strongest *across* turns | Multi-turn cases |
| Web search scrapes DuckDuckGo | Markup changes break it; no SLA | Keyed search API once a card is acceptable |
| Both agents on one provider | Comparison narrowed to one vendor's lineup | `OSS_MODEL=groq/...` on a key with daily tokens |
| Sessions in-process, KB in-memory | Lost on restart; blocks multi-replica | Redis + persistent vector store |
| Guardrail toggle is process-global | An eval run affects live traffic on that server | Per-request config |
| Eval endpoints debug-gated, not authenticated | Anyone who reaches them can spend your quota | Auth before any deployment |
| Results written only at the end | A crash loses the run | Checkpoint per item |
| Rate limiting per-process and spoofable | Weak against determined abuse | WAF/proxy limits, or per-caller keys |
| Single judge | No cross-judge consensus | Second family, report disagreement |
