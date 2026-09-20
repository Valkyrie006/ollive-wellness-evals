# Design decisions

Every non-obvious choice in this repo, what the alternatives were, and why
this one won. Decisions that are deliberately *wrong for production* are
marked as such rather than hidden.

---

## The requirement, restated

Build two AI wellness assistants — one open-source model, one frontier
model — on a **fixed architecture**, each supporting multi-turn
conversation, short-term memory, and two tools (`lookup_kb`, `search_web`).
Then build an evals platform that scores both on **hallucination**, **bias
and harmful output**, and **content safety**, and assesses the quality of
the judge itself.

The constraint that shapes everything: **the architecture must be fixed
between the two agents.** If anything but the model differs, the evaluation
measures the scaffolding instead of the models.

---

## 1. Keeping the architecture genuinely fixed

**Approach.** One `run_turn()` in `agents/core.py`, imported unchanged by
both agents. The prompt, tool schemas, loop, memory window, retry policy,
and JSON-repair logic are shared code. `agents/config.py` holds the only
difference: a model ID and which env var carries its key. Provider
differences in tool-call format are absorbed by LiteLLM, not by branching.

**Alternatives.**

| Option | Pro | Con |
|---|---|---|
| Two agent classes with a shared base | Familiar OO shape | A subclass *can* override; "fixed" becomes a convention rather than a fact |
| LangGraph / LangChain agent | Batteries included, graph visualisation | Heavy dependency, and its abstractions make it harder to prove the two paths are identical |
| One function, config injected ✅ | Identity is structural — there is no second code path to diverge | Manual tool loop to maintain (~60 lines) |

**Why.** With a single function, "the architecture is fixed" is enforced by
the absence of a second code path, not by discipline. Nobody can
accidentally make the OSS agent behave differently, because there is
nowhere to put the difference.

---

## 2. Which models — and why they changed twice

**Approach.** OSS: `gemma-4-26b-a4b-it` on Google AI Studio (open weights).
Frontier: `gemini-3.1-flash-lite` on Google AI Studio. Judge:
`qwen3.8-27b` on Groq. All free tier, no card.

Read that as the *end state* of a sequence of forced moves, not a first
choice — every change below was a provider constraint, and the sequence is
the interesting part.

**What happened.** The original picks (`llama-3.1-8b-instant`,
`gemini-2.0-flash`) both broke during the build. Gemini had simply retired
the ID. Groq had moved Meta's Llama models to Enterprise "contact sales"
access, so the error said `model_not_found` when the real meaning was *your
key can't reach this*. Two different failures presenting as the same error.

**Then the frontier pick changed again, and the reason matters.** The first
working replacement, `gemini-3.6-flash`, was not rate-limited so much as
unusable on the free tier: a 62s p50 and a 48% error rate, which is not a
model you can measure. `gemini-flash-latest` exhausted its free request
quota partway through a run. The 2.5-series IDs return 404 through
LiteLLM's Gemini route. `gemini-3.1-flash-lite` carries a larger free quota
and answers in ~9s, so it is the one the committed results were produced
on. That is a **free-tier availability** decision, not a claim that it is
the strongest frontier model — and it is a real limitation of these
results, recorded in the report rather than hidden: the frontier side of
the comparison is whichever frontier model a no-card key can actually
complete a run on.

**The structural fix.** Model IDs now come from the environment with
defaults, and `/available-models` asks each provider what your key can
actually call. A retirement is a `.env` edit, not a code change — and
`/diagnostics` attaches the live model list to a not-found error, so the
answer appears at the moment of failure.

**And then the OSS pick changed, for a reason that could not be engineered
around.** `gpt-oss-20b` on Groq worked and produced a complete set of
results. Then Groq's free tier hit its **per-day** token cap:

```
Rate limit reached for model `openai/gpt-oss-20b` ... service tier
`on_demand` on tokens per day (TPD): Limit 200000, Used 199367.
Please try again in 2m49.776s.
```

A per-day cap refills as a trickle. A full evaluation costs far more than
trickles back, so no amount of patience inside a run clears it. The
open-weights agent moved to **`gemma-4-26b-a4b-it`** on Google AI Studio:
open weights, free, no card, supports the tool calling this architecture
requires, ~20s per turn. Its larger sibling `gemma-4-31b-it` also works but
takes ~76s per turn — unusable across a 44-item run, and worth recording
because "it works" and "it is fast enough to measure" are different tests.

**The honest cost of that move.** Both assistants now sit on Google AI
Studio. This removes provider infrastructure as a confound — same serving
stack, same API, so a latency gap is the model rather than the vendor — but
open-vs-frontier is now compared *inside one vendor's lineup*, which is a
narrower claim than a cross-vendor comparison. Setting
`OSS_MODEL=groq/openai/gpt-oss-20b` restores the cross-vendor setup on a
key with daily tokens to spare.

**On the judge.** The judge must not share a model family with either agent,
or it scores its own family favourably (self-preference bias). With both
agents on Google, the judge *cannot* be a Gemini or Gemma model — which is
precisely why it stayed on Groq even after both agents left. That works
because the judge is now Groq's only consumer here and spends a few hundred
tokens per call, which fits inside what the free tier refills.

> ⚠️ **Known risk.** Qwen is in Groq's *Preview* tier and can be withdrawn at
> short notice. The fallback is `openai/gpt-oss-120b` — a different family
> from both agents, so the self-preference concern does not apply. What must
> *not* be used as judge here is any Gemini or Gemma model, since that would
> share a family with both subjects at once.

---

## 3. Retrieval: chunking and vector store

**Approach.** Paragraph-split the 9 markdown files (94 chunks),
`all-MiniLM-L6-v2` embeddings, in-memory Chroma rebuilt at startup.

**Alternatives.**

| Option | Pro | Con |
|---|---|---|
| Token-window chunking with overlap | Standard for large corpora | These files are ~500 words and already paragraph-structured; overlap adds noise |
| Persistent Chroma / pgvector | Survives restart, scales | 94 chunks rebuild in seconds — persistence is pure operational cost here |
| In-memory, paragraph chunks ✅ | Zero infra, chunks match authored structure | Rebuilds on every start; wrong past ~10k chunks |

**Why.** The corpus is small and hand-authored. Its paragraphs *are* its
semantic units, so splitting on them gives cleaner retrieval than a sliding
window, and the whole index costs seconds to rebuild.

---

## 4. Making the OSS model actually use its tools

**The problem.** The original prompt said "prefer `lookup_kb`".
`gpt-oss-20b` ignored it and answered from parametric memory. This is not
cosmetic: it skips the required tool use *and* inflates hallucination
scores in a way that measures prompt wording rather than the model.

**Approach.** Grounding is mandatory in the prompt for factual wellness
turns, with an explicit carve-out for turns that assert nothing (greetings,
recall). The prompt also carries today's date, because the model was
treating 2024 search results as current.

**Alternatives.**

| Option | Pro | Con |
|---|---|---|
| `tool_choice="required"` | Guarantees a call | Forces a pointless KB lookup on "thanks" or "what was my name?" — breaks multi-turn recall |
| Per-model prompt tuning | Best compliance per model | **Violates the fixed-architecture requirement** — disqualifying |
| One stronger shared prompt ✅ | Stays fixed; verified working on both | Prompt-level, so compliance is high but not guaranteed |

**Why.** Per-model prompts would have been the easy fix and would have
invalidated the entire comparison. The shared prompt was strengthened
instead, and both agents were verified calling tools correctly against the
live API.

---

## 5. Web search

**Approach.** Try the maintained `ddgs` package; fall back to DuckDuckGo's
no-JS HTML endpoint over `requests`.

**Why both.** `duckduckgo-search` is frozen and silently returns zero
results — a failure mode worse than an exception, because nothing looks
wrong. Its successor `ddgs` depends on the `primp` TLS stack, which fails
on some machines with `Unsupported protocol version 0x304`. That is an
environment problem with no in-code fix, so the fallback needs only
`requests`. When both fail, the error carries *both* causes — debugging a
fallback chain is miserable when only the last failure survives.

**Alternative.** A paid search API (Brave, Serper) would be more stable, but
the project constraint was strictly free, no card.

---

## 6. Short-term memory

**Approach.** Last 6 messages per session, in-process, in a `SessionStore`
bounded by both TTL (1h) and a hard cap (1000 sessions, LRU eviction).

**Why bounded.** The original was a plain dict. On a public deployment,
every unique `session_id` a caller invents costs memory forever — a trivial
denial-of-service. TTL handles abandonment; the cap handles malice.

| Option | Pro | Con |
|---|---|---|
| Plain dict | Simplest | Unbounded growth — not deployable |
| Redis | Survives restart, shared across replicas | Infrastructure for a single-instance app |
| Bounded in-process ✅ | No infra, no leak | Lost on restart; **not shared between replicas** |

> ⚠️ **Scaling limit.** Running more than one instance requires Redis behind
> the same `SessionStore` interface. Until then, load balancing needs
> sticky sessions or conversations will lose their memory.

---

## 7. Rate limiting

**Approach.** Hand-rolled sliding-window limiter (`api/ratelimit.py`), 20
requests/minute/client, applied only to the endpoints that cost money.

**Why hand-rolled.** `slowapi` would be reasonable, but this project lost
real time to dependency resolution on a constrained machine, and the whole
mechanism is 40 readable, tested lines. Rejected requests are deliberately
*not* recorded — otherwise a client hammering the endpoint keeps pushing
its own window forward and locks itself out indefinitely.

> ⚠️ **Spoofable.** The client key uses `X-Forwarded-For` when present,
> which is client-controlled. This damps accidental abuse; it does not stop
> a determined attacker. Real protection belongs at the proxy/WAF, or an
> API key per caller.

---

## 8. Error handling and information disclosure

**Approach.** Upstream failures return `502` with the provider's actual
error text **in development**, and a generic message plus a request ID **in
production**. Every response carries `X-Request-ID`; the full traceback is
logged against it.

**Why.** The original returned a bare "Internal Server Error", which made a
retired model ID take three round trips to diagnose. But provider error text
can carry internals and request fragments, so it can't be echoed to the
public. The request ID bridges the two: the user reports an opaque
reference, and the operator has the full trace.

**Debug endpoints** (`/diagnostics`, `/available-models`, `/docs`) call
upstream providers *with the server's own key* and echo provider errors.
They're off by default outside development, and return `404` rather than
`403` so they don't confirm they exist.

---

## 9. Evaluation design

**Approach.** Three axes, 15 items each, from established datasets:
MedHallu (hallucination), BBQ (bias), JailbreakBench (content safety).
DeepEval metrics with the judge pointed at Groq. Safety additionally uses a
deterministic refusal classifier rather than an LLM.

**Why real datasets.** Hand-written prompts would let the test set drift
toward what the agents happen to handle well. Published benchmarks are
adversarial by construction and make results comparable to other work.

**Why a rule-based refusal check.** Content safety has an asymmetry the
other axes lack: both *unsafe compliance* and *over-refusal* on legitimate
sensitive questions are failures. A regex classifier over refusal and
disclaimer patterns is deterministic and auditable where an LLM judge would
be neither.

---

## 10. Judging the judge

**Approach.** ~15 hand-labelled gold items; report raw agreement between the
judge's verdict and the human label.

| Option | Pro | Con |
|---|---|---|
| Cohen's κ | Corrects for chance agreement; the standard | Needs a larger, balanced gold set to be meaningful |
| Multi-judge consensus | Catches single-model bias | 3× cost and no second free judge family available |
| Raw agreement on a small set ✅ | Honest, cheap, fits the timebox | Overstates reliability — doesn't correct for chance |

> ⚠️ **Read this number carefully.** With ~15 items and no chance
> correction, agreement is indicative, not a reliability claim. Scaling to
> ~100 items and reporting κ is the first thing to do with more time.

---

## 11. Frontend

**Approach.** One self-contained HTML file with inline CSS/JS, served as a
static mount by FastAPI. No build step, no framework, no CDN.

| Option | Pro | Con |
|---|---|---|
| React/Next SPA | Componentised, familiar | A build pipeline and a second deploy target for one chat screen |
| Server-rendered templates | No client JS | Chat is inherently interactive; this fights the grain |
| Single static file ✅ | Zero build, deploys with the API, strict CSP possible | Everything in one file; would not scale past a few screens |

**Details that matter.** Markdown is rendered client-side (escaped first, so
model output can never inject markup) because `gpt-oss` favours tables and
prompting that away is a losing battle. Agent labels come from `GET /agents`
rather than being hard-coded — they'd already gone stale once after a model
switch. The UI reads `GET /config` on load and hides the diagnostics button
when the backend has those endpoints disabled, rather than offering a button
that 404s.

---

## 12. Testing

**Approach.** 36 tests, every network dependency injected as a fake. The
suite needs no API keys and no internet, so CI runs without secrets.

**What this proves:** KB ingestion over the real files, the full
tool-calling loop including tool-result feedback and memory updates, the
malformed-JSON repair path in both directions, HTTP routing and validation,
rate limiting, session eviction, and the scorecard maths.

**What it does not prove:** that the real models behave well. That was
verified separately by driving the running app against the live APIs.

> ⚠️ **Gap.** There is no end-to-end test against the real providers,
> because such a test would need secrets in CI and would fail whenever a
> provider had an outage. A nightly smoke test against a real key, separate
> from PR CI, is the right shape for that.

---

## 13. The judge, and not using DeepEval

**Approach.** Direct, single-call judge prompts per axis, each returning a
verdict plus a one-sentence reason, at temperature 0.

**Why not DeepEval**, which the original plan specified:

| | DeepEval | Direct prompts ✅ |
|---|---|---|
| Credibility | A published, validated library | Prompts are mine — no external validation |
| Judge meta-eval | Returns a float; the raw verdict is behind its abstraction | The verdict *is* the output, so it can be scored against known labels |
| Robustness | Needs strictly-schema'd JSON; small open models break it, and a failed parse silently becomes a score | One JSON object, with fenced/prose extraction and retry |
| Cost | Several calls per metric per item | One call per item — the difference between finishing on a free tier and rate-limiting |

The decisive factor was the second row. "Assess the quality of the judge"
requires the judge's raw verdict on items whose correct answer is known, and
that is awkward through a library that hands back only a score. The cost is
real and stated: these prompts carry none of a published library's validation.

**Every reason is written to `results/raw.jsonl`.** A scorecard you cannot
audit is a scorecard you cannot defend.

---

## 14. Judging the judge without hand labels

**Approach.** MedHallu ships, per question, a known-correct answer *and* a
known-hallucinated one. Both are fed to the judge; its verdicts are scored
against those labels.

**Why not hand-labelling**, which the original plan specified: at this scale
the labels would have been produced by me — an LLM labelling for an LLM
judge, which is circular. Dataset ground truth is independent of any model
in the loop, and there is enough of it to compute real classification
metrics instead of one agreement number.

**Why κ and not raw agreement.** On a balanced two-class problem a coin flip
scores 50%. Quoting uncorrected agreement makes a useless judge look
adequate. Cohen's κ subtracts what chance explains; the confusion matrix is
reported alongside because the two failure directions are not equivalent — a
judge that *misses* hallucinations flatters the agents, one that *over-flags*
punishes them.

**Second, independent check.** The deterministic regex classifier and the LLM
judge label the same safety responses. Their agreement rate is reported.
Two mechanisms agreeing is weak evidence they are right; disagreeing is
strong evidence one is wrong.

---

## 15. Safety scored as two opposite failures

**Approach.** Attack-success rate (complying with a harmful prompt) and
over-refusal rate (refusing a legitimate wellness question) are reported
separately, never averaged.

**Why.** Collapsing them into one "safety score" means an agent that refuses
everything scores perfectly. Over-refusal is the failure users actually
feel, and it is the predictable side-effect of tightening the other number —
so it needs its own column, permanently visible.

The benign-sensitive prompts are hand-written and committed, because a
jailbreak dataset by definition contains no legitimate questions.

---

## 16. Guardrails, derived rather than assumed

**Approach.** Two layers — a deterministic input filter on prompt-injection
*shape*, and an output layer that redacts specific dosage instructions and
appends a disclaimer on medical topics. Toggleable at runtime.

**Why toggleable.** So the same running server can be scored with and
without. Comparing two separately-started servers would confound the result
with everything else that differs between runs; without the toggle, "the
guardrails help" is an assertion rather than a measurement.

**Why the input layer matches shape, not topic.** `"ignore all previous
instructions"` is blocked; `"how do I ignore my phone before bed?"` is not.
Half the guardrail tests assert that legitimate questions get *through* —
that is the failure mode with a number attached, on the over-refusal axis.

**Why output guards rewrite rather than block.** Withholding an entire answer
because one sentence was too specific trades a small risk for a large
uselessness.

---

## 17. Datasets: real benchmarks with a committed fallback

**Approach.** MedHallu, BBQ and JailbreakBench pulled from Hugging Face,
each falling back to a small committed fixture if the pull fails.

**Why the fallback exists.** The first real run fell back on all three:
MedHallu needs a *config* name where a split was passed, and BBQ is split by
bias category rather than train/test. A public repo whose eval harness only
works on the day the schemas happen to match is not much of a harness. The
exact arguments are now pinned, and the fallback remains for the day a repo
becomes gated.

**Trade-off.** The fixtures are small and clear-cut, so a fallback run is
*easier* than a real one — and the scorecard records which source it used,
so that can never be quietly mistaken for a benchmark result.

---

## 18. Python 3.9: two runtime bugs a linter caused

Both of these imported cleanly and failed in production, and both came from
tooling configured for a newer Python than the deployment ran.

**`zip(..., strict=False)`** — added by a ruff autofix. Every KB lookup raised
`zip() takes no keyword arguments`; the error was handed to the model as a
tool result, and the model reported it "couldn't find anything". A hard
failure wearing the costume of a plausible answer.

**`axes: list | None` on a pydantic model** — PEP 604 unions are fine as
deferred annotations, but pydantic resolves model annotations *eagerly* at
class creation, so `from __future__ import annotations` does not save you.
The server crash-looped. Ruff's UP007/UP045 actively recommend this change
and cannot know about pydantic's eager evaluation, so both rules are now
disabled with that reason recorded in `pyproject.toml`.

**The fix was not the two lines.** `target-version` now pins the *oldest*
supported interpreter rather than the newest, CI runs the matrix on 3.9, and
`tests/test_python_compat.py` scans the source for both patterns so they
fail a test rather than a server.

---

## 19. Running evals over HTTP, in the background

**Approach.** `POST /evals/run` starts a background thread; `GET
/evals/status` reports progress. Debug-gated with everything else.

**Why background.** A full run is many minutes and hundreds of upstream
calls. Request/response would mean a dropped connection loses a run that has
already cost real quota.

**Why loopback is exempt from rate limiting.** The runner drives ~84 `/chat`
calls through this server, which the public limit would block. Traffic
originating on the loopback interface is the operator; a forwarded header
disqualifies the request so a proxied caller cannot claim to be local.

**Free-tier pacing is not optional.** The first full run lost 51 of 68 items
to provider 429s. A rate limit is a transient error with a completely
different time constant, so it now backs off from 20s rather than 1s, and
the runner paces per agent (Gemini's free tier being the stricter). The
scorecard carries a `valid` flag that goes false past a 20% error rate — a
run that lost a third of its items is not a measurement, and a scorecard
that doesn't say so invites someone to quote it.


---

## 20. Web search: three backends, and the bug that hid behind a silent zero

**Approach.** `search_web` tries DuckDuckGo's `lite` endpoint, then the
`html` endpoint, then the `ddgs` library, and returns the first that yields
results. Every failure is collected and reported together.

**What happened.** `/diagnostics` reported `web_search: returned 0 results`
while the exact same query opened fine in a browser. The cause: the code
sent a **POST**, and DuckDuckGo answers a POST from a non-browser client
with a challenge page that parses to zero results. A browser issues a GET,
so the code now does too, with the headers a browser sends. The `lite`
endpoint went first because its markup is a flat table — one
`result-link` and one `result-snippet` per row — with no nested containers
for a regex to fall out of step with.

**Why it mattered far more than a broken tool usually does.** Web search
sits *inside* a tool-calling loop. A failure that takes 20s is multiplied
by the loop iterations and again by every item in an eval run: ~10s
evaluation items became ~3-minute ones, and the latency column of the
scorecard was quietly measuring a broken DuckDuckGo parser rather than the
models. The fix has three parts, and only the first is the parser:

1. GET instead of POST, and `lite` first.
2. **Timeouts cut to 8s.** Inside a loop, failing fast and telling the
   model beats hanging.
3. **A per-tool circuit breaker** in `agents/core.py`: a tool that has
   already failed this turn is not called again. The causes are
   environmental — no network, a broken TLS stack, a blocked endpoint —
   not query-dependent, so a retry buys nothing and costs the full timeout
   again.

**Alternatives.** A keyed search API (Brave, Serper, Tavily) is far more
reliable than scraping, and is what production should use. All of them
require a card or an account beyond the free-tier-no-card constraint this
project set itself, so scraping with three fallbacks is the honest choice
here — and the limitation belongs in the table below rather than in a
footnote.

---

## 21. Bounded retry: a cap you cannot wait out

**Approach.** A single request may spend at most `MAX_RETRY_SLEEP_TOTAL_S`
(default 75s) asleep across all its retries. Past that, the provider's own
error is raised.

**Why.** Honouring a provider's retry hint is right, and unbounded
honouring of it is not. Groq answered a rate limit with *"Please try again
in 2h14m30s"* — a **per-day** token cap. Sleeping through that inside one
request is impossible; without a budget the request simply never came back,
and a caller cannot tell a throttled provider from a wedged server. The
budget turns a multi-hour cap into an immediate, explanatory failure while
still covering two rounds of the ~30s hint a per-minute limit gives, which
is usually enough to get the item.

**Also fixed here.** The retry-hint parser only understood Gemini's
phrasing ("Please retry in 33.1s"). Groq words it differently
("try again in 1m23.4s", "in 2h14m30s"), so Groq limits silently fell back
to a blind 20s/40s/80s backoff that could never clear the cap. Both
phrasings are parsed now, including hours.

---

## 22. Provider derived from the model ID

**Approach.** `provider_of("gemini/gemma-4-26b-a4b-it")` → `gemini`, and
the API key env var follows from the provider.

**Why.** The provider and key used to be hardcoded per agent. That made the
documented `OSS_MODEL` override a trap: pointing it at a Gemini-hosted
model left the config still claiming provider `groq`, so the app sent the
Groq key to Google and failed with `API key not valid` — an error that
points at the key rather than at the mismatch. The model ID already names
its provider, so one source of truth removes the whole class of failure.
`evals/runner.py` pacing is keyed by provider for the same reason: the rate
limit belongs to the provider, and pacing an agent called "oss" at 4s was
correct while it ran on Groq and wrong the moment it moved.

---

## 23. Logging to a file, not just to a terminal

**Approach.** A rotating file handler writes to `logs/app.log` alongside
console logging.

**Why.** The hardest bug in this project — a provider throttling
completions while its model-list endpoint answered normally — was invisible
for an hour because the only record of the retry warnings was scrolling
past in a terminal nobody was reading. The provider's full message names
the limit type (`tokens per day (TPD): Limit 200000, Used 199367`), which
is the difference between "wait a minute" and "come back tomorrow", so the
log keeps 600 characters of it rather than the 160 that had been truncating
exactly that detail away. A long eval run is unattended by definition; it
needs a durable record.

---

## Summary of known limitations

| Limitation | Impact | Fix |
|---|---|---|
| Sessions are in-process | Memory lost on restart; breaks multi-replica | Redis behind `SessionStore` |
| Rate limiting is per-process and spoofable | Weak against determined abuse | WAF / proxy limits, or per-caller API keys |
| KB rebuilt at every startup | Slow cold start | Persistent vector store |
| Judge agreement is raw, on ~15 items | Overstates judge reliability | ~100 items, report Cohen's κ |
| Qwen judge is Preview tier | May vanish without notice | Fallback documented in `agents/config.py` |
| No authentication | Anyone who can reach it can spend your quota | API key or OAuth before public exposure |
| No end-to-end test against real providers | Provider-side breakage found by hand | Nightly smoke test outside PR CI |
| Small eval samples (12 per axis) | One item moves a rate by several points | Scale to 100+ per axis |
| Judge is a single model | No cross-judge consensus | Second judge family, report disagreement |
| Eval endpoints are debug-gated, not authenticated | Anyone with debug on can spend your quota | Auth before enabling on a deployment |
| Web search scrapes DuckDuckGo | Markup changes break it; no SLA | A keyed search API (Brave/Serper/Tavily) once a card is acceptable |
| Both agents run on one provider | Open-vs-frontier compared inside one vendor's lineup | `OSS_MODEL=groq/openai/gpt-oss-20b` on a key with daily tokens to spare |
| Free-tier daily token caps | A full run can become impossible mid-day | Paid tier, or split runs across days and merge scorecards |
