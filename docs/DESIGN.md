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

**Approach.** OSS: `openai/gpt-oss-20b` on Groq. Frontier:
`gemini-3.6-flash` on Google AI Studio. Judge: `qwen3.8-27b` on Groq. All
free tier, no card.

**What happened.** The original picks (`llama-3.1-8b-instant`,
`gemini-2.0-flash`) both broke during the build. Gemini had simply retired
the ID. Groq had moved Meta's Llama models to Enterprise "contact sales"
access, so the error said `model_not_found` when the real meaning was *your
key can't reach this*. Two different failures presenting as the same error.

**The structural fix.** Model IDs now come from the environment with
defaults, and `/available-models` asks each provider what your key can
actually call. A retirement is a `.env` edit, not a code change — and
`/diagnostics` attaches the live model list to a not-found error, so the
answer appears at the moment of failure.

**On the judge.** The judge must not share a model family with either agent,
or it scores its own family favourably (self-preference bias). Moving the
OSS agent onto GPT-OSS forced the judge off it, onto Qwen. That leaves three
distinct families: Qwen judging GPT-OSS and Gemini.

> ⚠️ **Known risk.** Qwen is in Groq's *Preview* tier and can be withdrawn at
> short notice. The fallback is `openai/gpt-oss-120b`, but that shares a
> family with the OSS agent, and if you use it the self-preference caveat
> belongs in the evaluation report.

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
