# Architecture: where this is, and where it should go

Diagrams live in **[DIAGRAMS.md](DIAGRAMS.md)**. Decisions and their
alternatives live in **[DESIGN.md](DESIGN.md)**. This file is the shape of
the system, what building and running it taught us, and the trajectory.

---

## 1. Today

**One process does everything.** The FastAPI app serves the UI, hosts both
agents, *and* runs the evaluation as a background thread inside itself.
Results are JSON and JSONL on local disk.

Shape: **a monolith with an embedded test harness.** It answers *"which
model is better, today, on this dataset?"* well — 44 items, zero errors, a
signed conclusion.

**What's right:** the harness calls `POST /chat`, so it is a *client* of the
app and cannot drift from what users get. The judge is a separate model
family. Results are plain files, so every number is inspectable.

**What's structurally limited** — consequences of the shape, not bugs:

| Limit | Consequence |
|---|---|
| Harness runs **inside** the system under test | Guardrail toggle leaks into live traffic; eval competes with the app for rate limits; a crash takes both down |
| Results written **only at the end** | Two runs lost to a sleeping laptop, ~20 min of real quota each |
| Results are **files, not records** | Answers "what is it now?", never "did this get worse since Tuesday?" |
| Scores carry **no version tuple** | A scorecard from `gpt-oss-20b` and one from `gemma-4` are incomparable, and nothing in the format stops you comparing them |

---

## 2. What the evaluation taught us about the design

Four findings that could not have been asserted in advance.

**Scaffolding determines safety; the model determines accuracy.** Both
agents scored *identically* on attack success (17%) and over-refusal (0%),
while differing sharply on hallucination (50% vs 33%) and bias (50% vs 17%).
Safety came from the parts held constant — prompt and guardrails.

> Don't fix safety by swapping models. And note this is only visible
> *because* the architecture is fixed: with per-model prompts, identical
> safety numbers and divergent accuracy numbers would be uninterpretable.

**Guardrails are not uniformly effective.** 17% → 0% attack success on the
frontier agent, unchanged on the open-source one, zero over-refusal cost on
both.

> A guardrail interacts with the model behind it, so its effect is a
> per-deployment claim. The runtime toggle earned its place — without an A/B
> on the same server this would have averaged to "17% → 8%" and hidden both
> facts.

**Judge reliability is axis-dependent.** κ = 1.00 on hallucination, but only
**65% agreement** with the deterministic refusal classifier on safety.

> You cannot validate a judge once and call it validated. Ground truth
> exists for hallucination and nothing else, so two of three axes are
> unvalidated. This is the largest open gap in the platform.

**A broken tool corrupts measurements silently.** Web search failing inside
the tool loop turned ~10s items into ~3-minute ones; the latency column was
measuring a broken HTML parser rather than the models.

> An eval harness must distinguish *system* failure from *model* failure.
> Hence errored items excluded from denominators, a `valid: false` flag past
> a 20% error rate, and a per-tool circuit breaker.

---

## 3. What the build taught us about the code

Eleven defects were found between the first working version and a clean run.

| Defect | How it presented |
|---|---|
| `search_web` sent a **POST**; DuckDuckGo answers a POST from a non-browser client with a challenge page | "returned 0 results" while the same query opened fine in a browser |
| Retry sleep was **unbounded** | "try again in 2h14m30s" became a hang with no error |
| Groq's retry-hint wording never parsed (only Gemini's was) | Groq limits fell back to a blind backoff that could never clear a per-day cap |
| Provider and key **hardcoded per agent** | The documented `OSS_MODEL` override became a trap: it sent the Groq key to Google |
| No **circuit breaker** on a failing tool | A dead dependency retried at full timeout every iteration |
| `run_all` assigned `label = f"{agent}/{axis}"`, shadowing its own parameter | Scorecard labelled `"frontier/safety"` instead of the run label |
| Latency finding **hardcoded** to "favours the open-source agent" | True of one run; false at 22.1s vs 6.6s |
| Guardrail finding read **only one agent** | Reported "no change", hid frontier's 17% → 0% |
| Comparisons stated with **no minimum sample size** | "Both agents within 0 points" — from 4 items on one side |
| UI rendered `---` as literal dashes | Disclaimer separator showed as text |
| Two UI error hints gave **stale advice** | Pointed at a file that no longer holds model IDs |

### The pattern: almost every one was silent

Nothing threw. Web search returned an empty list; the scorecard wrote a
wrong label; the report asserted a conclusion its own data contradicted.
Each was found by *looking*, not by a failure — so the defect class here is
not "crashes" but **"confidently wrong output."**

Three consequences that generalise:

**Observability first, not fifteenth.** One defect was invisible for an hour
because the only record of the retry warnings was scrolling past in a
terminal. A rotating file handler is what exposed the provider's full
message — `tokens per day (TPD): Limit 200000, Used 199367` — the difference
between "wait a minute" and "come back tomorrow." A long run is unattended
by definition.

**Anything asserted in prose that duplicates data is a latent bug.** Three
defects are the same mistake: a conclusion written once against one dataset,
surviving into a run whose numbers had moved.

**Derive, never duplicate.** Two more are a second copy of a fact drifting
from the first — provider/key hardcoded per agent, and pacing keyed by agent
name rather than by the provider that owns the rate limit.

---

## 4. The missing capability: failure attribution

The scorecard says *"open-source: 50% hallucination."* That cannot be acted
on, because five causes produce it and each has a different fix:

| Cause | Fix |
|---|---|
| Retrieval never surfaced the right chunk | chunking, embedding model, `k` |
| It did, but the model ignored it | prompt |
| It did, the model used it, still wrong | model |
| A tool errored | engineering |
| The judge was wrong | fix the instrument, not the agent |

The cheapest high-value change is to **record which one, per failed item.**
For hallucination that is one extra field — *was the ground-truth-bearing
chunk in the retrieved set?* If yes and the answer was still wrong, it's a
prompt or model problem. If no, it's retrieval, and swapping models won't
help at all.

Without this you optimise blind. With it, the scorecard stops being a report
card and becomes a work queue.

---

## 5. North star

Four structural changes: **harness outside the system under test**,
**per-item checkpointing on a queue**, **results in a store keyed by a
version tuple** (build · dataset · prompt · judge · guardrails), and
**evaluation as a gate rather than an errand**. Sequence diagram in
[DIAGRAMS.md](DIAGRAMS.md).

| Change | The pain it fixes |
|---|---|
| Harness outside the SUT | Guardrail toggle stops leaking into live traffic; runs stop competing with the app for quota |
| Queue + workers, checkpoint per item | Two runs lost to a sleeping laptop; at 45 min and real quota per run, resumability isn't a nicety |
| Version tuple per run | A score is only meaningful as a tuple of what produced it |
| Results store, not files | Only a store answers "did this get worse?" — the question that matters once you're shipping |
| Two judge families, humans on disagreement only | 65% agreement on safety means a third of verdicts are contested; routing *only* disagreements bounds judge error and keeps the human step small |
| CI gates on regression thresholds | Evals are currently something a person runs. They should block a merge |
| Parallel workers | 45 min sequential, purely because the harness shares rate limits with the app |
| Online eval on sampled traffic | Offline sets go stale and never contain the thing that breaks you |

### What it buys the business

**Ship velocity** — without a gate every prompt change is a gamble and the
team slows down out of fear; with one you ship *faster* because you can
prove you didn't break anything. Usually the largest and least-discussed
return.

**Model cost arbitrage** — if an open model is good enough for even 60% of
traffic that's a direct margin line. Today's data says *not yet*, but open
models improve monthly and this is how you notice the day it flips.

**Quantified liability** — wellness advice is medical-adjacent. "17% attack
success, 0% over-refusal" is exposure with a number on it, which is what an
incident review, a security questionnaire or a regulator will ask for.

**Vendor independence** — demonstrated equivalence lets you switch when a
provider raises prices or degrades. This project lived it: Groq's per-day
cap forced a model change mid-build.

**Trust as a product feature** — in health, publishing a measured
hallucination rate is a sales asset.

*Cost:* roughly a week of engineering; each run is a few hundred model calls
— cents on a paid tier. The expensive line is human adjudication, which is
why only judge disagreements reach a person.

---

## 6. Roadmap — by value per unit of effort

The instinct is to build the pipeline first. That's wrong: items 1 and 2
cost about a day together and capture most of the value.

| # | Build | Effort | Why here |
|---|---|---|---|
| 1 | Frozen baseline + CI gate | Hours | Stops regressions immediately; no new infrastructure — diff two JSON files |
| 2 | Failure attribution (§4) | ~1 day | Report card → work queue |
| 3 | Ground truth for bias + safety | Days | Two of three axes unvalidated; ~50 labelled items per axis |
| 4 | Multi-turn eval cases | Days | Jailbreaks are strongest *across* turns; we measure the easy case |
| 5 | Scale with stratification | Days | n=6 means one item moves a rate 17 points |
| 6 | Second judge + disagreement routing | ~1 week | Bounds judge error rather than caveating it |
| 7 | Separate harness, queue, results store | ~1 week | Pays once runs are frequent and someone depends on them |
| 8 | Online eval on sampled traffic | ~1 week | Catches drift a frozen offline set never will |

**Code track**, from §3: assertions on its own output; checkpoint each item;
startup health gate; structured logging with run/request ids; per-request
guardrail config; auth on the eval endpoints; Redis sessions and a
persistent vector store.

---

## The honest framing

Nothing in the north star is *wrong* with the current design for what it is:
a one-day build on free tiers, where a background thread and a JSON file are
proportionate. The gap is not sloppiness.

It is that the current shape answers **"which model is better today?"** and
the north star answers **"did anything get worse since the last deploy?"**

The first is a report. The second is infrastructure — and it is the one an
evals platform exists to provide.
