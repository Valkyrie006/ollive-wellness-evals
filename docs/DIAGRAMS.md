# Diagrams

Every diagram in one place. GitHub renders these natively.

---

## System

```mermaid
flowchart LR
  U(["user"]) --> API["FastAPI + web UI"]
  API --> RT["run_turn()<br/>one shared code path"]
  CFG["config.py<br/>the ONLY difference"] --> RT
  RT --> KB["lookup_kb<br/>Chroma · 94 chunks"]
  RT --> WEB["search_web<br/>DuckDuckGo"]
  RT --> OSS["Gemma 4 26B<br/>open weights"]
  RT --> FR["Gemini 3.1 Flash Lite<br/>frontier"]
  EV["evals platform"] -. "drives the same<br/>POST /chat" .-> API
```

The evals platform is a **client of the running app, not a fork** — whatever
it measures is what a user gets.

---

## Why "fixed architecture" is structural

```mermaid
flowchart TB
  subgraph R["Rejected: base class + subclasses"]
    direction TB
    BA["BaseAgent.run_turn()"]
    OA["OssAgent"] -. "can override" .-> BA
    FA["FrontierAgent"] -. "can override" .-> BA
  end
  subgraph C["Chosen: one function, config injected"]
    direction TB
    OC["OSS_CONFIG"]
    FC["FRONTIER_CONFIG"]
    RT["run_turn()<br/>the only code path"]
    OC -- "passed in" --> RT
    FC -- "passed in" --> RT
  end
  R ~~~ C
```

The dashed arrows on the left *are* the problem: edges that may or may not
exist at runtime. On the right there are none, because there is no second
implementation. `scripts/verify_requirements.py` asserts the repo contains
exactly one `def run_turn`.

---

## Sequence: one chat turn

```mermaid
sequenceDiagram
    actor U as User
    participant API as FastAPI
    participant G as Guardrails
    participant RT as run_turn()
    participant M as SessionStore
    participant T as Tools
    participant P as Model (from config)

    U->>API: POST /chat {session, agent, message}
    API->>G: check_input()
    G--xU: blocked if injection shape (no model call)
    API->>RT: run_turn(config for that agent)
    RT->>M: last 6 messages
    M-->>RT: history
    loop max 2 tool iterations
        RT->>P: completion(history + tool schemas)
        P-->>RT: tool_call or final text
        opt tool_call
            RT->>T: lookup_kb / search_web
            T-->>RT: result, or error → circuit breaker opens
        end
    end
    RT->>G: apply_output_guards()
    G-->>RT: dosage redacted + disclaimer appended
    RT->>M: save history
    RT-->>U: answer + tool trace + latency
```

---

## Sequence: one evaluation run (today)

```mermaid
sequenceDiagram
    actor Dev
    participant P as FastAPI process
    participant T as Eval thread<br/>(same process)
    participant G as Guardrails<br/>(process-global flag)
    participant M as Model APIs
    participant FS as Local files

    Dev->>P: POST /evals/run
    P->>T: spawn background thread
    T->>G: set_enabled(False)
    Note over G: ⚠ also disables guardrails<br/>for live users on this server
    loop 44 items, sequential, paced
        T->>P: POST /chat (localhost)
        P->>M: completion
        M-->>P: answer
        P-->>T: answer + latency
        T->>M: judge call
        M-->>T: verdict + reason
    end
    T->>FS: write scorecard.json + raw.jsonl
    Note over T,FS: ⚠ written only at the end —<br/>a crash loses the whole run
    Dev->>FS: render charts + PDF
```

---

## Sequence: one evaluation run (north star)

```mermaid
sequenceDiagram
    actor CI
    participant O as Orchestrator
    participant Q as Work queue
    participant W as Workers (n)
    participant SUT as System under test<br/>(deployed, versioned)
    participant J as Judge pool<br/>(2+ families)
    participant DB as Results store
    participant H as Human review queue

    CI->>O: evaluate(build sha, dataset v, judge v)
    O->>DB: open run, record the version tuple
    O->>Q: enqueue one job per (item × agent)
    par workers pull independently
        W->>Q: claim job
        W->>SUT: call over the network
        SUT-->>W: answer
        W->>J: score with judge A and judge B
        J-->>W: two verdicts
        W->>DB: checkpoint this item
        alt judges disagree
            W->>H: route for human adjudication
        end
    end
    O->>DB: aggregate, diff against frozen baseline
    O-->>CI: pass / fail on regression thresholds
```

---

## How the judge gets judged

```mermaid
flowchart LR
  MH["MedHallu<br/>ships a correct AND a<br/>hallucinated answer<br/>per question"]
  AG["POST /chat<br/>per item, per agent"]
  JU["judge<br/>3rd model family, temp 0"]
  MC["meta_check<br/>accuracy · precision · recall<br/>F1 · Cohen's kappa"]
  RC["rule-based refusal<br/>classifier"]
  MH -- "question" --> AG --> JU
  MH -- "known labels" --> MC
  JU -- "same verdicts,<br/>scored against labels" --> MC
  JU -- "safety verdicts" --> X{"agreement?"}
  RC --> X
  X -- "65% on safety" --> MC
```

The labels are used **twice**: once to ask the agent a question, once to
grade the judge's verdict on the answer.

---

## Why safety is two numbers

```mermaid
flowchart TB
  P(["prompt"]) --> K{"actually harmful?"}
  K -- "harmful<br/>(JailbreakBench)" --> H{"response"}
  K -- "benign but sensitive<br/>(controls)" --> B{"response"}
  H -- "refused" --> HG["correct refusal"]
  H -- "complied" --> HB["<b>attack success</b><br/>failure #1"]
  B -- "answered" --> BG["correct answer"]
  B -- "refused" --> BB["<b>over-refusal</b><br/>failure #2"]
  HB -.-> WHY["averaging these hides<br/>an agent that refuses<br/>everything: perfect on #1,<br/>fails #2"]
  BB -.-> WHY
```

The benign controls are the part that is easy to omit and expensive to:
without them, the safest-looking agent is the most useless one.
