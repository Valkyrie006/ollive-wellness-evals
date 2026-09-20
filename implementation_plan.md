# Implementation Plan — Wellness Assistant + Evals Platform (1-Day POC)

Companion to `plan.md` (the why/what). This is the how: file-by-file build order, with acceptance checks per step so you know when to move on. Follow top to bottom — each step assumes the previous ones are done.

## Repo layout (create this first)

```
ollive-wellness-evals/
├── .env.example
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── core.py            # the tool-calling loop — THE fixed architecture
│   ├── tools.py            # lookup_kb, search_web
│   ├── prompts.py          # shared system prompt
│   └── config.py           # model configs for "oss" and "frontier"
├── kb/
│   ├── source/              # your 9 provided .md files go here
│   └── ingest.py            # chunk + embed + load into Chroma
├── api/
│   └── main.py              # FastAPI app
├── ui/
│   └── index.html            # minimal chat page
├── evals/
│   ├── datasets/
│   │   ├── prepare.py        # pulls + samples HF datasets into jsonl
│   │   ├── hallucination.jsonl
│   │   ├── bias.jsonl
│   │   └── safety.jsonl
│   ├── judge.py               # DeepEval custom model wrapper (Groq gpt-oss-20b)
│   ├── refusal_check.py       # regex refusal/compliance detector
│   ├── runner.py               # runs both agents against all 3 datasets, scores them
│   └── meta_check.py            # judge-quality mini check
├── results/
│   ├── scorecard.json
│   └── chart.png
├── README.md
└── evaluation_report.md
```

Acceptance check: `tree ollive-wellness-evals` matches the above (empty files are fine at this point).

---

## Step 0 — Environment (≈30 min)

1. Sign up at console.groq.com → create an API key → `GROQ_API_KEY`.
2. Sign up at aistudio.google.com → create an API key → `GOOGLE_API_KEY`.
3. `.env.example`:
   ```
   GROQ_API_KEY=
   GOOGLE_API_KEY=
   ```
4. `requirements.txt`:
   ```
   litellm
   sentence-transformers
   chromadb
   duckduckgo-search
   fastapi
   uvicorn
   deepeval
   datasets
   python-dotenv
   matplotlib
   ```
5. `pip install -r requirements.txt`.

**Check:** `python -c "import litellm, chromadb, sentence_transformers, fastapi, deepeval"` runs with no errors.

---

## Step 1 — Knowledge base ingestion (`kb/ingest.py`) (≈45 min)

Your 9 files (`01_Diet.md` … `09_Nature_General_Welfare.md`, ~470–550 words each) are small — no heavy pipeline needed.

```python
# kb/ingest.py
import glob, re
import chromadb
from sentence_transformers import SentenceTransformer

def load_chunks(source_dir="kb/source"):
    chunks = []
    for path in sorted(glob.glob(f"{source_dir}/*.md")):
        text = open(path).read()
        # split on paragraph breaks (blank lines) — files are short, this gives
        # ~3-5 chunks per file, ~30-40 chunks total
        for i, para in enumerate(p.strip() for p in text.split("\n\n") if p.strip()):
            chunks.append({"id": f"{path}-{i}", "text": para, "source": path})
    return chunks

def build_kb(source_dir="kb/source", collection_name="wellness_kb"):
    chunks = load_chunks(source_dir)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks]).tolist()
    client = chromadb.Client()  # in-memory, per Decision #6
    coll = client.get_or_create_collection(collection_name)
    coll.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return client, coll
```

**Check:** `python -c "from kb.ingest import build_kb; c,coll=build_kb(); print(coll.count())"` prints ~30-40.

---

## Step 2 — Tools (`agents/tools.py`) (≈30 min)

```python
# agents/tools.py
from duckduckgo_search import DDGS

def lookup_kb(coll, embed_model, query: str, k: int = 4):
    q_emb = embed_model.encode([query]).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "score": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

def search_web(query: str, max_results: int = 4):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_kb",
        "description": "Search the internal wellness knowledge base for relevant, vetted information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer", "default": 4}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the live web for information not covered by the knowledge base.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]
```

**Check:** call both functions directly with a sample query and confirm non-empty results.

---

## Step 3 — Shared system prompt (`agents/prompts.py`) (≈15 min)

Write ONE prompt used by both agents, unchanged. It should:
- Establish the wellness-assistant persona.
- Instruct the model to prefer `lookup_kb` for anything the KB might cover, and `search_web` only when the KB doesn't have it.
- Require an explicit disclaimer whenever the answer touches diagnosis, medication, or dosage ("not a substitute for professional medical advice").
- Instruct honest refusal-to-fabricate: "if you don't know and can't find it via a tool, say so — never invent a fact."

**Check:** read it aloud — does it read as ONE prompt any model could follow, with no OSS-specific or frontier-specific wording?

---

## Step 4 — Core tool-calling loop (`agents/core.py`) (≈1 hr) — THE fixed architecture

```python
# agents/core.py
import litellm
from agents.tools import lookup_kb, search_web, TOOL_SCHEMAS
from agents.prompts import SYSTEM_PROMPT

SESSIONS = {}  # session_id -> list[messages], in-process memory (Decision #6)
MAX_TOOL_ITERATIONS = 2

def get_history(session_id, window=6):
    return SESSIONS.setdefault(session_id, [])[-window:]

def run_turn(session_id, user_message, model_config, coll, embed_model):
    history = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + \
               [{"role": "user", "content": user_message}]
    tool_calls_log = []

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        resp = litellm.completion(model=model_config["model"], messages=messages,
                                   tools=TOOL_SCHEMAS, api_key=model_config["api_key"])
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            messages.append({"role": "assistant", "content": msg.content})
            break
        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            try:
                args = json_loads_safe(tc.function.arguments)  # try/except + retry-once, per Decision #3
                if tc.function.name == "lookup_kb":
                    result = lookup_kb(coll, embed_model, **args)
                else:
                    result = search_web(**args)
            except Exception as e:
                result = {"error": str(e)}
            tool_calls_log.append({"name": tc.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    SESSIONS[session_id] = (history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": messages[-1]["content"]},
    ])[-6:]
    return {"response": messages[-1]["content"], "tool_calls": tool_calls_log}
```

`agents/config.py`:
```python
OSS_CONFIG = {"model": "groq/llama-3.1-8b-instant", "api_key": "<GROQ_API_KEY>"}
FRONTIER_CONFIG = {"model": "gemini/gemini-2.0-flash", "api_key": "<GOOGLE_API_KEY>"}
JUDGE_CONFIG = {"model": "groq/gpt-oss-20b", "api_key": "<GROQ_API_KEY>"}
```

**Check:** call `run_turn("test", "What's a good way to start meditating?", OSS_CONFIG, coll, embed_model)` and confirm it returns a response and, on a follow-up question, that memory carries over. Repeat with `FRONTIER_CONFIG` — same code path, only the config differs (this IS the "fixed architecture" requirement — verify by diffing what changed: only the config object).

---

## Step 5 — API + UI (≈1 hr)

`api/main.py`: one FastAPI app, `POST /chat` with body `{session_id, agent: "oss"|"frontier", message}`, dispatches to `run_turn` with the matching config, times the call, returns `{response, tool_calls, latency_ms}`. Load the Chroma collection and embed model once at startup, not per-request.

`ui/index.html`: a single page — a text input, a send button, a message list, a toggle for oss/frontier — `fetch("/chat", {...})`. No framework, no build step.

**Check:** `uvicorn api.main:app --reload`, open `ui/index.html` in a browser (or serve it via FastAPI's `StaticFiles`), have a real multi-turn conversation with both agents, confirm both tools fire at least once each across your test conversation.

---

## Step 6 — Eval datasets (`evals/datasets/prepare.py`) (≈45 min)

```python
from datasets import load_dataset
import json, random

random.seed(0)

def sample_and_save(dataset_id, split, n, out_path, transform):
    ds = load_dataset(dataset_id, split=split)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    with open(out_path, "w") as f:
        for i in idx:
            f.write(json.dumps(transform(ds[i])) + "\n")

sample_and_save("UTAustin-AIHealth/MedHallu", "train", 15, "evals/datasets/hallucination.jsonl",
                 lambda r: {"question": r["Question"], "context": r["Knowledge"], "ground_truth": r["Ground Truth"]})
sample_and_save("walledai/BBQ", "test", 15, "evals/datasets/bias.jsonl",
                 lambda r: {"question": r["question"], "category": r.get("category")})
sample_and_save("walledai/JailbreakBench", "train", 15, "evals/datasets/safety.jsonl",
                 lambda r: {"prompt": r.get("prompt") or r.get("Behavior")})
```

Note: verify each dataset's exact column names against its Hugging Face dataset card at run time — HF dataset schemas do change — adjust the `transform` lambdas accordingly if a field name differs.

**Check:** three JSONL files exist, each with ~15 lines.

---

## Step 7 — Judge wrapper (`evals/judge.py`) (≈30 min)

DeepEval's built-in metrics default to a paid OpenAI judge — override with Groq's `gpt-oss-20b` (Decision #5):

```python
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="groq/gpt-oss-20b", api_key=None):
        self.model, self.api_key = model, api_key
    def load_model(self): return self
    def generate(self, prompt: str) -> str:
        r = litellm.completion(model=self.model, api_key=self.api_key,
                                messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return "groq-gpt-oss-20b"
```

**Check:** `GroqJudge(api_key=...).generate("Say hi")` returns text.

---

## Step 8 — Eval runner (`evals/runner.py`) (≈1 hr)

For each of the 3 axes: send every sampled item's question/prompt to both agents' `/chat` endpoint (fresh `session_id` per item, single turn), log `{agent, axis, item, response, tool_calls, latency_ms}` to `results/raw.jsonl`, then score:

- **Hallucination:** `HallucinationMetric(model=GroqJudge(...))` with `LLMTestCase(input=question, actual_output=response, context=[ground_truth or top KB chunk])`.
- **Bias:** `BiasMetric(model=GroqJudge(...))` on the same `LLMTestCase` shape (context optional).
- **Safety:** run `evals/refusal_check.py`'s regex/keyword classifier (unsafe-compliance / appropriate-refusal / over-refusal) AND `ToxicityMetric(model=GroqJudge(...))` as a second signal; report both.

Aggregate mean scores per `{agent, axis}` into `results/scorecard.json`. Build `results/chart.png` with matplotlib: grouped bar chart, one group per axis, one bar per agent.

**Check:** `results/scorecard.json` has 2 agents × 3 axes = 6 entries; `chart.png` renders and is legible.

---

## Step 9 — Judge meta-check (`evals/meta_check.py`) (≈30 min)

1. Sample 5 items per axis (15 total) already used above.
2. Hand-label each with your own verdict (hallucinated/not, biased/not, unsafe/not) into `evals/gold_labels.jsonl`.
3. Compare the judge's verdict on those same 15 items to your labels; compute `% agreement = matches / 15`.
4. Print/save the agreement % — this goes straight into the evaluation report.

**Check:** a single number (or 3 numbers, one per axis) you can quote in the report.

---

## Step 10 — README + 1-page evaluation report (≈45 min)

`README.md` must cover: setup instructions (env vars, `pip install`, how to run the API + UI + eval runner), architecture decisions (link/summarize `plan.md`'s Section 2), trade-offs made (Section 7's accepted limitations), and what you'd improve with more time (larger datasets, multi-labeler kappa, ensemble judge, guardrails, public deploy).

`evaluation_report.md`: one page — lead sentence with the headline result, `chart.png` embedded, the judge-agreement % from Step 9, and a one-paragraph recommendation (which assistant to ship, and why, given the three axes).

**Check:** a person who has never seen this project can read `README.md` and get the OSS assistant + frontier assistant both running from scratch.

---

## Step 11 — Stretch (only if time remains, in this order)

1. Latency table: you already log `latency_ms` per call in Step 8 — just aggregate p50/p95 per agent into a small table in the report.
2. Guardrails: pick the worst-scoring axis from Step 8, add one regex/keyword filter (input or output side) in `agents/core.py`, re-run Step 8's runner, show the before/after delta.
3. Public OSS deploy: wrap the Groq call in a small Hugging Face Space (Gradio) so there's a shareable link.
4. Demo: record a 2–3 minute Loom walking through both agents and the eval report.

---

## Known risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Groq/Gemini free-tier rate limits throttle the ~90-call eval run (2 agents × 3 axes × 15 items) | Add a small `time.sleep(1-2)` between calls in the runner; run axes sequentially, not in parallel |
| Tool-call JSON malformed on either model | try/except + retry-once around `json.loads(tc.function.arguments)` in `agents/core.py` |
| HF dataset column names differ from what's assumed in Step 6 | Check the dataset card on huggingface.co before writing the `transform` lambda; adjust field names |
| DeepEval metric fails silently or errors on a short/empty response | Wrap each metric call in try/except, log failures separately, don't let one bad item kill the whole run |
