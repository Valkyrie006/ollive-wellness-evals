# What these tests actually verify

Built and run in a sandboxed cloud environment whose network is locked to
package registries only (pypi/npm) - no access to Groq, Google AI, Hugging
Face, or DuckDuckGo. That's a property of this build environment, not of the
code: on your own machine, with the two free API keys from `README.md`, the
real network calls work normally.

**Verified here (14/14 passing, real code paths, no shortcuts):**
- `test_kb_ingest.py` - the real 9 provided `.md` files are read and chunked
  correctly (9 sources, ~30-90 paragraph chunks); chunks load into a real
  Chroma collection and are queryable (embeddings come from a small
  deterministic fake standing in for sentence-transformers, since the real
  model has to download from huggingface.co).
- `test_tools.py` - `lookup_kb` returns correctly ranked/shaped results
  against the real Chroma collection; the tool schemas are valid
  OpenAI-style function specs.
- `test_agent_core.py` - the entire tool-calling loop (`agents/core.run_turn`,
  the actual "fixed architecture") end-to-end: it calls a tool, feeds the
  tool's result back to the model, returns the final answer, and updates
  short-term memory across turns - all against a scripted fake LLM in place
  of Groq/Gemini. Also proves the malformed-tool-JSON retry logic both (a)
  recovers a lightly-truncated call and (b) safely errors out on genuinely
  broken JSON without crashing the loop.
- `test_api.py` - the real FastAPI app, real routes, real request/response
  schema, correct 400 on an unknown agent name - with the KB/embedder faked
  and the LLM call monkeypatched.
- `test_evals.py` - the refusal/compliance classifier's logic on both
  harmful and legitimate-sensitive prompts, and the scorecard aggregation
  math (mean scores + attack-success-rate).

**Confirmed but NOT fixable from here - needs a real run:**
Actually starting `uvicorn api.main:app` for real in this sandbox fails at
startup, and only for one reason: `sentence-transformers` needs to download
`all-MiniLM-L6-v2` from `huggingface.co`, which this sandbox's network
policy returns `403` for (confirmed directly - see the session transcript).
Everything downstream of that (real Groq tool-calling, real Gemini calls,
real DuckDuckGo search, the real eval judge, the real HF dataset downloads
in `evals/datasets/prepare.py`) needs actual internet access and the two
free API keys, and has not been exercised against the real services yet.
