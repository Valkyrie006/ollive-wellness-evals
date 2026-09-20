"""Exercises the ENTIRE tool-calling loop (agents/core.run_turn) - the "fixed
architecture" itself - with a scripted fake LLM standing in for Groq/Gemini.
This is the most important test in the suite: it proves the loop correctly
(1) calls a tool, (2) feeds the tool result back to the model, (3) returns
the model's final answer, and (4) updates short-term memory - all without
needing a real model API.
"""
import chromadb

import agents.core as core
from kb.ingest import build_kb
from tests.fakes import FakeMessage, FakeToolCall, fake_embed_fn, make_scripted_completion_fn


def setup_kb():
    client = chromadb.Client()
    _, coll = build_kb(embed_fn=fake_embed_fn, chroma_client=client)
    return coll


def test_run_turn_executes_a_tool_call_then_returns_final_answer():
    core.SESSIONS.clear()
    coll = setup_kb()

    tool_call = FakeToolCall("call_1", "lookup_kb", '{"query": "healthy diet basics", "k": 2}')
    script = [
        FakeMessage(content=None, tool_calls=[tool_call]),
        FakeMessage(content="Eat a balanced diet with whole foods.", tool_calls=None),
    ]
    completion_fn = make_scripted_completion_fn(script)

    result = core.run_turn(
        session_id="s1",
        user_message="How should I structure my diet?",
        model_config={"model": "fake/model"},
        api_key=None,
        coll=coll,
        embed_fn=fake_embed_fn,
        completion_fn=completion_fn,
    )

    assert result["response"] == "Eat a balanced diet with whole foods."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "lookup_kb"
    assert "result" in result["tool_calls"][0]
    assert completion_fn.call_count() == 2  # one call that requested the tool, one that answered


def test_run_turn_updates_short_term_memory_across_turns():
    core.SESSIONS.clear()
    coll = setup_kb()

    script1 = [FakeMessage(content="Nice to meet you!", tool_calls=None)]
    core.run_turn(
        session_id="s2", user_message="My name is Sam.",
        model_config={"model": "fake/model"}, api_key=None,
        coll=coll, embed_fn=fake_embed_fn,
        completion_fn=make_scripted_completion_fn(script1),
    )

    history = core.get_history("s2")
    assert len(history) == 2  # user turn + assistant turn
    assert history[0]["content"] == "My name is Sam."
    assert history[1]["content"] == "Nice to meet you!"


def test_run_turn_recovers_from_malformed_tool_call_json():
    """Smaller open models occasionally emit
    malformed tool-call JSON. Confirm a broken arguments string doesn't
    crash the loop - it should be caught and logged as a tool error, and
    the assistant should still produce a final answer.

    Note: `_safe_json_loads` does a light one-shot repair (trailing comma /
    missing closing brace), so a *simply* truncated JSON string actually
    gets recovered rather than erroring - that's intended behavior, not a
    bug (see the "recovered" test below). This test uses JSON that's broken
    in a way the repair can't fix, to exercise the actual error path.
    """
    core.SESSIONS.clear()
    coll = setup_kb()

    broken_tool_call = FakeToolCall("call_1", "lookup_kb", "{not: valid json ###")  # genuinely unparseable
    script = [
        FakeMessage(content=None, tool_calls=[broken_tool_call]),
        FakeMessage(content="Here's some general advice.", tool_calls=None),
    ]
    result = core.run_turn(
        session_id="s3", user_message="What should I eat?",
        model_config={"model": "fake/model"}, api_key=None,
        coll=coll, embed_fn=fake_embed_fn,
        completion_fn=make_scripted_completion_fn(script),
    )
    assert result["response"] == "Here's some general advice."
    assert "error" in result["tool_calls"][0]["result"]


def test_run_turn_repairs_a_lightly_truncated_tool_call_json():
    """The one-shot repair (missing closing brace) SHOULD succeed here -
    this is the companion case to the test above, confirming the repair
    path actually recovers a call instead of just always erroring.
    """
    core.SESSIONS.clear()
    coll = setup_kb()

    truncated_tool_call = FakeToolCall("call_1", "lookup_kb", '{"query": "diet"')  # missing closing brace
    script = [
        FakeMessage(content=None, tool_calls=[truncated_tool_call]),
        FakeMessage(content="Here's some general advice.", tool_calls=None),
    ]
    result = core.run_turn(
        session_id="s4", user_message="What should I eat?",
        model_config={"model": "fake/model"}, api_key=None,
        coll=coll, embed_fn=fake_embed_fn,
        completion_fn=make_scripted_completion_fn(script),
    )
    assert "error" not in result["tool_calls"][0]["result"]
    assert result["tool_calls"][0]["args"] == {"query": "diet"}


def test_transient_upstream_errors_are_retried_then_succeed(monkeypatch):
    """A 503 'high demand' from a free tier is retryable and must not surface
    as a failure. A permanent error (retired model ID) must NOT be retried -
    masking it behind three slow retries is how a config bug becomes a
    mystery.
    """
    monkeypatch.setattr(core.time, "sleep", lambda *_: None)  # no real backoff in tests
    core.SESSIONS.clear()
    coll = setup_kb()

    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("GeminiException - 503 This model is currently experiencing high demand.")
        return FakeMessage(content="Here's some advice.", tool_calls=None)

    # the scripted fake returns a response object; wrap it the same way
    def completion_fn(**kwargs):
        msg = flaky(**kwargs)
        return make_scripted_completion_fn([msg])(**kwargs)

    result = core.run_turn(
        session_id="retry-1", user_message="hi",
        model_config={"model": "fake/model"}, api_key=None,
        coll=coll, embed_fn=fake_embed_fn, completion_fn=completion_fn,
    )
    assert result["response"] == "Here's some advice."
    assert calls["n"] == 3  # two failures, then success


def test_permanent_upstream_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda *_: None)
    core.SESSIONS.clear()
    coll = setup_kb()

    calls = {"n": 0}

    def always_model_not_found(**kwargs):
        calls["n"] += 1
        raise RuntimeError('GroqException - {"code":"model_not_found"}')

    try:
        core.run_turn(
            session_id="retry-2", user_message="hi",
            model_config={"model": "fake/model"}, api_key=None,
            coll=coll, embed_fn=fake_embed_fn, completion_fn=always_model_not_found,
        )
        raise AssertionError("expected the permanent error to propagate")
    except RuntimeError as e:
        assert "model_not_found" in str(e)
    assert calls["n"] == 1  # raised immediately, no retries


def test_final_pass_nudges_a_tool_happy_model_into_answering():
    """A model that keeps choosing tools would otherwise exhaust every
    iteration and hand the user the "couldn't finish" fallback even though
    it had gathered good tool results.

    Observed live: gpt-oss-20b made three consecutive lookup_kb calls and
    the user got the fallback instead of an answer. The first fix withheld
    tools on the final pass, which Groq rejects outright with
    tool_use_failed if the model tries a call anyway - so the loop now
    nudges with an instruction instead, which every provider tolerates.
    """
    core.SESSIONS.clear()
    coll = setup_kb()

    seen_messages = []

    def greedy(**kwargs):
        seen_messages.append(kwargs["messages"])
        nudged = any(
            isinstance(m.get("content"), str) and "Do not call any more tools" in m["content"]
            for m in kwargs["messages"]
        )
        assert "tools" in kwargs, "tools must stay in every request - withholding them breaks Groq"
        if nudged:
            return make_scripted_completion_fn(
                [FakeMessage(content="Here is what I found.", tool_calls=None)]
            )(**kwargs)
        return make_scripted_completion_fn([
            FakeMessage(content=None,
                        tool_calls=[FakeToolCall(f"c{len(seen_messages)}", "lookup_kb",
                                                 '{"query": "habits"}')])
        ])(**kwargs)

    result = core.run_turn(
        session_id="greedy-1", user_message="What are good habits?",
        model_config={"model": "fake/model"}, api_key=None,
        coll=coll, embed_fn=fake_embed_fn, completion_fn=greedy,
    )

    assert result["response"] == "Here is what I found."
    # the nudge is a working-list message only - it must not leak into
    # the stored conversation the user sees next turn
    stored = core.get_history("greedy-1")
    assert not any("Do not call any more tools" in (m.get("content") or "") for m in stored)


def test_a_failing_tool_is_short_circuited_after_its_first_failure(monkeypatch):
    """Regression: a broken web-search backend once turned ~10s evaluation
    items into ~3-minute ones, because the model kept reaching for the dead
    tool and each attempt paid the full network timeout again. The tool must
    be called once; every later request in the same turn is answered from
    the recorded failure without touching the network."""
    core.SESSIONS.clear()
    coll = setup_kb()

    calls = {"n": 0}

    def exploding_search(**kwargs):
        calls["n"] += 1
        raise RuntimeError("web search failed (lite: timeout; html: timeout)")

    monkeypatch.setattr(core, "search_web", exploding_search)

    # a model that stubbornly asks for the same broken tool on every
    # iteration the loop allows it
    script = [
        FakeMessage(tool_calls=[FakeToolCall(f"c{i}", "search_web", '{"query": "a"}')])
        for i in range(core.MAX_TOOL_ITERATIONS)
    ] + [FakeMessage(content="Here's what I can tell you without the web.")]

    result = core.run_turn(
        session_id="cb1",
        user_message="any recent news on sleep research?",
        model_config={"model": "fake/model"},
        api_key=None,
        coll=coll,
        embed_fn=fake_embed_fn,
        completion_fn=make_scripted_completion_fn(script),
    )

    assert calls["n"] == 1, "the failing tool must be attempted only once per turn"
    assert result["response"] == "Here's what I can tell you without the web."
    # the model is told why, on every subsequent attempt
    later = [t for t in result["tool_calls"][1:]]
    assert later and all("already failed" in (t["result"].get("note") or "") for t in later)


def test_circuit_breaker_does_not_block_a_different_healthy_tool(monkeypatch):
    core.SESSIONS.clear()
    coll = setup_kb()

    monkeypatch.setattr(core, "search_web",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("no network")))

    script = [
        FakeMessage(tool_calls=[FakeToolCall("c1", "search_web", '{"query": "a"}')]),
        FakeMessage(tool_calls=[FakeToolCall("c2", "lookup_kb", '{"query": "sleep", "k": 2}')]),
        FakeMessage(content="Grounded in the knowledge base instead."),
    ]

    result = core.run_turn(
        session_id="cb2",
        user_message="how do I sleep better?",
        model_config={"model": "fake/model"},
        api_key=None,
        coll=coll,
        embed_fn=fake_embed_fn,
        completion_fn=make_scripted_completion_fn(script),
    )

    assert result["response"] == "Grounded in the knowledge base instead."
    kb_call = result["tool_calls"][1]
    assert kb_call["name"] == "lookup_kb"
    assert "error" not in kb_call["result"], "a healthy tool must be unaffected"
