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
from tests.fakes import fake_embed_fn, FakeMessage, FakeToolCall, make_scripted_completion_fn


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
    """Per plan.md Decision #3: smaller OSS models occasionally emit
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
