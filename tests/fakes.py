"""Fakes standing in for the network-dependent pieces this sandbox can't
reach (Groq, Gemini, Hugging Face model downloads, DuckDuckGo) - see
tests/README.md for exactly what this does and doesn't prove.
"""
from __future__ import annotations

import hashlib


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic, dependency-free stand-in for sentence-transformers.
    Turns each text into an 8-dim vector from a hash of its words, so
    near-duplicate/related texts land closer together than unrelated ones
    - good enough to exercise Chroma's add/query plumbing without a real
    model download.
    """
    vectors = []
    for text in texts:
        words = text.lower().split()
        vec = [0.0] * 8
        for w in words:
            h = int(hashlib.md5(w.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % 8] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class FakeToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = FakeToolCallFunction(name, arguments)


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


def make_scripted_completion_fn(script: list[FakeMessage]):
    """Returns a `completion_fn(model=..., messages=..., tools=..., api_key=...)`
    that plays back `script` in order, one FakeMessage per call - simulating
    a real litellm.completion() response without any network call.
    """
    calls = {"n": 0}

    def _completion_fn(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        return FakeResponse(script[i])

    _completion_fn.call_count = lambda: calls["n"]
    return _completion_fn
