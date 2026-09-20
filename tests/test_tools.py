import chromadb

from agents.tools import TOOL_SCHEMAS, lookup_kb
from kb.ingest import build_kb
from tests.fakes import fake_embed_fn


def test_lookup_kb_returns_ranked_results():
    client = chromadb.Client()
    _, coll = build_kb(embed_fn=fake_embed_fn, chroma_client=client)
    results = lookup_kb(coll, fake_embed_fn, "meditation breathing exercises", k=3)
    assert len(results) == 3
    for r in results:
        assert "text" in r and "source" in r and "score" in r


def test_tool_schemas_are_openai_style_function_specs():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"lookup_kb", "search_web"}
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert "query" in t["function"]["parameters"]["properties"]


# --------------------------------------------------------------------------
# web search: regression tests for the two bugs that corrupted a whole
# evaluation run - a POST that DuckDuckGo answers with a challenge page, and
# a parser kept in step with markup that had changed.
# --------------------------------------------------------------------------

LITE_HTML = """
<table>
<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=xx"
        class="result-link">Sleep hygiene basics</a></td></tr>
<tr><td class="result-snippet">Keeping a fixed wake time is the strongest single lever.</td></tr>
<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb&amp;rut=yy"
        class="result-link">Mindfulness &amp; stress</a></td></tr>
<tr><td class="result-snippet">Short daily sessions beat occasional long ones.</td></tr>
</table>
"""


def test_lite_parser_extracts_title_snippet_and_unwrapped_url(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_get", lambda url, query: LITE_HTML)

    out = tools._search_via_lite("sleep", 4)
    assert len(out) == 2
    assert out[0]["title"] == "Sleep hygiene basics"
    assert "fixed wake time" in out[0]["snippet"]
    # the /l/?uddg= wrapper must be unwrapped, not passed through
    assert out[0]["url"] == "https://example.com/a"
    # HTML entities in titles must be decoded
    assert out[1]["title"] == "Mindfulness & stress"


def test_search_uses_get_not_post(monkeypatch):
    """DuckDuckGo answers a POST from a non-browser client with a challenge
    page that parses to zero results. This asserts the method, because the
    failure it guards against is silent."""
    from agents import tools
    seen = {}

    class _Resp:
        text = LITE_HTML

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout,
                    ua=(headers or {}).get("User-Agent"))
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    # a POST would raise here, since only `get` is stubbed
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("web search must not use POST")))

    out = tools._search_via_lite("sleep hygiene", 2)
    assert out and seen["params"] == {"q": "sleep hygiene"}
    assert seen["ua"], "must send a browser User-Agent"
    assert seen["timeout"] <= 10, "search must fail fast inside a tool loop"


def test_search_web_reports_every_backend_failure(monkeypatch):
    from agents import tools

    def boom(name):
        def _f(query, max_results):
            raise RuntimeError(f"{name} exploded")
        return _f

    monkeypatch.setattr(tools, "_BACKENDS",
                        (("lite", boom("lite")), ("html", boom("html"))))
    try:
        tools.search_web("anything")
    except RuntimeError as e:
        # all causes survive, not just the last one
        assert "lite exploded" in str(e) and "html exploded" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_search_web_falls_back_to_the_next_backend(monkeypatch):
    from agents import tools
    good = [{"title": "t", "snippet": "s", "url": "https://example.com"}]
    monkeypatch.setattr(tools, "_BACKENDS", (
        ("lite", lambda q, n: []),                      # 0 results
        ("html", lambda q, n: (_ for _ in ()).throw(RuntimeError("nope"))),
        ("ddgs", lambda q, n: good),
    ))
    assert tools.search_web("x") == good
