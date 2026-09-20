import chromadb

from agents.tools import lookup_kb, TOOL_SCHEMAS
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
