"""The two required tools: lookup_kb (over the wellness knowledge base) and
search_web (DuckDuckGo, no API key - see plan.md Decision on web search).

Both the KB collection and the embedding function are passed in rather than
imported globally, so this module can be unit-tested with fakes.
"""
from __future__ import annotations
from typing import Callable


def lookup_kb(coll, embed_fn: Callable[[list[str]], list[list[float]]], query: str, k: int = 4) -> list[dict]:
    q_emb = embed_fn([query])
    res = coll.query(query_embeddings=q_emb, n_results=k)
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[0.0] * len(docs)])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "source": meta.get("source"), "score": dist})
    return out


def _ddgs_class():
    """`duckduckgo-search` was frozen and renamed to `ddgs`; the old package
    still imports but silently returns 0 results. Prefer the maintained
    package and fall back to the legacy one so this works on either install.
    """
    try:
        from ddgs import DDGS  # maintained successor
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # legacy, frozen
        return DDGS


def search_web(query: str, max_results: int = 4) -> list[dict]:
    DDGS = _ddgs_class()
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {
            # `ddgs` renamed these keys; accept both spellings.
            "title": r.get("title"),
            "snippet": r.get("body") or r.get("description"),
            "url": r.get("href") or r.get("url") or r.get("link"),
        }
        for r in results
    ]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_kb",
            "description": "Search the internal wellness knowledge base (diet, exercise, meditation, habits, retreats, natural eating, supplements, nature/wellbeing) for relevant, vetted information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the knowledge base"},
                    "k": {"type": "integer", "description": "Number of results to return", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for information not covered by the wellness knowledge base, or that is time-sensitive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for on the web"},
                },
                "required": ["query"],
            },
        },
    },
]
