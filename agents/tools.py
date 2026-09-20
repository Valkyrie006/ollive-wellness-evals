"""The two required tools: lookup_kb (over the wellness knowledge base) and
search_web (DuckDuckGo, no API key - see plan.md Decision on web search).

Both the KB collection and the embedding function are passed in rather than
imported globally, so this module can be unit-tested with fakes.
"""
from __future__ import annotations

from collections.abc import Callable


def lookup_kb(coll, embed_fn: Callable[[list[str]], list[list[float]]], query: str, k: int = 4) -> list[dict]:
    q_emb = embed_fn([query])
    res = coll.query(query_embeddings=q_emb, n_results=k)
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[0.0] * len(docs)])[0]
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
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


def _search_via_library(query: str, max_results: int) -> list[dict]:
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


def _strip_tags(fragment: str) -> str:
    import html as html_lib
    import re
    return html_lib.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps outbound links as /l/?uddg=<encoded-target>."""
    from urllib.parse import parse_qs, unquote, urlparse
    if "uddg=" not in href:
        return href
    qs = parse_qs(urlparse(href).query)
    target = qs.get("uddg", [""])[0]
    return unquote(target) or href


def _search_via_html(query: str, max_results: int) -> list[dict]:
    """Dependency-light fallback: DuckDuckGo's no-JS HTML endpoint, parsed
    with stdlib regex over `requests`.

    Exists because `ddgs` relies on the `primp` TLS stack, which fails on
    some machines with `ValueError: Unsupported protocol version 0x304` -
    an environment problem with no in-code fix. This path needs only
    `requests`, so web search still works wherever that installs.
    """
    import re

    import requests

    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=20,
    )
    resp.raise_for_status()

    out = []
    for block in re.split(r'<div class="result[ _]', resp.text)[1:]:
        link = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        out.append({
            "title": _strip_tags(link.group(2)),
            "snippet": _strip_tags(snippet.group(1)) if snippet else "",
            "url": _unwrap_redirect(link.group(1)),
        })
        if len(out) >= max_results:
            break
    return out


def search_web(query: str, max_results: int = 4) -> list[dict]:
    """Try the maintained library first, fall back to the HTML endpoint.

    Both failing is reported as one error carrying both causes - debugging a
    fallback chain is miserable when only the last failure survives.
    """
    errors = []
    for name, fn in (("ddgs", _search_via_library), ("html", _search_via_html)):
        try:
            results = fn(query, max_results)
            if results:
                return results
            errors.append(f"{name}: returned 0 results")
        except Exception as e:  # noqa: BLE001 - reported back to the model as a tool error
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("web search failed (" + "; ".join(errors) + ")")


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
