"""The two required tools: lookup_kb (over the wellness knowledge base) and
search_web (DuckDuckGo, no API key - see docs/DESIGN.md on web search).

Both the KB collection and the embedding function are passed in rather than
imported globally, so this module can be unit-tested with fakes.
"""
from __future__ import annotations

import html as html_lib
import logging
import re
from collections.abc import Callable
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger("wellness.tools")

# Short on purpose. Web search sits inside a tool-calling loop, so a slow
# failure is multiplied by the number of iterations and by every item in an
# eval run. Failing in seconds and telling the model so is far better than
# hanging: a 20s timeout per backend once cost this project ~3 minutes per
# evaluation item and quietly corrupted the latency table.
SEARCH_TIMEOUT = 8

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}


def lookup_kb(coll, embed_fn: Callable[[list[str]], list[list[float]]], query: str, k: int = 4) -> list[dict]:
    q_emb = embed_fn([query])
    res = coll.query(query_embeddings=q_emb, n_results=k)
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[0.0] * len(docs)])[0]
    # Plain zip, not zip(strict=...): that keyword is Python 3.10+ and this
    # has to run on 3.9 too. Chroma returns these three lists at equal
    # length for a single query, so there is nothing for strict= to catch.
    for doc, meta, dist in zip(docs, metas, dists):  # noqa: B905
        out.append({"text": doc, "source": meta.get("source"), "score": dist})
    return out


# --------------------------------------------------------------------------
# web search
# --------------------------------------------------------------------------

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
    with DDGS(timeout=SEARCH_TIMEOUT) as ddgs:
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
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps outbound links as //duckduckgo.com/l/?uddg=<target>."""
    if "uddg=" not in href:
        return href if href.startswith("http") else "https:" + href
    qs = parse_qs(urlparse(href).query)
    target = qs.get("uddg", [""])[0]
    return unquote(target) or href


def _get(url: str, query: str):
    """GET, never POST.

    DuckDuckGo answers a POST from a non-browser client with a challenge
    page that parses to zero results - which is how this failed silently for
    a whole evaluation run: `/diagnostics` said "returned 0 results" while
    the same URL opened fine in a browser. A browser issues a GET, so this
    does too, with the headers a browser sends.
    """
    import requests

    resp = requests.get(url, params={"q": query}, headers=_BROWSER_HEADERS,
                        timeout=SEARCH_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _search_via_lite(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo's `lite` endpoint - the most stable target of the three.

    Its markup is a flat table: one `<a class="result-link">` per result and
    one `<td class="result-snippet">` per result, in the same order. No
    nested result containers to keep a regex in step with.
    """
    text = _get("https://lite.duckduckgo.com/lite/", query)
    links = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                       text, re.S)
    if not links:
        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
                           text, re.S)
    snippets = re.findall(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', text, re.S)

    out = []
    for i, (href, title) in enumerate(links[:max_results]):
        out.append({
            "title": _strip_tags(title),
            "snippet": _strip_tags(snippets[i]) if i < len(snippets) else "",
            "url": _unwrap_redirect(href),
        })
    return out


def _search_via_html(query: str, max_results: int) -> list[dict]:
    """The no-JS `html` endpoint, parsed with stdlib regex over `requests`.

    Kept as a second scraping target because the two endpoints have failed
    independently: `lite` has been rate-limited while `html` answered, and
    the reverse. Neither needs an API key, which is the constraint that
    rules out Brave/Serper/Tavily here.
    """
    text = _get("https://html.duckduckgo.com/html/", query)
    out = []
    for block in re.split(r'<div class="result[ _]', text)[1:]:
        link = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            link = re.search(r'href="([^"]+)"[^>]*class="result__a"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        if not snippet:
            snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</td>', block, re.S)
        out.append({
            "title": _strip_tags(link.group(2)),
            "snippet": _strip_tags(snippet.group(1)) if snippet else "",
            "url": _unwrap_redirect(link.group(1)),
        })
        if len(out) >= max_results:
            break
    return out


# Ordered by observed reliability, not by elegance. `lite` is first because
# it is the one that actually works on a stock machine: `ddgs` depends on
# the `primp` TLS stack, which raises "Unsupported protocol version 0x304"
# on some hosts - an environment fault with no in-code fix.
_BACKENDS = (
    ("lite", _search_via_lite),
    ("html", _search_via_html),
    ("ddgs", _search_via_library),
)


def search_web(query: str, max_results: int = 4) -> list[dict]:
    """Try each backend in order, return the first that yields results.

    Every backend failing is reported as one error carrying all the causes -
    debugging a fallback chain is miserable when only the last failure
    survives.
    """
    errors = []
    for name, fn in _BACKENDS:
        try:
            results = fn(query, max_results)
            if results:
                if errors:
                    logger.info("web search: %s succeeded after %s", name, "; ".join(errors))
                return results
            errors.append(f"{name}: returned 0 results")
        except Exception as e:  # noqa: BLE001 - reported back to the model as a tool error
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
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
