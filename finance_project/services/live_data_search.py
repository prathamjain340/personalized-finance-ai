import html
import os
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


_WEB_SEARCH_ENABLED = os.getenv("LIVE_DATA_WEB_SEARCH_ENABLED", "true").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_WEB_SEARCH_TIMEOUT_SECONDS = float(os.getenv("LIVE_DATA_WEB_SEARCH_TIMEOUT_SECONDS", "6"))
_WEB_SEARCH_MAX_RESULTS = int(os.getenv("LIVE_DATA_WEB_SEARCH_MAX_RESULTS", "3"))


def search_web(query: str, max_results: int | None = None) -> list[dict[str, Any]]:
    text = str(query or "").strip()
    if not _WEB_SEARCH_ENABLED or not text:
        return []

    limit = max(1, min(8, int(max_results or _WEB_SEARCH_MAX_RESULTS)))
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": text},
            headers=headers,
            timeout=_WEB_SEARCH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except Exception:
        return []

    body = resp.text or ""
    anchor_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>.*?</a>[\s\S]{0,300}?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    anchors = anchor_pattern.findall(body)
    snippets = snippet_pattern.findall(body)
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for idx, (raw_href, raw_title) in enumerate(anchors):
        if len(results) >= limit:
            break

        url = _normalize_search_url(raw_href)
        if not url or url in seen_urls:
            continue

        title = _clean_html_text(raw_title)
        if not title:
            continue

        snippet = _clean_html_text(snippets[idx]) if idx < len(snippets) else ""
        seen_urls.add(url)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    return results


def _normalize_search_url(raw_href: str) -> str:
    href = html.unescape(str(raw_href or "").strip())
    if not href:
        return ""

    if href.startswith("/l/?"):
        parsed = urlparse(href)
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return unquote(target).strip()
        return ""

    if href.startswith("http://") or href.startswith("https://"):
        return href

    return ""


def _clean_html_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
