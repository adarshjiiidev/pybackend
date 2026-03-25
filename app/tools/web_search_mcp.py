"""
Web Search MCP — Fast scraping layer that sits ABOVE existing search providers.

Adds DuckDuckGo + Google HTML scraping (1-3s) as Phase 1.
Falls back to existing OpenRouter and Groq Compound search in tool_executor.py.

Uses BeautifulSoup + lxml for reliable HTML parsing.
"""

import asyncio
import httpx
import logging
from typing import Optional
from urllib.parse import quote_plus, unquote

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Shared httpx client with short timeouts
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=3.0),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
    return _http_client


# ── DuckDuckGo HTML scrape (fastest, no API key, ~1-2s) ───────────────────

async def search_duckduckgo(query: str, num_results: int = 5) -> list[dict]:
    """Scrape DuckDuckGo HTML search with BeautifulSoup. Returns list of result dicts."""
    client = _get_http_client()
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    resp = await client.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    for result_div in soup.select(".result")[:num_results]:
        title_tag = result_div.select_one(".result__a")
        snippet_tag = result_div.select_one(".result__snippet")

        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = title_tag.get("href", "")
        if "uddg=" in href:
            href = unquote(href.split("uddg=")[-1].split("&")[0])
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        results.append({"title": title, "url": href, "snippet": snippet})

    return results


# ── Google HTML scrape (backup, ~2-3s) ────────────────────────────────────

async def search_google(query: str, num_results: int = 5) -> list[dict]:
    """Scrape Google search results with BeautifulSoup. Multiple selector fallbacks for robustness."""
    client = _get_http_client()
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&num={num_results}"

    resp = await client.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    # Try multiple container selectors — Google changes their markup often
    containers = (
        soup.select("div.g")
        or soup.select("div[data-sokoban-container]")
        or soup.select("div[jscontroller] h3")  # fallback: just find h3 headings
    )

    for g_div in containers[:num_results]:
        # If we got h3 tags directly, wrap them
        h3 = g_div if g_div.name == "h3" else g_div.select_one("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)
        if not title or len(title) < 3:
            continue

        # Find the href
        href = ""
        link_tag = h3.find_parent("a") or (h3 if h3.name == "a" else None)
        if link_tag:
            href = link_tag.get("href", "")
            if href.startswith("/url?q="):
                href = unquote(href.split("/url?q=")[-1].split("&")[0])

        # Extract snippet — try multiple CSS classes
        snippet = ""
        container = g_div if g_div.name != "h3" else h3.find_parent("div")
        if container:
            snippet_div = (
                container.select_one("[class*='VwiC3b']")
                or container.select_one("[class*='IsZvec']")
                or container.select_one("span[class]")
            )
            if snippet_div:
                snippet = snippet_div.get_text(strip=True)[:300]
            else:
                all_text = container.get_text(separator=" ", strip=True)
                snippet = all_text.replace(title, "", 1).strip()[:300]

        results.append({"title": title, "url": href, "snippet": snippet})

    # Last resort: extract any organic result links visible on the page
    if not results:
        for a_tag in soup.select("a[href*='http']")[:num_results * 2]:
            href = a_tag.get("href", "")
            if "/url?q=" in href:
                href = unquote(href.split("/url?q=")[-1].split("&")[0])
            if not href.startswith("http") or "google.com" in href:
                continue
            title = a_tag.get_text(strip=True)[:120]
            if len(title) > 5:
                results.append({"title": title, "url": href, "snippet": ""})
            if len(results) >= num_results:
                break

    return results


# ── Fast scrape search: DDG + Google raced ────────────────────────────────

async def fetch_page(url: str, max_chars: int = 6000) -> str:
    """
    Fetch a URL and extract clean readable text.
    Strips scripts, styles, nav, footer — extracts article/main/body.
    Truncated at max_chars. Timeout: connect=3s, read=5s.
    """
    client = _get_http_client()
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        # Prefer semantic content containers
        content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.find("body")
        )
        text = content.get_text(separator="\n", strip=True) if content else soup.get_text(separator="\n", strip=True)

        # Clean up blank lines
        import re
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:max_chars]
    except Exception as e:
        logger.debug(f"fetch_page failed for {url}: {e}")
        return ""


# ── Fast scrape search: DDG + Google raced ────────────────────────────────

async def fast_scrape_search(query: str, num_results: int = 5, fetch_top_page: bool = True) -> str:
    """
    Fully parallel: DDG + Google + page fetch all launch at time 0.
    No sequential waiting. Total latency ~1.5-2s.
    """
    async def _try_search(coro, name: str):
        try:
            results = await coro
            if results:
                logger.info(f"✅ MCP {name}: {len(results)} results for: {query[:50]}")
                return results
        except Exception as e:
            logger.debug(f"MCP {name} failed: {e}")
        return []

    async def _try_page(url: str):
        if not url or not url.startswith("http"):
            return ""
        try:
            return await asyncio.wait_for(fetch_page(url), timeout=4.0)
        except Exception:
            return ""

    # Launch DDG + Google simultaneously
    ddg_task = asyncio.create_task(_try_search(search_duckduckgo(query, num_results), "DDG"))
    google_task = asyncio.create_task(_try_search(search_google(query, num_results), "Google"))

    # Wait for first search result (max 5s)
    done, pending = await asyncio.wait(
        {ddg_task, google_task},
        timeout=5.0,
        return_when=asyncio.FIRST_COMPLETED,
    )

    results = []
    page_task = None

    for task in done:
        r = task.result()
        if r:
            results = r
            # Immediately kick off page fetch in parallel from first result
            if fetch_top_page:
                top_url = r[0].get("url", "")
                page_task = asyncio.create_task(_try_page(top_url))
            for p in pending:
                p.cancel()
            break

    # If first search failed, wait for the other
    if not results and pending:
        done2, pending2 = await asyncio.wait(pending, timeout=3.0)
        for task in done2:
            r = task.result()
            if r:
                results = r
                if fetch_top_page:
                    top_url = r[0].get("url", "")
                    page_task = asyncio.create_task(_try_page(top_url))
                for p in pending2:
                    p.cancel()
                break
        for p in pending2:
            p.cancel()

    if not results:
        if page_task:
            page_task.cancel()
        return ""

    snippet_summary = " | ".join(
        f"{r['title']}: {r['snippet']}" for r in results if r.get("snippet")
    )

    # Collect page content (already running in background)
    page_content = ""
    if page_task:
        try:
            page_content = await page_task
            if page_content:
                logger.info(f"📄 Page content fetched ({len(page_content)} chars)")
        except Exception:
            pass

    if page_content:
        combined = f"Search Results:\n{snippet_summary}\n\nTop Article Content:\n{page_content}"
        return combined[:6000]

    return snippet_summary[:3000] if snippet_summary else ""


async def close():
    """Cleanup HTTP client."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
