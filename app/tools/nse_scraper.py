"""
Fast NSE data scraper — Google/DuckDuckGo search primary, NSE API fallback.
Uses BeautifulSoup + lxml for reliable HTML parsing.
"""

import httpx
import re
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .nse_cache import get_nse_cache

logger = logging.getLogger(__name__)

# ── yfinance ticker map for Indian indices and stocks ─────────────────────
_YF_TICKER_MAP: Dict[str, str] = {
    "NIFTY":     "^NSEI",
    "NIFTY50":   "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX":    "^BSESN",
    "FINNIFTY":  "NIFTY_FIN_SERVICE.NS",
    "MIDCAP":    "^NSMIDCP",
}


def _fetch_yfinance(symbol: str) -> Dict[str, Any]:
    """Synchronous yfinance fetch — run in executor to avoid blocking event loop."""
    import yfinance as yf

    symbol_up = symbol.strip().upper()
    # Map to yfinance ticker; NSE stocks get .NS suffix
    ticker_sym = _YF_TICKER_MAP.get(symbol_up, f"{symbol_up}.NS")

    tk = yf.Ticker(ticker_sym)
    info = tk.fast_info  # fast_info is much quicker than .info

    ltp = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
    prev_close = getattr(info, "previous_close", None)
    open_price = getattr(info, "open", None)
    day_high = getattr(info, "day_high", None)
    day_low = getattr(info, "day_low", None)
    volume = getattr(info, "last_volume", None)

    if ltp is None:
        raise ValueError(f"yfinance returned no price for {ticker_sym}")

    change = round(ltp - prev_close, 2) if prev_close else None
    change_pct = round((change / prev_close) * 100, 2) if (change and prev_close) else None

    return {
        "symbol": symbol_up,
        "ltp": round(ltp, 2),
        "change": change,
        "changePercent": change_pct,
        "open": open_price,
        "high": day_high,
        "low": day_low,
        "prevClose": prev_close,
        "volume": volume,
        "timestamp": datetime.now().isoformat(),
        "source": "yfinance",
    }


class NSEFastScraper:
    """Fast NSE data scraper using official hidden APIs."""
    
    BASE_URL = "https://www.nseindia.com"
    
    # NSE requires specific headers to prevent blocking
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",  # Removed 'br' as brotli library is not installed
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.last_cookie_refresh = None
        self.lock = asyncio.Lock()
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or refresh NSE session client."""
        async with self.lock:
            # Initialize or refresh client every 5 minutes to keep session alive
            if self.client is None or self.client.is_closed or (
                self.last_cookie_refresh and
                datetime.now() - self.last_cookie_refresh > timedelta(minutes=5)
            ):
                if self.client and not self.client.is_closed:
                    await self.client.aclose()

                self.client = httpx.AsyncClient(
                    headers=self.HEADERS,
                    timeout=10.0,
                    follow_redirects=True
                )
                # Visit homepage to get cookies
                await self.client.get(self.BASE_URL)
                self.last_cookie_refresh = datetime.now()
                logger.info("NSE session and connection pool initialized")

            return self.client
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get live stock quote data.
        Priority: cache → yfinance → Google/DDG scrape → NSE API fallback.
        Speed: ~0.5-2s.
        """
        # Check cache first
        cache = get_nse_cache()
        cached_data = cache.get("quote", symbol)
        if cached_data:
            return cached_data

        symbol_up = symbol.strip().upper()

        # Phase 1: yfinance (fast, reliable, handles NSE stocks + indices)
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _fetch_yfinance, symbol_up),
                timeout=5.0
            )
            if result.get("ltp") is not None:
                cache.set("quote", result, symbol_up)
                logger.info(f"✅ yfinance price for {symbol_up}: ₹{result['ltp']}")
                return result
        except Exception as e:
            logger.debug(f"yfinance failed for {symbol_up}: {e}")

        # Phase 2: Google search scrape (BeautifulSoup)
        try:
            result = await fetch_google_price(symbol_up)
            if result.get("ltp") is not None:
                cache.set("quote", result, symbol_up)
                return result
        except Exception as e:
            logger.debug(f"Google price scrape failed for {symbol_up}: {e}")

        # Phase 3: DuckDuckGo scrape
        try:
            result = await fetch_ddg_price(symbol_up)
            if result.get("ltp") is not None:
                cache.set("quote", result, symbol_up)
                return result
        except Exception as e:
            logger.debug(f"DDG price scrape failed for {symbol_up}: {e}")

        # Phase 4: NSE API (5s timeout, last resort)
        try:
            client = await self._get_session()
            url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol_up}"
            response = await asyncio.wait_for(client.get(url), timeout=5.0)
            response.raise_for_status()
            data = response.json()
            price_info = data.get("priceInfo", {})
            result = {
                "symbol": symbol_up,
                "ltp": price_info.get("lastPrice"),
                "change": price_info.get("change"),
                "changePercent": price_info.get("pChange"),
                "open": price_info.get("open"),
                "high": price_info.get("intraDayHighLow", {}).get("max"),
                "low": price_info.get("intraDayHighLow", {}).get("min"),
                "prevClose": price_info.get("previousClose"),
                "volume": data.get("preOpenMarket", {}).get("totalTradedVolume", 0),
                "timestamp": datetime.now().isoformat(),
                "source": "nse_api"
            }
            cache.set("quote", result, symbol_up)
            return result
        except Exception as e:
            logger.warning(f"All sources failed for {symbol_up}: {e}")
            return {"error": f"All sources failed for {symbol_up}", "symbol": symbol_up}
    
    async def get_fii_dii_data(self) -> Dict[str, Any]:
        """
        Get FII/DII participation data.
        Fast alternative to Puppeteer FII/DII scraper.
        
        Speed: ~1-2 seconds
        """
        try:
            # Check cache first
            cache = get_nse_cache()
            cached_data = cache.get("fii_dii")
            if cached_data:
                return cached_data

            client = await self._get_session()
            
            # NSE FII/DII API endpoint
            url = f"{self.BASE_URL}/api/fiidiiTradeReact"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            result = {
                "date": datetime.now().date().isoformat(),
                "fii": {"buy": 0, "sell": 0, "net": 0},
                "dii": {"buy": 0, "sell": 0, "net": 0},
                "timestamp": datetime.now().isoformat(),
                "source": "nse_api"
            }
            
            # Extract FII/DII data from response
            for entry in data:
                category = entry.get("category", "").lower()
                
                if "fii" in category or "fpi" in category:
                    result["fii"]["buy"] = float(entry.get("buyValue", 0))
                    result["fii"]["sell"] = float(entry.get("sellValue", 0))
                    result["fii"]["net"] = float(entry.get("netValue", 0))
                
                if "dii" in category:
                    result["dii"]["buy"] = float(entry.get("buyValue", 0))
                    result["dii"]["sell"] = float(entry.get("sellValue", 0))
                    result["dii"]["net"] = float(entry.get("netValue", 0))
            
            # Store in cache
            cache.set("fii_dii", result)
            return result
        except Exception as e:
            logger.error(f"NSE FII/DII fetch error: {e}")
            return {"error": str(e)}
    
    async def get_option_chain(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """
        Get option chain data.
        Fast alternative to Puppeteer option chain scraper.
        
        Speed: ~1-2 seconds
        """
        try:
            # Check cache first
            cache = get_nse_cache()
            cached_data = cache.get("option_chain", symbol)
            if cached_data:
                return cached_data

            client = await self._get_session()
            
            # Determine if index or equity
            if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                url = f"{self.BASE_URL}/api/option-chain-indices?symbol={symbol}"
            else:
                url = f"{self.BASE_URL}/api/option-chain-equities?symbol={symbol}"
            
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("records", {})
            
            result = {
                "symbol": symbol,
                "underlyingValue": records.get("underlyingValue"),
                "timestamp": records.get("timestamp"),
                "expiryDates": records.get("expiryDates", []),
                "strikePrices": records.get("strikePrices", [])[:10],  # First 10 strikes
                "data": records.get("data", [])[:20],  # First 20 rows for speed
                "filteredData": records.get("filteredData", [])[:20],
                "source": "nse_api"
            }

            # Store in cache
            cache.set("option_chain", result, symbol)
            return result
        except Exception as e:
            logger.error(f"NSE option chain error for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and indices."""
        try:
            # Check cache first
            cache = get_nse_cache()
            cached_data = cache.get("market_status")
            if cached_data:
                return cached_data

            client = await self._get_session()
            
            url = f"{self.BASE_URL}/api/marketStatus"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            result = {
                "marketState": data.get("marketState", []),
                "timestamp": datetime.now().isoformat(),
                "source": "nse_api"
            }

            # Store in cache
            cache.set("market_status", result)
            return result
        except Exception as e:
            logger.error(f"NSE market status error: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close the persistent httpx client."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.info("NSE scraper connection pool closed")


# Global instance
_nse_scraper: Optional[NSEFastScraper] = None


def get_nse_scraper() -> NSEFastScraper:
    """Get or create NSE scraper instance."""
    global _nse_scraper
    if _nse_scraper is None:
        _nse_scraper = NSEFastScraper()
    return _nse_scraper


# Convenience functions for tool executor
async def fetch_nse_quote(symbol: str) -> Dict[str, Any]:
    """Fetch NSE stock quote - FAST (0.5-1.5s)."""
    scraper = get_nse_scraper()
    return await scraper.get_stock_quote(symbol)


async def fetch_fii_dii() -> Dict[str, Any]:
    """Fetch FII/DII data - FAST (1-2s)."""
    scraper = get_nse_scraper()
    return await scraper.get_fii_dii_data()


async def fetch_option_chain(symbol: str = "NIFTY") -> Dict[str, Any]:
    """Fetch option chain - FAST (1-2s)."""
    scraper = get_nse_scraper()
    return await scraper.get_option_chain(symbol)


async def fetch_market_status() -> Dict[str, Any]:
    """Fetch market status."""
    scraper = get_nse_scraper()
    return await scraper.get_market_status()


# ───────────────────────────────────────────────────────────────────────────
#  Google + DuckDuckGo price scrapers (BeautifulSoup + lxml)
#  Primary method for getting live prices — much faster and more reliable
#  than NSE API which frequently times out.
# ───────────────────────────────────────────────────────────────────────────

# Map common user-facing terms to Google Finance ticker/search queries
_GOOGLE_QUERY_MAP: Dict[str, str] = {
    "NIFTY":     "nifty 50 index price today",
    "NIFTY50":   "nifty 50 index price today",
    "BANKNIFTY": "bank nifty index price today",
    "SENSEX":    "sensex bse index price today",
    "FINNIFTY":  "nifty financial services index price",
    "MIDCAP":    "nifty midcap 100 index price",
}

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def _parse_price_from_text(text: str) -> Optional[float]:
    """Extract a plausible Indian stock/index price from text."""
    # Match patterns like 22,400.50 or 1,234.5 or 45678.90
    matches = re.findall(r'[\d,]+\.\d{1,2}', text)
    for m in matches:
        try:
            val = float(m.replace(',', ''))
            # Plausible Indian stock/index price range
            if 1.0 < val < 200000:
                return val
        except ValueError:
            continue
    return None


def _parse_change_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract change and change% like +125.30 (+0.56%) or -320.10 (-1.42%)."""
    m = re.search(r'([+-]?[\d,]+\.?\d*)\s*\(([+-]?[\d\.]+)\s*%\)', text)
    if m:
        try:
            change = float(m.group(1).replace(',', ''))
            pct = float(m.group(2))
            return change, pct
        except ValueError:
            pass
    return None, None


async def fetch_google_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch real-time price from Google search using BeautifulSoup.
    Parses the knowledge panel / finance widget at the top of results.
    Speed: ~0.5-1.5s
    """
    symbol_up = symbol.strip().upper()
    query = _GOOGLE_QUERY_MAP.get(symbol_up, f"{symbol_up} NSE India stock price today")
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=en"

    cache = get_nse_cache()
    cached = cache.get("google_price", symbol_up)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(headers=_SEARCH_HEADERS, timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        html = resp.text
        price = None
        change = None
        change_pct = None

        # Strategy 1: data-last-price attribute (most reliable)
        tag = soup.find(attrs={"data-last-price": True})
        if tag:
            try:
                price = float(tag["data-last-price"].replace(",", ""))
            except (ValueError, KeyError):
                pass

        # Strategy 2: Look for large price text in finance card
        if price is None:
            # Google finance card has big price spans
            for span in soup.select("span[data-value], span[data-lp]"):
                val = span.get("data-value") or span.get("data-lp")
                if val:
                    try:
                        price = float(val.replace(",", ""))
                        if 1 < price < 200000:
                            break
                    except ValueError:
                        continue

        # Strategy 3: Regex on raw HTML for price patterns near NSE/BSE
        if price is None:
            m = re.search(r'data-last-price="([\d,\.]+)"', html)
            if m:
                price = float(m.group(1).replace(",", ""))

        if price is None:
            # Look for price near NSE text
            m = re.search(r'>\s*([\d,]+\.\d{1,2})\s*</span>', html)
            if m:
                candidate = float(m.group(1).replace(",", ""))
                if 100 < candidate < 200000:
                    price = candidate

        if price is None:
            m = re.search(r'"(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)"[^}]{0,200}"INR"', html)
            if m:
                price = float(m.group(1).replace(",", ""))

        # Extract change info
        m = re.search(r'([+-][\d,]+\.?\d*)\s*\(([+-]?[\d\.]+)\s*%\)', html)
        if m:
            change = float(m.group(1).replace(",", ""))
            change_pct = float(m.group(2))

        if price is None:
            logger.debug(f"Google price scrape: no price found for {symbol_up}")
            return {"error": "Could not parse price from Google", "symbol": symbol_up}

        result: Dict[str, Any] = {
            "symbol": symbol_up,
            "ltp": price,
            "change": change,
            "changePercent": change_pct,
            "open": None, "high": None, "low": None, "prevClose": None,
            "timestamp": datetime.now().isoformat(),
            "source": "google_finance",
        }

        cache.set("google_price", result, symbol_up, ttl_seconds=15)
        logger.info(f"✅ Google price for {symbol_up}: ₹{price}" + (f" ({change_pct:+.2f}%)" if change_pct else ""))
        return result

    except Exception as e:
        logger.debug(f"Google price fetch error for {symbol_up}: {e}")
        return {"error": str(e), "symbol": symbol_up}


async def fetch_ddg_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch stock/index price via DuckDuckGo search scraping.
    Uses BeautifulSoup to parse DDG HTML results.
    Speed: ~1-2s
    """
    from urllib.parse import quote_plus

    symbol_up = symbol.strip().upper()
    query = _GOOGLE_QUERY_MAP.get(symbol_up, f"{symbol_up} NSE stock price today")
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    cache = get_nse_cache()
    cached = cache.get("ddg_price", symbol_up)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(headers=_SEARCH_HEADERS, timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Get all result snippets and look for price patterns
        price = None
        change = None
        change_pct = None

        for snippet_tag in soup.select(".result__snippet")[:5]:
            snippet_text = snippet_tag.get_text(strip=True)

            if not price:
                price = _parse_price_from_text(snippet_text)
            if not change:
                change, change_pct = _parse_change_from_text(snippet_text)
            if price:
                break

        # Also check result titles
        if not price:
            for title_tag in soup.select(".result__a")[:5]:
                title_text = title_tag.get_text(strip=True)
                price = _parse_price_from_text(title_text)
                if price:
                    break

        if price is None:
            logger.debug(f"DDG price scrape: no price found for {symbol_up}")
            return {"error": "Could not parse price from DDG", "symbol": symbol_up}

        result: Dict[str, Any] = {
            "symbol": symbol_up,
            "ltp": price,
            "change": change,
            "changePercent": change_pct,
            "open": None, "high": None, "low": None, "prevClose": None,
            "timestamp": datetime.now().isoformat(),
            "source": "duckduckgo",
        }

        cache.set("ddg_price", result, symbol_up, ttl_seconds=15)
        logger.info(f"✅ DDG price for {symbol_up}: ₹{price}" + (f" ({change_pct:+.2f}%)" if change_pct else ""))
        return result

    except Exception as e:
        logger.debug(f"DDG price fetch error for {symbol_up}: {e}")
        return {"error": str(e), "symbol": symbol_up}
