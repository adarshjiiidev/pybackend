"""
Fast NSE data scraper using direct HTTP API calls.
10x faster than Puppeteer (0.5-2s vs 10-30s response time).
"""

import httpx
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .nse_cache import get_nse_cache

logger = logging.getLogger(__name__)


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
        Get live stock quote data (LTP, volume, etc.).
        Fast alternative to Puppeteer NSE LTP scraper.
        
        Speed: ~0.5-1.5 seconds
        """
        try:
            # Check cache first
            cache = get_nse_cache()
            cached_data = cache.get("quote", symbol)
            if cached_data:
                return cached_data

            client = await self._get_session()
            
            url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol.upper()}"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant fields
            price_info = data.get("priceInfo", {})
            
            result = {
                "symbol": symbol,
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

            # Store in cache
            cache.set("quote", result, symbol)
            return result
        except Exception as e:
            logger.error(f"NSE quote fetch error for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
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
