"""
Python client for Puppeteer microservice.
Calls the Node.js service to scrape pages requiring browser automation.
"""

import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PuppeteerClient:
    """Client for calling Puppeteer microservice."""
    
    def __init__(self, base_url: str = "http://localhost:3001", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = 60.0  # Scraping can be slow
    
    async def scrape_nse_ltp(self, symbol: str) -> Dict[str, Any]:
        """Scrape NSE LTP calculator data for a symbol."""
        return await self._post("/api/scrape/nse-ltp", {"symbol": symbol})
    
    async def scrape_fii_dii(self) -> Dict[str, Any]:
        """Scrape FII/DII participation data from NSE."""
        return await self._post("/api/scrape/fii-dii", {})
    
    async def scrape_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict[str, Any]:
        """Scrape NSE option chain data."""
        params = {"symbol": symbol}
        if expiry:
            params["expiry"] = expiry
        return await self._post("/api/scrape/option-chain", params)
    
    async def scrape_tradingview(self, symbol: str, interval: str = "1D") -> Dict[str, Any]:
        """Scrape TradingView chart data."""
        return await self._post("/api/scrape/tradingview", {
            "symbol": symbol,
            "interval": interval
        })
    
    async def scrape_generic(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape any generic page."""
        params = {"url": url}
        if selector:
            params["selector"] = selector
        return await self._post("/api/scrape/generic", params)
    
    async def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to microservice."""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Puppeteer service error: {e}")
            raise Exception(f"Failed to scrape data: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if microservice is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False


# Global instance
_puppeteer_client: Optional[PuppeteerClient] = None


def get_puppeteer_client() -> PuppeteerClient:
    """Get or create Puppeteer client instance."""
    global _puppeteer_client
    if _puppeteer_client is None:
        from ..config import settings
        _puppeteer_client = PuppeteerClient(
            base_url=getattr(settings, 'puppeteer_service_url', 'http://localhost:3001'),
            api_key=getattr(settings, 'puppeteer_api_key', None)
        )
    return _puppeteer_client


async def scrape_with_puppeteer(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal function to scrape data using Puppeteer service.
    
    Args:
        endpoint: One of 'nse-ltp', 'fii-dii', 'option-chain', 'tradingview', 'generic'
        params: Parameters specific to the endpoint
    
    Returns:
        Scraped data from the microservice
    """
    client = get_puppeteer_client()
    
    endpoint_map = {
        "nse-ltp": client.scrape_nse_ltp,
        "fii-dii": client.scrape_fii_dii,
        "option-chain": client.scrape_option_chain,
        "tradingview": client.scrape_tradingview,
        "generic": client.scrape_generic
    }
    
    scraper = endpoint_map.get(endpoint)
    if not scraper:
        raise ValueError(f"Unknown endpoint: {endpoint}. Valid: {list(endpoint_map.keys())}")
    
    # Call the appropriate scraper with params
    if endpoint == "nse-ltp":
        return await scraper(params["symbol"])
    elif endpoint == "fii-dii":
        return await scraper()
    elif endpoint == "option-chain":
        return await scraper(params["symbol"], params.get("expiry"))
    elif endpoint == "tradingview":
        return await scraper(params["symbol"], params.get("interval", "1D"))
    elif endpoint == "generic":
        return await scraper(params["url"], params.get("selector"))
