import yfinance as yf
import time
import logging
from typing import Dict, Any, List, Optional
from requests.exceptions import RequestException

logger = logging.getLogger("financial_agent")

class MarketDataTool:
    """
    Tool to fetch market data using yfinance with retry logic.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_ticker_info(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches basic info for a ticker.
        """
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                # Basic validation to check if data is valid
                if 'symbol' not in info: 
                     # Sometimes yfinance returns empty dict on failure without raising
                    raise ValueError(f"No data found for {ticker_symbol}")
                return info
            except (RequestException, ValueError, Exception) as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker_symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch info for {ticker_symbol} after {self.max_retries} attempts.")
                    return {}

    def get_market_data(self, tickers: List[str], period: str = "1y") -> Dict[str, Any]:
        """
        Fetches historical market data for a list of tickers.
        """
        data_map = {}
        for t in tickers:
            info = self.get_ticker_info(t)
            try:
                # We can also fetch history if needed, but for 'metrics' info is often enough 
                # for current price, PE, etc. 
                # If historical data is requested, we can use history()
                # For now, we return the info dict which contains most 'current' metrics
                data_map[t] = info
            except Exception as e:
                logger.error(f"Error processing {t}: {e}")
                data_map[t] = {}
        return data_map

    def get_history(self, ticker_symbol: str, period: str = "1mo") -> Any:
        try:
             ticker = yf.Ticker(ticker_symbol)
             return ticker.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching history for {ticker_symbol}: {e}")
            return None
