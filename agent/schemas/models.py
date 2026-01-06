from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class FinancialQuery(BaseModel):
    """
    Structured representation of the user's financial query.
    """
    tickers: List[str] = Field(description="List of stock tickers (e.g., AAPL, TCS.NS, NIFTY).")
    intent: Literal["market_data", "comparative_analysis", "general_chat", "options_trading"] = Field(
        description="Type of query: 'market_data' for ticker stats, 'comparative_analysis' for comparisons, 'general_chat' for non-financial, 'options_trading' for LTP Calculator/option chain analysis."
    )
    timeframe: str = Field(description="Timeframe for the query e.g., '1y', 'ytd', '1mo'. Default to '1y' if unspecified.")
    original_query: str = Field(description="The original natural language query from the user.")
    language: str = Field(default="english", description="Detected language: 'english' or 'hindi'")

class MarketMetrics(BaseModel):
    """
    Standardized market metrics for a ticker.
    """
    ticker: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    volume: Optional[int] = None
    currency: str = "USD"
    last_updated: str
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None

class FinancialInsight(BaseModel):
    """
    Structure for the final generated response.
    """
    executive_summary: str = Field(description="Concise summary of the findings.")
    key_metrics: List[MarketMetrics] = Field(description="List of key financial metrics for relevant tickers.")
    comparative_analysis: Optional[str] = Field(description="Comparison text if applicable, else None.")
    risk_factors: List[str] = Field(description="List of potential risks identified.")
    final_insight: str = Field(description="Concluding insight or recommendation context.")
    disclaimer: str = Field(description="Standard financial disclaimer.")
