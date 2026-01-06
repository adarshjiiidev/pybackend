from typing import List
from datetime import datetime
from agent.schemas.state import AgentState
from agent.schemas.models import MarketMetrics

def normalize_data(state: AgentState):
    """
    Node to normalize raw fetched data into structural MarketMetrics.
    """
    raw_data = state.get('fetched_data', {})
    if not raw_data:
        return {"normalized_metrics": []}
    
    metrics_list: List[MarketMetrics] = []
    
    for ticker, info in raw_data.items():
        if not info:
            continue
            
        try:
            # Map yfinance info keys to our schema
            metrics = MarketMetrics(
                ticker=ticker,
                price=info.get('currentPrice') or info.get('regularMarketPrice'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                eps=info.get('trailingEps'),
                volume=info.get('volume'),
                currency=info.get('currency', 'USD'),
                last_updated=datetime.now().isoformat(),
                # Add profit margins for accurate profitability analysis
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins')
            )
            metrics_list.append(metrics)
        except Exception as e:
            # Log error but continue
            print(f"Error normalizing {ticker}: {e}")
            
    return {"normalized_metrics": metrics_list}
