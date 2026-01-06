from agent.schemas.state import AgentState
from agent.tools.market_data import MarketDataTool

def fetch_data(state: AgentState):
    """
    Node to fetch data from yfinance based on the parsed query.
    """
    query = state.get('parsed_query')
    if not query:
        return {"error": "No parsed query found."}

    tickers = query.tickers
    if not tickers:
        return {"fetched_data": {}}

    # Filter out special GREETING ticker
    valid_tickers = [t for t in tickers if t != "GREETING"]
    
    if not valid_tickers:
        return {"fetched_data": {}}

    tool = MarketDataTool()
    
    # We fetch basic info for all tickers
    data = tool.get_market_data(valid_tickers)
    
    # If timeframe implies history needed (not implemented fully in this MVP but placeholder)
    # history = tool.get_history(tickers[0], period=query.timeframe)
    
    return {"fetched_data": data}
