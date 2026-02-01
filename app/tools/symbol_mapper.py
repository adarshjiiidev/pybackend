"""
Symbol mapping for Yahoo Finance.
Maps common stock names and indices to correct Yahoo Finance symbols.
"""

# Indian Stock Symbol Mappings
# Common names -> Yahoo Finance symbols
STOCK_SYMBOL_MAP = {
    # Popular stocks with different ticker names
    "HUL": "HINDUNILVR.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "RELIANCE": "RELIANCE.NS",
    "RIL": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "MARUTI": "MARUTI.NS",
    "BAJAJ": "BAJFINANCE.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "AXIS": "AXISBANK.NS",
    "LT": "LT.NS",
    "LARSEN": "LT.NS",
    "HCLTECH": "HCLTECH.NS",
    "HCL": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "NESTLE": "NESTLEIND.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANI": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "TITAN": "TITAN.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "JSW": "JSWSTEEL.NS",
    "HINDALCO": "HINDALCO.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "EICHER": "EICHERMOT.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "PIDILITE": "PIDILITIND.NS",
}

# Indian Market Indices
INDEX_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",  # Financial Services index
    "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
    "SENSEX": "^BSESN",
    "BSE SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTYIT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTYAUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTYMETAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTYREALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTYENERGY": "^CNXENERGY",
    "NIFTY MIDCAP": "^NSEMDCP50",
    "NIFTYMIDCAP": "^NSEMDCP50",
}


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol to its correct Yahoo Finance format.
    
    Args:
        symbol: User-provided symbol (e.g., "HUL", "NIFTY", "RELIANCE")
    
    Returns:
        Correct Yahoo Finance symbol (e.g., "HINDUNILVR.NS", "^NSEI", "RELIANCE.NS")
    """
    symbol_upper = symbol.upper().strip()
    
    # Check if it's an index
    if symbol_upper in INDEX_SYMBOL_MAP:
        return INDEX_SYMBOL_MAP[symbol_upper]
    
    # Check if it's a known stock with different ticker
    if symbol_upper in STOCK_SYMBOL_MAP:
        return STOCK_SYMBOL_MAP[symbol_upper]
    
    # If already has exchange suffix, return as-is
    if symbol_upper.endswith(('.NS', '.BO', '-USD', '.BSE')):
        return symbol_upper
    
    # Default: assume NSE stock, add .NS suffix
    return f"{symbol_upper}.NS"


def is_index(symbol: str) -> bool:
    """Check if a symbol is an index."""
    symbol_upper = symbol.upper().strip()
    return symbol_upper in INDEX_SYMBOL_MAP or symbol_upper.startswith('^')


def get_display_name(symbol: str) -> str:
    """Get user-friendly display name for a symbol."""
    symbol_upper = symbol.upper().strip()
    
    # Remove exchange suffixes for display
    if symbol_upper.endswith(('.NS', '.BO', '.BSE')):
        symbol_upper = symbol_upper.rsplit('.', 1)[0]
    
    # Index display names
    index_display = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "BANK NIFTY",
        "^BSESN": "SENSEX",
        "^CNXIT": "NIFTY IT",
        "^CNXAUTO": "NIFTY AUTO",
    }
    
    if symbol in index_display:
        return index_display[symbol]
    
    return symbol_upper
