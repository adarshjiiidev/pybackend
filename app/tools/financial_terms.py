"""
Financial terms and acronyms that should NOT be treated as stock symbols.
These are knowledge base terms that require explainer/educational responses.
"""

# Financial Terms - Common acronyms and phrases in finance
FINANCIAL_TERMS = {
    # Portfolio & Investment Terms
    "WTB": "Willing To Buy",
    "WTS": "Willing To Sell",
    "HODL": "Hold On for Dear Life",
    "DCA": "Dollar Cost Averaging",
    "NAV": "Net Asset Value",
    "AUM": "Assets Under Management",
    "SIP": "Systematic Investment Plan",
    "CAGR": "Compound Annual Growth Rate",
    "XIRR": "Extended Internal Rate of Return",
    "IRR": "Internal Rate of Return",
    "ROI": "Return on Investment",
    "ROE": "Return on Equity",
    "ROCE": "Return on Capital Employed",
    "P/E": "Price to Earnings Ratio",
    "PE": "Price to Earnings Ratio",
    "P/B": "Price to Book Ratio",
    "PB": "Price to Book Ratio",
    "EPS": "Earnings Per Share",
    "DPS": "Dividend Per Share",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
    "EBIT": "Earnings Before Interest and Taxes",
    "FCF": "Free Cash Flow",
    "OCF": "Operating Cash Flow",
    
    # Market & Trading Terms
    "IPO": "Initial Public Offering",
    "FPO": "Follow-on Public Offering",
    "OFS": "Offer for Sale",
    "AGM": "Annual General Meeting",
    "EGM": "Extraordinary General Meeting",
    "SEBI": "Securities and Exchange Board of India",
    "NSE": "National Stock Exchange",
    "BSE": "Bombay Stock Exchange",
    "MCX": "Multi Commodity Exchange",
    "NCDEX": "National Commodity and Derivatives Exchange",
    "NSDL": "National Securities Depository Limited",
    "CDSL": "Central Depository Services Limited",
    "KYC": "Know Your Customer",
    "DEMAT": "Dematerialized Account",
    "DP": "Depository Participant",
    
    # Trading & Orders
    "MIS": "Margin Intraday Square-off",
    "CNC": "Cash and Carry",
    "NRML": "Normal",
    "BO": "Bracket Order",
    "CO": "Cover Order",
    "GTD": "Good Till Date",
    "GTC": "Good Till Cancelled",
    "IOC": "Immediate or Cancel",
    "SL": "Stop Loss",
    "SLM": "Stop Loss Market",
    "LIMIT": "Limit Order",
    "MARKET": "Market Order",
    
    # Derivatives & Options
    "FNO": "Futures and Options",
    "CE": "Call European",
    "PE": "Put European",
    "ATM": "At The Money",
    "ITM": "In The Money",
    "OTM": "Out of The Money",
    "IV": "Implied Volatility",
    "VWAP": "Volume Weighted Average Price",
    "OI": "Open Interest",
    "PCR": "Put Call Ratio",
    "VIX": "Volatility Index",
    
    # Mutual Funds & ETF
    "ELSS": "Equity Linked Savings Scheme",
    "ETF": "Exchange Traded Fund",
    "MF": "Mutual Fund",
    "NFO": "New Fund Offer",
    "TER": "Total Expense Ratio",
    "EXIT LOAD": "Exit Load Fee",
    "LOCK-IN": "Lock-in Period",
    
    # Tax & Regulatory
    "LTCG": "Long Term Capital Gains",
    "STCG": "Short Term Capital Gains",
    "STT": "Securities Transaction Tax",
    "CTT": "Commodities Transaction Tax",
    "PAN": "Permanent Account Number",
    "AADHAAR": "Unique Identification Number",
    "ITR": "Income Tax Return",
    "TDS": "Tax Deducted at Source",
    
    # General Finance
    "APR": "Annual Percentage Rate",
    "EMI": "Equated Monthly Installment",
    "CIBIL": "Credit Information Bureau India Limited",
    "FICO": "Fair Isaac Corporation",
    "GDP": "Gross Domestic Product",
    "CPI": "Consumer Price Index",
    "WPI": "Wholesale Price Index",
    "REPO": "Repurchase Agreement",
    "CRR": "Cash Reserve Ratio",
    "SLR": "Statutory Liquidity Ratio",
    "RBI": "Reserve Bank of India",
    "NBFC": "Non-Banking Financial Company",
}

# Additional contextual terms (lowercase) that might appear in queries
CONTEXT_TERMS = {
    "what is", "explain", "define", "meaning", "means",
    "full form", "expansion", "stands for", "definition",
    "trading", "investing", "market", "stock market",
    "portfolio", "mutual fund", "derivative", "option",
    "futures", "forex", "commodity", "crypto", "cryptocurrency"
}


def is_financial_term(text: str) -> bool:
    """
    Check if the given text is a known financial term.
    
    Args:
        text: Text to check (symbol or query)
    
    Returns:
        True if it's a known financial term, False otherwise
    """
    text_upper = text.upper().strip()
    
    # Direct match in financial terms
    if text_upper in FINANCIAL_TERMS:
        return True
    
    # Check if query contains contextual terms + a financial term
    text_lower = text.lower()
    for context in CONTEXT_TERMS:
        if context in text_lower:
            # Extract potential term after context word
            for term in FINANCIAL_TERMS:
                if term.lower() in text_lower:
                    return True
    
    return False


def get_term_definition(term: str) -> str | None:
    """
    Get the definition/expansion of a financial term.
    
    Args:
        term: Financial term (e.g., "WTB", "CAGR")
    
    Returns:
        Definition string or None if not found
    """
    return FINANCIAL_TERMS.get(term.upper().strip())
