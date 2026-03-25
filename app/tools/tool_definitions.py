"""
Advanced tool definitions for Groq function calling.
Provides comprehensive financial analysis tools.
"""

from typing import Any, Literal


# Tool definitions for Groq function calling
FINANCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": "Get comprehensive fundamental data for Indian stocks including financials, ratios, and company info. Use for NSE/BSE listed companies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., RELIANCE, TCS, INFY). Will auto-add .NS for NSE stocks"
                    },
                    "include_financials": {
                        "type": "boolean",
                        "description": "Whether to include detailed financial statements",
                        "default": True
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Calculate technical indicators (RSI, MACD, Moving Averages, Bollinger Bands) for a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "indicators": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["RSI", "MACD", "SMA", "EMA", "BB", "ATR"]
                        },
                        "description": "List of technical indicators to calculate"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                        "description": "Time period for analysis",
                        "default": "3mo"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_news",
            "description": "Search for latest financial news about stocks, sectors, or market events. Essential for market sentiment and breaking news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (company name, sector, or event)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of news articles to return",
                        "default": 5
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back",
                        "default": 7
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Analyze overall market sentiment using indices, VIX, breadth indicators, and advance-decline ratios",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "enum": ["INDIA", "GLOBAL", "CRYPTO"],
                        "description": "Which market to analyze",
                        "default": "INDIA"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare multiple stocks side-by-side across key metrics (PE, ROE, growth, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of stock symbols to compare"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["valuation", "profitability", "growth", "dividend", "risk"]
                        },
                        "description": "Which metric categories to compare"
                    }
                },
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_analysis",
            "description": "Get comprehensive sector performance, top performers, and trends",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name (e.g., Banking, IT, Pharma, Auto)"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1d", "1w", "1m", "3m", "6m", "1y"],
                        "description": "Performance period",
                        "default": "1m"
                    }
                },
                "required": ["sector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for additional context, research, or breaking information not available in market data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_optimization",
            "description": "Optimize portfolio allocation using modern portfolio theory, risk-return analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "stocks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of stock symbols to include in portfolio"
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["conservative", "moderate", "aggressive"],
                        "description": "Risk tolerance level"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Total investment amount in INR"
                    }
                },
                "required": ["stocks", "risk_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the proprietary knowledge base for domain-specific trading concepts, strategies, and terminology. Contains detailed information on: LTP calculator, WTB/WTT shifting, support/resistance, pressure analysis, game of percentage, options strategies (ITM/OTM/ATM), scenario building, 75% rule, and more. Use this FIRST for any trading concept or strategy question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive search query about the concept. Use the full concept name, not abbreviations. Examples: 'weak towards bottom rules and conditions', 'LTP calculator how to use', 'shifting pressure analysis method', 'support and resistance levels identification'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_nse_quote",
            "description": "Get FAST real-time NSE stock quote (LTP, volume, OHLC). Uses direct API - 10x faster than browser scraping (0.5-1.5s response time).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "NSE stock symbol (e.g., RELIANCE, TCS, INFY, HDFCBANK)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_fii_dii",
            "description": "Get FAST FII/DII participation data from NSE. Direct API call - much faster than browser scraping (1-2s response).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_option_chain",
            "description": "Get FAST NSE option chain data. Direct API - 10x faster than browser scraping (1-2s response).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol for option chain (NIFTY, BANKNIFTY, FINNIFTY, or stock symbol)",
                        "default": "NIFTY"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_market_status",
            "description": "Get current NSE market status and trading state. Fast API call.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_with_puppeteer",
            "description": "SLOW browser scraping - only use as FALLBACK when fast APIs don't work. For TradingView or special pages only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "enum": ["nse-ltp", "fii-dii", "option-chain", "tradingview", "generic"],
                        "description": "Type of data to scrape: nse-ltp (NSE LTP calculator), fii-dii (FII/DII data), option-chain (NSE options), tradingview (charts), generic (any URL)"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for scraping. For nse-ltp/option-chain/tradingview: {symbol: 'RELIANCE'}. For generic: {url: 'https://...', selector: '.price'}. For fii-dii: {}"
                    }
                },
                "required": ["endpoint", "params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_history",
            "description": "Fetch historical daily price data for Indian market indices like NIFTY 50, SENSEX, BANKNIFTY, NIFTYIT. Returns an array of daily {time, open, high, low, close, value} bars ready for chart generation. Use this when the user asks to compare, chart, or analyse index performance over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Index symbol(s). Use: 'NIFTY50' for Nifty 50, 'SENSEX' for BSE Sensex, 'BANKNIFTY' for Bank Nifty, 'NIFTYIT' for Nifty IT."
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of calendar days of history to fetch. Default 90 (3 months).",
                        "default": 90
                    }
                },
                "required": ["symbols"]
            }
        }
    }
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get all available tool definitions for Groq function calling."""
    return FINANCIAL_TOOLS


def get_tool_names() -> list[str]:
    """Get list of all available tool names."""
    return [tool["function"]["name"] for tool in FINANCIAL_TOOLS]
