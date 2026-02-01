"""
Formatting utilities for agent outputs.
Ensures clean, user-facing responses without internal reasoning.
"""

from typing import Any
import re


def format_stock_info(stock_data: dict[str, Any]) -> str:
    """Format stock information for LLM context."""
    if "error" in stock_data:
        return f"Error fetching data: {stock_data['error']}"
    
    formatted = f"""Stock: {stock_data.get('name', 'N/A')} ({stock_data.get('symbol')})
Sector: {stock_data.get('sector', 'N/A')} | Industry: {stock_data.get('industry', 'N/A')}

Price Information:
- Current Price: ₹{stock_data.get('currentPrice', 'N/A')}
- Previous Close: ₹{stock_data.get('previousClose', 'N/A')}
- Day Range: ₹{stock_data.get('dayLow', 'N/A')} - ₹{stock_data.get('dayHigh', 'N/A')}
- 52 Week Range: ₹{stock_data.get('fiftyTwoWeekLow', 'N/A')} - ₹{stock_data.get('fiftyTwoWeekHigh', 'N/A')}

Key Metrics:
- Market Cap: ₹{format_large_number(stock_data.get('marketCap'))}
- P/E Ratio: {stock_data.get('peRatio', 'N/A')}
- Beta: {stock_data.get('beta', 'N/A')}
- Volume: {format_large_number(stock_data.get('volume'))}
- Dividend Yield: {format_percent(stock_data.get('dividendYield'))}
"""
    return formatted


def format_large_number(num: Any) -> str:
    """Format large numbers with Indian numbering system."""
    if num is None:
        return "N/A"
    
    try:
        num = float(num)
        if num >= 1e7:  # Crores
            return f"{num / 1e7:.2f}Cr"
        elif num >= 1e5:  # Lakhs
            return f"{num / 1e5:.2f}L"
        elif num >= 1e3:  # Thousands
            return f"{num / 1e3:.2f}K"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return str(num)


def format_percent(value: Any) -> str:
    """Format percentage values."""
    if value is None:
        return "N/A"
    
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def format_for_llm(data: dict[str, Any], data_type: str) -> str:
    """
    Format various data types for LLM consumption.
    
    Args:
        data: Data dictionary
        data_type: Type of data (stock, crypto, indices, historical)
    
    Returns:
        Formatted string for LLM context
    """
    if data_type == "stock":
        return format_stock_info(data)
    
    elif data_type == "crypto":
        return f"""Cryptocurrency: {data.get('name', 'N/A')}
Current Price: ${data.get('currentPrice', 'N/A')}
24h Change: {format_percent(data.get('dayChange', 0) / 100 if data.get('dayChange') else None)}
Market Cap: ${format_large_number(data.get('marketCap'))}
Volume: {format_large_number(data.get('volume'))}
"""
    
    elif data_type == "indices":
        formatted = "Indian Market Indices:\n"
        for name, index_data in data.items():
            if "error" not in index_data:
                formatted += f"\n{name}: {index_data.get('price', 'N/A')} "
                formatted += f"({index_data.get('changePercent', 'N/A')}%)\n"
        return formatted
    
    else:
        return str(data)


def clean_response(text: str) -> str:
    """
    Clean agent response by removing internal reasoning markers.
    
    Args:
        text: Raw agent response
    
    Returns:
        Cleaned response suitable for user display
    """
    # Remove common reasoning markers
    text = re.sub(r'\[REASONING\].*?\[/REASONING\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[INTERNAL\].*?\[/INTERNAL\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[THOUGHT\].*?\[/THOUGHT\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text
