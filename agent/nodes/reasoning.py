import os
import json
from agent.llm_factory import get_llm
from langchain_core.messages import HumanMessage
from agent.schemas.state import AgentState

def analyze_market(state: AgentState):
    """
    Node to perform reasoning on the active data.
    """
    query = state.get('parsed_query')
    metrics = state.get('normalized_metrics', [])
    context = state.get('retrieved_docs', [])
    
    if not query:
        return {"error": "No query to analyze."}

    # Construct context for the LLM
    data_summary = ""
    for m in metrics:
        data_summary += (
            f"Ticker: {m.ticker}\n"
            f"Price: {m.price} {m.currency}\n"
            f"Market Cap: {m.market_cap}\n"
            f"PE Ratio: {m.pe_ratio}\n"
            f"EPS: {m.eps}\n"
            f"Volume: {m.volume}\n"
            f"Profit Margin: {m.profit_margin * 100 if m.profit_margin else 'N/A'}%\n"
            f"Operating Margin: {m.operating_margin * 100 if m.operating_margin else 'N/A'}%\n\n"
        )
    
    docs_text = "\n\n".join(context)
    
    prompt_template = """
    You are a professional financial analyst with deep expertise in fundamental analysis and the INDIAN STOCK MARKET.
    
    User Query: "{query}"
    
    Intent: {intent}
    
    {context_section}
    
    CRITICAL FINANCIAL ANALYSIS RULES:
    
    1. PROFITABILITY METRICS (DO NOT CONFUSE THESE):
       - EPS (Earnings Per Share): Total profit ÷ number of shares. NEVER use this to compare profitability between companies.
       - Net Profit Margin: Net Income ÷ Revenue. THIS is the correct metric for profitability comparison.
       - Operating Margin: Operating Income ÷ Revenue. Use for operational efficiency.
    
    2. VALUATION METRICS:
       - P/E Ratio: Shows valuation premium, NOT profitability. High P/E = growth expectations.
       - Market Cap: Total company value (Price × Shares Outstanding).
    
    3. LIQUIDITY/TRADING:
       - Volume: Average daily trading volume. Indicates stock liquidity.
       - For mega-cap stocks (>$1T), liquidity is rarely a concern.
    
    4. COMPARATIVE ANALYSIS FRAMEWORK:
       For comparing companies:
       a) Revenue Growth (YoY, QoQ)
       b) Net Profit Margin (to compare profitability)
       c) Operating Margin (to compare efficiency)
       d) P/E Ratio (to compare valuation)
       e) Revenue Mix/Diversification (to assess risk)
    
    5. INDIAN MARKET SPECIFICS:
       - NSE/BSE context for Indian stocks
       - Currency should be INR for Indian stocks
       - Consider local regulations (SEBI, RBI policies)
    
    Task:
    - If intent is 'general_chat': Provide a helpful, accurate answer using your knowledge.
    - If intent is 'market_data' or 'comparative_analysis':
      1. Analyze the provided data using CORRECT financial metrics
      2. Use Net Profit Margin (NOT EPS) for profitability comparisons
      3. Explain P/E in context of growth expectations, not profitability
      4. Highlight revenue diversification as a key risk factor
      5. Provide actionable insights based on fundamental analysis
    
    NEVER say "Higher EPS indicates higher profitability." This is INCORRECT.
    ALWAYS use Net Profit Margin or Operating Margin for profitability statements.
    
    Keep analysis objective, professional, and technically accurate.
    """
    
    context_section = ""
    if metrics:
        context_section += "Current Market Data:\n" + data_summary + "\n"
    if docs_text:
        context_section += "Historical Context:\n" + docs_text + "\n"
        
    final_prompt = prompt_template.format(
        query=query.original_query,
        intent=query.intent,
        context_section=context_section
    )
    
    llm = get_llm(temperature=0.3)
    
    response = llm.invoke([HumanMessage(content=final_prompt)])
    
    return {"analysis_result": {"text": response.content}}
