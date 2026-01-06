import os
import json
from pathlib import Path
from agent.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from agent.schemas.state import AgentState

# Load LTP Calculator system prompt once at module level
LTP_SYSTEM_PROMPT = ""
try:
    sys_prompt_path = Path(__file__).parent.parent.parent / "sys_prmpt.txt"
    if sys_prompt_path.exists():
        with open(sys_prompt_path, "r", encoding="utf-8") as f:
            LTP_SYSTEM_PROMPT = f.read()
except Exception as e:
    print(f"[WARNING] Could not load sys_prmpt.txt: {e}")

# Professional analysis prompt
PROFESSIONAL_PROMPT = """
You are a professional financial analyst with deep expertise in fundamental analysis and the INDIAN STOCK MARKET.

User Query: "{query}"
Intent: {intent}
Language: {language}

{context_section}

CRITICAL FINANCIAL ANALYSIS RULES:

1. PROFITABILITY METRICS:
   - Net Profit Margin: Net Income ÷ Revenue. Use THIS for profitability comparison.
   - Operating Margin: Operating Income ÷ Revenue. Use for efficiency.
   - NEVER use EPS alone to compare profitability between companies.

2. VALUATION METRICS:
   - P/E Ratio: Shows valuation premium, NOT profitability.
   - Market Cap: Total company value.

3. COMPARATIVE ANALYSIS:
   a) Net Profit Margin (profitability)
   b) Operating Margin (efficiency)  
   c) P/E Ratio (valuation)
   d) Revenue Diversification (risk)

4. INDIAN MARKET: Use INR, consider NSE/BSE context, SEBI regulations.

{language_instruction}

Keep analysis objective, accurate, and insightful.
"""

def analyze_market(state: AgentState):
    """
    Multi-layer reasoning node with AI-driven layer selection.
    - options_trading: Uses Daddy's AI persona with LTP Calculator knowledge
    - market_data/comparative_analysis: Uses professional fundamental analysis
    - general_chat: Natural conversation
    """
    query = state.get('parsed_query')
    metrics = state.get('normalized_metrics', [])
    context = state.get('retrieved_docs', [])
    
    if not query:
        return {"error": "No query to analyze."}

    intent = query.intent
    language = getattr(query, 'language', 'english')
    
    # Build market data context
    data_summary = ""
    for m in metrics:
        data_summary += (
            f"Ticker: {m.ticker}\n"
            f"Price: {m.price} {m.currency}\n"
            f"Market Cap: {m.market_cap}\n"
            f"PE Ratio: {m.pe_ratio}\n"
            f"Profit Margin: {m.profit_margin * 100 if m.profit_margin else 'N/A'}%\n"
            f"Operating Margin: {m.operating_margin * 100 if m.operating_margin else 'N/A'}%\n\n"
        )
    
    docs_text = "\n\n".join(context)
    
    # Build context section
    context_section = ""
    if metrics:
        context_section += "Current Market Data:\n" + data_summary + "\n"
    if docs_text:
        context_section += "Historical Context:\n" + docs_text + "\n"
    
    # Language-aware response style
    # Hindi query = WhatsApp/Hinglish style
    # English query = English professional style
    if language == "hindi":
        language_instruction = "Respond in Hindi/Hinglish (WhatsApp style). Be professional but use Hindi terms like 'aap', 'hota hai', etc."
    else:
        language_instruction = "Respond in professional English."
    
    # SYSTEM PROMPT IS ALWAYS APPLIED for financial knowledge
    system_prompt = LTP_SYSTEM_PROMPT if LTP_SYSTEM_PROMPT else "You are Daddy's AI - a financial expert for Indian stock market."
    
    # SELECT RESPONSE STYLE BASED ON INTENT
    if intent == "options_trading":
        # Options/LTP Calculator questions
        user_prompt = f"""
Query: {query.original_query}

{language_instruction}

Provide a helpful response using LTP Calculator concepts.
Explain with scenarios if applicable.
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
    elif intent == "general_chat":
        # General conversation but still with system knowledge
        user_prompt = f"""
User asked: {query.original_query}

{language_instruction}

Provide a helpful, natural response. Use your knowledge from the system prompt if relevant.
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
    else:
        # Professional fundamental analysis (market_data, comparative_analysis)
        user_prompt = PROFESSIONAL_PROMPT.format(
            query=query.original_query,
            intent=intent,
            language=language,
            context_section=context_section,
            language_instruction=language_instruction
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    
    llm = get_llm(temperature=0.3)
    response = llm.invoke(messages)
    
    return {"analysis_result": {"text": response.content, "intent": intent, "language": language}}

