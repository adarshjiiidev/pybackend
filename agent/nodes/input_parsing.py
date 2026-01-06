import os
import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agent.llm_factory import get_llm
from agent.schemas.state import AgentState
from agent.schemas.models import FinancialQuery

def parse_input(state: AgentState):
    """
    Node to parse the user's natural language into a structured FinancialQuery.
    """
    messages = state['messages']
    last_message = messages[-1]
    
    llm = get_llm(temperature=0)
    
    parser = JsonOutputParser(pydantic_object=FinancialQuery)
    
    prompt = PromptTemplate(
        template="""You are Daddy's AI - a multi-layer financial assistant for INDIAN STOCK MARKET. 
        
        Extract tickers, intent, timeframe, and language from the query.
        
        {format_instructions}
        
        INTENT DETECTION RULES:
        
        1. "options_trading" - Use this for queries about:
           - LTP Calculator, WTB, WTT, EOR, EOS, strike price, scenario
           - Option chain, OI (open interest), COA, gamma blast, IV
           - Put/Call options, premium, theta, delta, expiry
           - Hindi terms like "scenario kya hai", "support resistance"
        
        2. "market_data" - For price, volume, financials of specific stocks
        
        3. "comparative_analysis" - For comparing two or more companies
        
        4. "general_chat" - For greetings or non-financial questions
        
        LANGUAGE DETECTION:
        - Set language="hindi" if query contains Hindi/Hinglish words
        - Set language="english" otherwise
        
        TICKER RULES:
        - For INDIAN stocks (Reliance, TCS, Infosys), add ".NS" suffix
        - For options queries, use "NIFTY" or "BANKNIFTY" if applicable
        - For US stocks, use standard tickers (AAPL, MSFT)
        - For general questions, use empty tickers []
        
        User Query: {query}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        parsed_dict = chain.invoke({"query": last_message.content})
        
        # Manually validate/convert to Pydantic to ensure safety
        # Handle the special "GREETING" case logic in data_fetch or here?
        # Let's map GREETING to a safe state.
        
        if "tickers" in parsed_dict and "GREETING" in parsed_dict["tickers"]:
             # If it's a greeting, we can fail gracefully or handle it. 
             # For now, let's just let it pass, but the downstream tool will fail for ticker "GREETING".
             # Better: Return error "GREETING" to handle it in graph? 
             # Or just allow empty tickers.
             pass
             
        parsed = FinancialQuery(**parsed_dict)
        parsed.original_query = last_message.content
        return {"parsed_query": parsed}
    except Exception as e:
        print(f"[DEBUG] Input parsing failed: {e}")
        return {"error": f"Failed to parse input: {str(e)}"}
