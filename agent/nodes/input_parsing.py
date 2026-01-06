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
        template="""You are a financial AI assistant specialized in the INDIAN STOCK MARKET. Extract tickers, intent, and timeframe.
        
        {format_instructions}
        
        Rules:
        1. If the query implies INDIAN stocks (e.g. "Reliance", "Tata Motors", "HDFC"), prefer adding ".NS" suffix for NSE (e.g., "RELIANCE.NS").
        2. If the user asks for US stocks, use standard tickers (e.g. "AAPL").
        3. If the query is a greeting ("hi") or general question ("what is quantum computing"), set intent to "general_chat" and tickers to [].
        4. "market_data" is for price, volume, financials. "comparative_analysis" is for comparing two entities.
        
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
