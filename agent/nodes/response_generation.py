import os
from agent.llm_factory import get_llm
from agent.schemas.state import AgentState
from agent.schemas.models import FinancialInsight

def generate_response(state: AgentState):
    """
    Node to format the final response into the strict output format.
    """
    analysis = state.get('analysis_result', {})
    metrics = state.get('normalized_metrics', [])
    
    if not analysis:
        return {"final_response": "I could not generate an analysis."}
    
    # Logic to bypass strict structure for general chat
    query = state.get('parsed_query')
    if query and query.intent == "general_chat":
        return {"final_response": analysis.get('text', '')}

    llm = get_llm(temperature=0)
    
    # If it's a general chat, we might want a looser structure or just use the same structure creatively.
    # We will instruct it to use 'final_insight' for the main answer and leave others empty.
    
    structured_llm = llm.with_structured_output(FinancialInsight)
    
    prompt = f"""
    Based on the following analysis and metrics, generate a final structured report.
    
    Analysis Ref:
    {analysis.get('text', '')}
    
    Metrics Ref:
    {metrics}
    
    Ensure the output strictly adheres to the FinancialInsight schema.
    For 'key_metrics', populate correctly from the provided metrics data.
    """
    
    try:
        insight: FinancialInsight = structured_llm.invoke(prompt)
        
        # Convert Pydantic to Markdown string for final display
        md_output = f"""
# Financial Report

## Executive Summary
{insight.executive_summary}

## Key Metrics
"""
        # Create a table for metrics
        if insight.key_metrics:
            md_output += "| Ticker | Price | Market Cap | PE | Volume |\n"
            md_output += "| --- | --- | --- | --- | --- |\n"
            for m in insight.key_metrics:
                md_output += f"| {m.ticker} | {m.price} | {m.market_cap} | {m.pe_ratio} | {m.volume} |\n"
        
        if insight.comparative_analysis:
            md_output += f"\n## Comparative Analysis\n{insight.comparative_analysis}\n"
            
        md_output += "\n## Risk Factors\n"
        for risk in insight.risk_factors:
            md_output += f"- {risk}\n"
            
        md_output += f"\n## Final Insight\n{insight.final_insight}\n"
        md_output += f"\n> [!WARNING]\n> {insight.disclaimer}"
        
        return {"final_response": md_output}
        
        return {"final_response": md_output}
        
    except Exception as e:
        return {"final_response": f"Error formatting response: {str(e)}", "error": str(e)}
