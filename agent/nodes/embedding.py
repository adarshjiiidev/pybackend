from langchain_core.documents import Document
from agent.schemas.state import AgentState
from agent.tools.vector_store import VectorStoreTool

def embed_knowledge(state: AgentState):
    """
    Node to embed the normalized data into Pinecone for future retrieval.
    This acts as 'long term memory' of what we have seen.
    """
    metrics = state.get('normalized_metrics', [])
    if not metrics:
        return {}
        
    documents = []
    for m in metrics:
        # Create a text representation. 
        # In a real system, we might be more verbose or include summary analysis.
        content = (
            f"Financial Metrics for {m.ticker} on {m.last_updated}:\n"
            f"Price: {m.price} {m.currency}\n"
            f"Market Cap: {m.market_cap}\n"
            f"PE Ratio: {m.pe_ratio}\n"
            f"Volume: {m.volume}"
        )
        
        # Metadata for filtering
        meta = {
            "ticker": m.ticker,
            "type": "market_metrics",
            "date": m.last_updated
        }
        
        documents.append(Document(page_content=content, metadata=meta))
        
    try:
        tool = VectorStoreTool()
        if tool.index:
            tool.upsert_documents(documents)
    except Exception as e:
        # Log error but don't crash. Long term memory is optional.
        print(f"[WARNING] Failed to store embedding: {e}")
    
    # We don't necessarily update state here, mostly side-effect.
    return {}
