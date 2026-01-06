from agent.schemas.state import AgentState
from agent.tools.vector_store import VectorStoreTool

def retrieve_context(state: AgentState):
    """
    Node to retrieve relevant documents from Pinecone.
    """
    query = state.get('parsed_query')
    if not query:
        return {}

    try:
        tool = VectorStoreTool()
        if not tool.index:
            return {"retrieved_docs": []}

        # Search using the original query or constructed keywords
        search_query = query.original_query
        
        docs = tool.similarity_search(search_query, k=3)
        return {"retrieved_docs": [d.page_content for d in docs]}
        
    except Exception as e:
        print(f"[WARNING] Retrieval failed: {e}")
        return {"retrieved_docs": []}
