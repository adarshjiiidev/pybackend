"""RAG (Retrieval-Augmented Generation) package for knowledge base."""

# Qdrant-based RAG is the primary; keyword-based is the fallback (handled internally)
from .qdrant_kb import QdrantKBRAG, get_qdrant_rag
from .knowledge_base import KnowledgeBaseRAG, search_knowledge_base

# get_kb_rag now returns the Qdrant-backed instance
get_kb_rag = get_qdrant_rag

__all__ = [
    "QdrantKBRAG",
    "KnowledgeBaseRAG",
    "get_kb_rag",
    "get_qdrant_rag",
    "search_knowledge_base",
]
