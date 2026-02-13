"""
Async Centralized Data Pool for Knowledge Base
Provides unified async access to all knowledge with entity indexing, semantic search, and DB caching.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from ..rag.knowledge_base import KnowledgeBaseRAG
from ..rag.metadata_indexer import MetadataIndexer, get_metadata_indexer
from ..rag.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class AsyncDataPool:
    """
    Async centralized knowledge base with entity indexing, semantic search, and DB caching.
    Single source of truth for all domain knowledge.
    """
    
    def __init__(self, knowledge_dir: str = "txt"):
        """
        Initialize data pool with knowledge base and indexing.
        
        Args:
            knowledge_dir: Directory containing knowledge files
        """
        self.kb = KnowledgeBaseRAG(knowledge_dir)
        self.metadata_indexer = get_metadata_indexer()
        self.embedding_service = get_embedding_service()
        self.use_db_cache = True  # Toggle for DB caching
        
        # Index all documents
        self._build_indices()
        
        logger.info(
            f"✅ Async data pool initialized: {len(self.kb.documents)} files, "
            f"{len(self.metadata_indexer.get_all_symbols())} symbols, "
            f"{len(self.metadata_indexer.get_all_concepts())} concepts"
        )
    
    def _build_indices(self):
        """Build metadata and vector indices for all documents."""
        logger.info("📊 Building knowledge base indices...")
        
        # Index metadata for all documents
        for filename, doc_data in self.kb.documents.items():
            content = doc_data["content"]
            self.metadata_indexer.index_document(filename, content)
        
        logger.info(f"✅ Indexed {len(self.kb.documents)} documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 3,
        use_semantic: bool = True,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base with hybrid semantic + keyword search and DB caching.
        
        Args:
            query: Search query
            top_k: Number of results
            use_semantic: Enable semantic search (requires sentence-transformers)
            use_cache: Use database cache for results
            
        Returns:
            List of relevant documents with scores
        """
        # Check database cache first
        if use_cache and self.use_db_cache:
            try:
                from ..models.knowledge_cache import KnowledgeSearchCache
                cached = await KnowledgeSearchCache.get_cached(query)
                
                if cached and cached.results:
                    logger.info(f"💾 Using DB cached search results for: {query[:50]}...")
                    return cached.results[:top_k]
            except Exception as e:
                logger.debug(f"Cache check failed: {e}")
        
        # Perform search in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._search_sync,
            query,
            top_k,
            use_semantic
        )
        
        # Cache results to database (permanent until manually invalidated)
        if use_cache and self.use_db_cache and results:
            try:
                from ..models.knowledge_cache import KnowledgeSearchCache
                search_type = "hybrid" if use_semantic else "keyword"
                await KnowledgeSearchCache.set_cache(query, results, search_type)
                logger.debug(f"💾 Permanently cached search results for: {query[:50]}...")
            except Exception as e:
                logger.debug(f"Cache save failed: {e}")
        
        return results
    
    def _search_sync(
        self,
        query: str,
        top_k: int,
        use_semantic: bool
    ) -> List[Dict[str, Any]]:
        """
        Synchronous search implementation (called in executor).
        """
        # Get keyword-based results from knowledge base
        keyword_results = self.kb.search(query, top_k=top_k * 2)  # Get more for re-ranking
        
        if not use_semantic or not self.embedding_service.is_available():
            return keyword_results[:top_k]
        
        # Add semantic similarity scores
        documents = [
            {
                "filename": r["filename"],
                "title": r["title"],
                "content": r["content"],
                **self.metadata_indexer.get_metadata(r["filename"])
            }
            for r in keyword_results
        ]
        
        if not documents:
            return []
        
        # Compute semantic similarity
        similar = self.embedding_service.find_most_similar(
            query,
            documents,
            top_k=top_k,
            doc_text_key="content"
        )
        
        # Combine keyword score + semantic score
        results = []
        for doc, semantic_score in similar:
            # Find corresponding keyword result for score
            keyword_result = next(
                (r for r in keyword_results if r["filename"] == doc["filename"]),
                None
            )
            
            keyword_score = keyword_result["score"] if keyword_result else 0
            
            # Hybrid score: 60% semantic, 40% keyword
            hybrid_score = 0.6 * semantic_score + 0.4 * (keyword_score / 100)
            
            results.append({
                **doc,
                "score": hybrid_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score
            })
        
        # Sort by hybrid score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    async def get_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge about a specific stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., "SOC", "TCS")
            
        Returns:
            List of relevant documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_by_symbol_sync, symbol)
    
    def _get_by_symbol_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Sync implementation of get_by_symbol."""
        filenames = self.metadata_indexer.get_files_by_symbol(symbol)
        
        results = []
        for filename in filenames:
            if filename in self.kb.documents:
                doc = self.kb.documents[filename]
                metadata = self.metadata_indexer.get_metadata(filename)
                
                results.append({
                    "filename": filename,
                    "title": doc["title"],
                    "content": doc["content"],
                    **metadata
                })
        
        return results
    
    async def get_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge about a specific concept.
        
        Args:
            concept: Concept identifier (e.g., "coa", "soc", "ltp")
            
        Returns:
            List of relevant documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_by_concept_sync, concept)
    
    def _get_by_concept_sync(self, concept: str) -> List[Dict[str, Any]]:
        """Sync implementation of get_by_concept."""
        filenames = self.metadata_indexer.get_files_by_concept(concept)
        
        results = []
        for filename in filenames:
            if filename in self.kb.documents:
                doc = self.kb.documents[filename]
                metadata = self.metadata_indexer.get_metadata(filename)
                
                results.append({
                    "filename": filename,
                    "title": doc["title"],
                    "content": doc["content"],
                    **metadata
                })
        
        return results
    
    def get_available_topics(self) -> Dict[str, List[str]]:
        """
        Get all available knowledge topics for routing decisions.
        Synchronous as it's fast and used in sync contexts.
        
        Returns:
            Dict with symbols, concepts, and categories
        """
        return {
            "symbols": list(self.metadata_indexer.get_all_symbols()),
            "concepts": list(self.metadata_indexer.get_all_concepts()),
            "categories": list(set([
                meta.get("category", "general")
                for meta in self.metadata_indexer.metadata_cache.values()
            ])),
            "files": list(self.kb.documents.keys())
        }
    
    def has_knowledge_about(self, query: str) -> bool:
        """
        Quick check if knowledge base has information about a query.
        Synchronous for router performance.
        
        Args:
            query: User query
            
        Returns:
            True if relevant knowledge exists
        """
        # Check if query matches any known topics
        available = self.get_available_topics()
        query_lower = query.lower()
        
        # Check symbols
        for symbol in available["symbols"]:
            if symbol.lower() in query_lower:
                return True  
        
        # Check concepts
        for concept in available["concepts"]:
            if concept in query_lower:
                return True
        
        # Try keyword search
        results = self.kb.search(query, top_k=1)
        return len(results) > 0
    
    async def get_context(self, query: str, max_chars: int = 3000) -> str:
        """
        Get formatted context for a query (for LLM consumption).
        
        Args:
            query: User query
            max_chars: Maximum characters
            
        Returns:
            Formatted context string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.kb.get_relevant_context,
            query,
            max_chars
        )


# Global instance
_async_data_pool: Optional[AsyncDataPool] = None


def get_async_data_pool() -> AsyncDataPool:
    """Get or create global async data pool instance."""
    global _async_data_pool
    if _async_data_pool is None:
        _async_data_pool = AsyncDataPool()
    return _async_data_pool


# Convenience function for tool usage
async def search_data_pool_async(query: str) -> Dict[str, Any]:
    """
    Async search data pool (used by agents as a tool).
    
    Args:
        query: Search query
        
    Returns:
        Search results with metadata
    """
    try:
        pool = get_async_data_pool()
        results = await pool.search(query, top_k=3, use_semantic=True, use_cache=True)
        
        if not results:
            return {
                "found": False,
                "message": "No relevant information found"
            }
        
        return {
            "found": True,
            "results": [
                {
                    "title": r["title"],
                    "filename": r["filename"],
                    "category": r.get("category", "general"),
                    "concepts": r.get("concepts", []),
                    "symbols": r.get("symbols", []),
                    "content": r["content"][:1500] + ("..." if len(r["content"]) > 1500 else ""),
                    "score": round(r["score"], 3)
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Data pool search error: {e}")
        return {"found": False, "error": str(e)}
