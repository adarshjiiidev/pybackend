"""
Database model for caching knowledge base search results.
Stores search queries and their results to avoid redundant indexing.
"""

from beanie import Document
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import Field


class KnowledgeSearchCache(Document):
    """Cache for knowledge base search results - persists until manually updated."""
    
    query: str = Field(..., description="Search query (normalized lowercase)")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    
    # Metadata
    search_type: str = Field(default="hybrid", description="Type of search: keyword, semantic, hybrid")
    result_count: int = Field(default=0, description="Number of results")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, description="Number of times this cache was hit")
    updated_at: Optional[datetime] = None  # Track when KB content is updated
    
    class Settings:
        name = "knowledge_search_cache"
        indexes = [
            "query",
            "created_at"
        ]
    
    @classmethod
    async def get_cached(cls, query: str) -> Optional["KnowledgeSearchCache"]:
        """
        Get cached search results for a query.
        
        Args:
            query: Search query (will be normalized)
        
        Returns:
            Cached results or None if not found
        """
        query_normalized = query.lower().strip()
        
        # Find cache (no expiration check - cache forever)
        cache = await cls.find_one(cls.query == query_normalized)
        
        if cache:
            # Update access metadata
            cache.last_accessed = datetime.utcnow()
            cache.access_count += 1
            await cache.save()
        
        return cache
    
    @classmethod
    async def set_cache(
        cls,
        query: str,
        results: List[Dict[str, Any]],
        search_type: str = "hybrid"
    ) -> "KnowledgeSearchCache":
        """
        Cache search results permanently (until knowledge base is updated).
        
        Args:
            query: Search query
            results: Search results to cache
            search_type: Type of search performed
            
        Returns:
            Created or updated cache document
        """
        query_normalized = query.lower().strip()
        
        # Check if cache exists and update it
        existing = await cls.find_one(cls.query == query_normalized)
        
        if existing:
            # Update existing cache
            existing.results = results
            existing.search_type = search_type
            existing.result_count = len(results)
            existing.updated_at = datetime.utcnow()
            await existing.save()
            return existing
        
        # Create new cache
        cache = cls(
            query=query_normalized,
            results=results,
            search_type=search_type,
            result_count=len(results)
        )
        
        await cache.insert()
        return cache
    
    @classmethod
    async def invalidate_all(cls):
        """Invalidate all cache entries (call when KB files are updated)."""
        await cls.delete_all()
