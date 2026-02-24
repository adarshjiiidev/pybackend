"""
Database repository layer for CRUD operations.
Provides clean async interfaces for database access.
"""

from typing import Optional, Any
from datetime import datetime
import uuid
import logging

from ..models.db_models import ConversationMessage, UserSession, MarketDataCache

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for conversation message operations."""
    
    @staticmethod
    async def create_message(
        session_id: str,
        user_query: str,
        agent_mode: str,
        agent_response: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> ConversationMessage:
        """Save a chat message to database."""
        message = ConversationMessage(
            session_id=session_id,
            user_query=user_query,
            agent_mode=agent_mode,
            agent_response=agent_response,
            metadata=metadata or {}
        )
        await message.insert()
        logger.debug(f"Saved message for session {session_id}")
        return message
    
    @staticmethod
    async def get_conversation_history(
        session_id: str,
        limit: int = 50
    ) -> list[ConversationMessage]:
        """Retrieve conversation history for a session."""
        messages = await ConversationMessage.find(
            ConversationMessage.session_id == session_id
        ).sort(-ConversationMessage.created_at).limit(limit).to_list()
        
        # Reverse to get chronological order
        return list(reversed(messages))
    
    @staticmethod
    async def delete_session(session_id: str) -> int:
        """Delete all messages for a session."""
        result = await ConversationMessage.find(
            ConversationMessage.session_id == session_id
        ).delete()
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count


class SessionRepository:
    """Repository for user session operations."""
    
    @staticmethod
    async def create_session(
        user_id: Optional[str] = None,
        preferences: Optional[dict[str, Any]] = None
    ) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user_id,
            preferences=preferences or {}
        )
        await session.insert()
        logger.info(f"Created session {session.session_id}")
        return session
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[UserSession]:
        """Retrieve a session by ID."""
        return await UserSession.find_one(UserSession.session_id == session_id)
    
    @staticmethod
    async def update_session_preferences(
        session_id: str,
        preferences: dict[str, Any]
    ) -> Optional[UserSession]:
        """Update session preferences."""
        session = await SessionRepository.get_session(session_id)
        if session:
            session.preferences.update(preferences)
            session.updated_at = datetime.utcnow()
            await session.save()
            logger.debug(f"Updated preferences for session {session_id}")
        return session
    
    @staticmethod
    async def deactivate_session(session_id: str) -> bool:
        """Deactivate a session."""
        session = await SessionRepository.get_session(session_id)
        if session:
            session.active = False
            session.updated_at = datetime.utcnow()
            await session.save()
            return True
        return False


class CacheRepository:
    """Repository for market data cache operations."""
    
    @staticmethod
    async def get_cached_data(
        symbol: str,
        data_type: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve cached market data if not expired."""
        # Use cache_key with unique index for fastest lookup
        cache_key = f"{symbol}:{data_type}"
        cache_entry = await MarketDataCache.find_one(
            MarketDataCache.cache_key == cache_key
        )
        
        if cache_entry and not cache_entry.is_expired():
            logger.debug(f"Cache HIT for {symbol} ({data_type})")
            return cache_entry.data
        
        if cache_entry:
            logger.debug(f"Cache EXPIRED for {symbol} ({data_type})")
            await cache_entry.delete()
        
        logger.debug(f"Cache MISS for {symbol} ({data_type})")
        return None
    
    @staticmethod
    async def set_cached_data(
        symbol: str,
        data_type: str,
        data: dict[str, Any],
        ttl_seconds: int = 300
    ) -> MarketDataCache:
        """Store market data in cache."""
        # Generate cache key from symbol and data_type
        cache_key = f"{symbol}:{data_type}"
        
        # Delete existing cache entry if present using unique cache_key
        await MarketDataCache.find(
            MarketDataCache.cache_key == cache_key
        ).delete()
        
        # Create new cache entry
        cache_entry = MarketDataCache(
            cache_key=cache_key,
            symbol=symbol,
            data_type=data_type,
            data=data,
            ttl_seconds=ttl_seconds
        )
        await cache_entry.insert()
        logger.debug(f"Cached data for {symbol} ({data_type}) with TTL {ttl_seconds}s")
        return cache_entry
    
    @staticmethod
    async def invalidate_cache(symbol: str) -> int:
        """Clear all cached data for a symbol."""
        result = await MarketDataCache.find(
            MarketDataCache.symbol == symbol
        ).delete()
        logger.info(f"Invalidated cache for {symbol}: {result.deleted_count} entries")
        return result.deleted_count
    
    @staticmethod
    async def clear_expired_cache() -> int:
        """Remove all expired cache entries."""
        # Use MongoDB expression to find and delete all expired entries in a single query
        # expiry_time = cached_at + ttl_seconds (converted to ms for MongoDB $add)
        now = datetime.utcnow()
        result = await MarketDataCache.find({
            "$expr": {
                "$lt": [
                    {"$add": ["$cached_at", {"$multiply": ["$ttl_seconds", 1000]}]},
                    now
                ]
            }
        }).delete()
        
        deleted_count = result.deleted_count if result else 0
        logger.info(f"Cleared {deleted_count} expired cache entries")
        return deleted_count
