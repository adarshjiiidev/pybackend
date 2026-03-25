"""
Database repository layer for CRUD operations.
Provides clean async interfaces for database access.
"""

from typing import Optional, Any
from datetime import datetime
import uuid
import logging

from ..models.db_models import ConversationMessage, UserSession

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


