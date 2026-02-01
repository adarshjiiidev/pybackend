"""
MongoDB connection management using Motor (async driver).
Provides connection lifecycle management and database access.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from beanie import init_beanie
from typing import Optional
import logging

from .settings import settings
from ..models.db_models import ConversationMessage, UserSession, MarketDataCache

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def connect(cls):
        """Initialize MongoDB connection and Beanie ODM."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")
            
            cls.client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=5000
            )
            
            cls.db = cls.client[settings.mongodb_db_name]
            
            # Initialize Beanie with document models
            await init_beanie(
                database=cls.db,
                document_models=[
                    ConversationMessage,
                    UserSession,
                    MarketDataCache
                ]
            )
            
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @classmethod
    async def close(cls):
        """Close MongoDB connection."""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")
    
    @classmethod
    async def ping(cls) -> bool:
        """Check if MongoDB connection is alive."""
        try:
            await cls.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False


# Convenience functions
async def init_db():
    """Initialize database connection."""
    await Database.connect()


async def close_db():
    """Close database connection."""
    await Database.close()


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if Database.db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return Database.db
