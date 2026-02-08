"""
Database configuration and connection management.
Handles MongoDB connection using Motor async driver and Beanie ODM.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from beanie import init_beanie
from typing import Optional
import uuid

from .settings import settings
from ..models.db_models import (
    MarketDataCache,
    User,
    VerificationToken,
    TokenBlacklist,
    OTP
)
from ..models.chat_models import Conversation, Message

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def connect(cls):
        """Initialize MongoDB connection and Beanie ODM with migration."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")
            
            cls.client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=5000
            )
            
            cls.db = cls.client[settings.mongodb_db_name]
            
            # Migration: Update existing users with null user_id before initializing indexes
            try:
                users_collection = cls.db["users"]
                
                # Find users with null or missing user_id
                users_without_id = await users_collection.find(
                    {"$or": [{"user_id": None}, {"user_id": {"$exists": False}}]}
                ).to_list(length=None)
                
                if users_without_id:
                    logger.info(f"Found {len(users_without_id)} users without user_id. Migrating...")
                    
                    # Update each user with a new UUID
                    for user in users_without_id:
                        new_user_id = str(uuid.uuid4())
                        await users_collection.update_one(
                            {"_id": user["_id"]},
                            {"$set": {"user_id": new_user_id}}
                        )
                    
                    logger.info(f"Successfully migrated {len(users_without_id)} users")
            except Exception as migration_error:
                logger.warning(f"Migration warning: {migration_error}")
                # Continue anyway - this is just a best-effort migration
            
            # Initialize Beanie with document models
            await init_beanie(
                database=cls.db,
                document_models=[
                    # Authentication models
                    User,
                    VerificationToken,
                    TokenBlacklist,
                    OTP,
                    # Chat models
                    Conversation,
                    Message,
                    # Cache
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


# Convenience functions for backward compatibility
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
