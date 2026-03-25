"""
Database configuration and connection management.
Handles MongoDB connection using Motor async driver and Beanie ODM.
"""

import logging
import uuid
from typing import Optional

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ..models.chat_models import Conversation, Message
from ..models.db_models import (
    LoginAttempt,
    OTPRecord,
    TokenBlacklist,
    User,
    VerificationToken,
)
from ..models.knowledge_cache import KnowledgeSearchCache
# NOTE: data_pipeline models are imported lazily inside connect() to avoid
# circular imports (data_pool.py imports from config.database)

from .settings import settings

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
                serverSelectionTimeoutMS=5000,
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
                    logger.info(
                        f"Found {len(users_without_id)} users without user_id. Migrating..."
                    )

                    # Update each user with a new UUID
                    for user in users_without_id:
                        new_user_id = str(uuid.uuid4())
                        await users_collection.update_one(
                            {"_id": user["_id"]}, {"$set": {"user_id": new_user_id}}
                        )

                    logger.info(f"Successfully migrated {len(users_without_id)} users")
            except Exception as migration_error:
                logger.warning(f"Migration warning: {migration_error}")
                # Continue anyway - this is just a best-effort migration

            # Lazy import to avoid circular dependency:
            # data_pool.py imports config.database, so we can't import at module level
            from ..data_pipeline.data_pool import FundamentalsCache, OHLCVBar, SymbolMeta

            # Initialize Beanie with document models
            await init_beanie(
                database=cls.db,
                document_models=[
                    # Authentication models
                    User,
                    VerificationToken,
                    TokenBlacklist,
                    OTPRecord,
                    LoginAttempt,
                    # Chat models
                    Conversation,
                    Message,
                    # Cache models
                    KnowledgeSearchCache,  # Permanent KB search cache
                    # Data pipeline models — MUST be here for Beanie field descriptors to work
                    OHLCVBar,
                    SymbolMeta,
                    FundamentalsCache,
                ],
            )

            logger.info("MongoDB connection established successfully")

            # ── Performance indexes (idempotent, conflict-safe) ────────────────
            db = cls.db
            _otp_window_secs = (
                OTPRecord.__fields__
            )  # just import to resolve; actual value below
            _otp_ttl = 10 * 60  # OTP expires after 10 minutes (matches OTPService)
            _attempt_ttl = LoginAttempt.WINDOW_MINUTES * 60  # auto-purge after window

            indexes = [
                # (collection, index_spec, kwargs)
                # ── Users ──────────────────────────────────────────────────────
                ("users", [("email", 1)], {"unique": True, "background": True}),
                ("users", [("user_id", 1)], {"unique": True, "background": True}),
                # ── JWT blacklist (TTL auto-deletes expired entries) ───────────
                ("token_blacklist", [("jti", 1)], {"background": True}),
                (
                    "token_blacklist",
                    [("expires_at", 1)],
                    {"expireAfterSeconds": 0, "background": True},
                ),
                # ── OTP records (TTL auto-deletes after expiry) ───────────────
                ("otp_records", [("email", 1), ("purpose", 1)], {"background": True}),
                (
                    "otp_records",
                    [("expires_at", 1)],
                    {"expireAfterSeconds": 0, "background": True},
                ),
                # ── Login attempts (TTL auto-purges after sliding window) ──────
                (
                    "login_attempts",
                    [("email", 1), ("attempt_at", -1)],
                    {"background": True},
                ),
                (
                    "login_attempts",
                    [("attempt_at", 1)],
                    {"expireAfterSeconds": _attempt_ttl, "background": True},
                ),
                # ── Conversations & messages ───────────────────────────────────
                (
                    "conversations",
                    [("user_id", 1), ("updated_at", -1)],
                    {"background": True},
                ),
                # Anon conversation TTL — auto-delete after 7 days if no user_id
                (
                    "conversations",
                    [("created_at", 1)],
                    {
                        "expireAfterSeconds": 604800,
                        "partialFilterExpression": {"user_id": None},
                        "background": True,
                    },
                ),
                (
                    "messages",
                    [("conversation_id", 1), ("created_at", 1)],
                    {"background": True},
                ),
            ]
            for collection, spec, kwargs in indexes:
                try:
                    await db[collection].create_index(spec, **kwargs)
                except Exception as idx_err:
                    # Index already exists with same/different options — skip silently
                    logger.debug(f"Index on {collection}{spec} skipped: {idx_err}")
            logger.info(
                "✅ MongoDB indexes ensured (OTP TTL, LoginAttempt TTL, anon-convo TTL)"
            )

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
            await cls.client.admin.command("ping")
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
