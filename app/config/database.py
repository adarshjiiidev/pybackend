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
from ..models.db_models import LoginAttempt, TokenBlacklist, User, VerificationToken, Subscription, PaymentTransaction
from ..models.knowledge_cache import KnowledgeSearchCache
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

            # Best-effort migration for legacy users missing user_id.
            try:
                users_collection = cls.db["users"]
                users_without_id = await users_collection.find(
                    {"$or": [{"user_id": None}, {"user_id": {"$exists": False}}]}
                ).to_list(length=None)

                if users_without_id:
                    logger.info(
                        f"Found {len(users_without_id)} users without user_id. Migrating..."
                    )
                    for user in users_without_id:
                        await users_collection.update_one(
                            {"_id": user["_id"]},
                            {"$set": {"user_id": str(uuid.uuid4())}},
                        )
                    logger.info(f"Successfully migrated {len(users_without_id)} users")
            except Exception as migration_error:
                logger.warning(f"Migration warning: {migration_error}")

            # Lazy import to avoid circular dependency:
            # data_pool.py imports config.database.
            from ..data_pipeline.data_pool import FundamentalsCache, OHLCVBar, SymbolMeta

            await init_beanie(
                database=cls.db,
                document_models=[
                    # Authentication models
                    User,
                    VerificationToken,
                    TokenBlacklist,
                    LoginAttempt,
                    # Chat models
                    Conversation,
                    Message,
                    # Cache models
                    KnowledgeSearchCache,
                    # Data pipeline models
                    OHLCVBar,
                    SymbolMeta,
                    FundamentalsCache,
                    # Payment models
                    Subscription,
                    PaymentTransaction,
                ],
            )

            logger.info("MongoDB connection established successfully")

            db = cls.db
            _attempt_ttl = LoginAttempt.WINDOW_MINUTES * 60

            indexes = [
                # Users
                ("users", [("email", 1)], {"unique": True, "background": True}),
                ("users", [("user_id", 1)], {"unique": True, "background": True}),
                # JWT blacklist
                ("token_blacklist", [("jti", 1)], {"background": True}),
                (
                    "token_blacklist",
                    [("expires_at", 1)],
                    {"expireAfterSeconds": 0, "background": True},
                ),
                # Login attempts (TTL for brute-force window)
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
                # Conversations and messages
                (
                    "conversations",
                    [("user_id", 1), ("updated_at", -1)],
                    {"background": True},
                ),
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
                # Subscriptions
                ("subscriptions", [("user_id", 1)], {"unique": True, "background": True}),
                ("subscriptions", [("status", 1), ("expires_at", 1)], {"background": True}),
                # Payment transactions
                ("payment_transactions", [("user_id", 1), ("created_at", -1)], {"background": True}),
                ("payment_transactions", [("razorpay_order_id", 1)], {"unique": True, "background": True}),
            ]

            for collection, spec, kwargs in indexes:
                try:
                    await db[collection].create_index(spec, **kwargs)
                except Exception as idx_err:
                    logger.debug(f"Index on {collection}{spec} skipped: {idx_err}")

            # OTP storage is now in-memory only. Purge legacy persisted OTP data.
            try:
                purge_result = await db["otp_records"].delete_many({})
                if purge_result.deleted_count:
                    logger.info(
                        f"Purged {purge_result.deleted_count} legacy OTP records from MongoDB"
                    )
            except Exception as purge_err:
                logger.debug(f"Legacy OTP purge skipped: {purge_err}")

            logger.info("MongoDB indexes ensured (LoginAttempt TTL, anon-convo TTL)")

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
