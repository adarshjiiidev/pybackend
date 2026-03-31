"""
OTP (One-Time Password) service with in-memory ephemeral storage.

Security and behavior notes:
  - OTP records are never persisted to MongoDB.
  - OTPs expire automatically and are one-time use.
  - Codes are stored as HMAC-SHA256 hashes (never plaintext in memory store).
  - Rate limiting is process-local (per worker), not database-backed.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import ClassVar, Dict, List, TypedDict

from ..config import settings
from .email_service import EmailService

logger = logging.getLogger(__name__)


class _OTPEntry(TypedDict):
    code_hash: str
    expires_at: datetime
    attempts: int
    created_at: datetime


class OTPService:
    """Service for generating and validating OTPs without database persistence."""

    OTP_LENGTH: int = 6
    OTP_EXPIRY_MINUTES: int = 10
    MAX_ATTEMPTS: int = 5
    RATE_LIMIT_MINUTES: int = 15
    MAX_OTPS_PER_PERIOD: int = 3

    _records: ClassVar[Dict[str, _OTPEntry]] = {}
    _rate_window: ClassVar[Dict[str, List[datetime]]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _DEV_FALLBACK_SECRET: ClassVar[str] = "dev-otp-fallback-change-me-in-production"

    @classmethod
    def _record_key(cls, email: str, purpose: str) -> str:
        return f"{email.lower()}::{purpose}"

    @classmethod
    def _get_secret(cls) -> bytes:
        secret = settings.otp_secret
        if not secret:
            if settings.is_production:
                raise RuntimeError(
                    "OTP_SECRET is required in production for OTP hashing."
                )
            logger.warning(
                "OTP_SECRET not set. Using insecure dev fallback for OTP hashing."
            )
            secret = cls._DEV_FALLBACK_SECRET
        return secret.encode("utf-8")

    @classmethod
    def _hash_code(cls, code: str) -> str:
        return hmac.new(
            cls._get_secret(), code.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    @classmethod
    def _purge_expired_locked(cls, now: datetime) -> int:
        deleted = 0

        expired_keys = [
            key for key, entry in cls._records.items() if now >= entry["expires_at"]
        ]
        for key in expired_keys:
            cls._records.pop(key, None)
            deleted += 1

        window_start = now - timedelta(minutes=cls.RATE_LIMIT_MINUTES)
        stale_emails: list[str] = []
        for email, timestamps in cls._rate_window.items():
            fresh = [ts for ts in timestamps if ts >= window_start]
            if fresh:
                cls._rate_window[email] = fresh
            else:
                stale_emails.append(email)
        for email in stale_emails:
            cls._rate_window.pop(email, None)

        return deleted

    @staticmethod
    def generate_otp_code() -> str:
        """Generate a cryptographically secure 6-digit OTP code."""
        code = secrets.randbelow(10**OTPService.OTP_LENGTH)
        return f"{code:0{OTPService.OTP_LENGTH}d}"

    @staticmethod
    async def check_rate_limit(email: str) -> bool:
        """Return True if email is under the configured OTP request limit."""
        email = email.strip().lower()
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=OTPService.RATE_LIMIT_MINUTES)

        async with OTPService._lock:
            OTPService._purge_expired_locked(now)
            timestamps = OTPService._rate_window.get(email, [])
            recent = [ts for ts in timestamps if ts >= window_start]
            OTPService._rate_window[email] = recent
            return len(recent) < OTPService.MAX_OTPS_PER_PERIOD

    @staticmethod
    async def create_and_send_otp(
        email: str,
        purpose: str = "registration",
    ) -> tuple[bool, str]:
        """
        Create a new in-memory OTP record and send the plaintext code by email.

        Returns:
            (success: bool, message: str)
        """
        email = email.strip().lower()
        now = datetime.utcnow()

        try:
            async with OTPService._lock:
                OTPService._purge_expired_locked(now)

                window_start = now - timedelta(minutes=OTPService.RATE_LIMIT_MINUTES)
                timestamps = OTPService._rate_window.get(email, [])
                recent = [ts for ts in timestamps if ts >= window_start]
                if len(recent) >= OTPService.MAX_OTPS_PER_PERIOD:
                    return (
                        False,
                        f"Too many OTP requests. Please wait {OTPService.RATE_LIMIT_MINUTES} "
                        "minutes before trying again.",
                    )

                plaintext_code = OTPService.generate_otp_code()
                code_hash = OTPService._hash_code(plaintext_code)
                expires_at = now + timedelta(minutes=OTPService.OTP_EXPIRY_MINUTES)
                record_key = OTPService._record_key(email, purpose)

                OTPService._records[record_key] = {
                    "code_hash": code_hash,
                    "expires_at": expires_at,
                    "attempts": 0,
                    "created_at": now,
                }

                recent.append(now)
                OTPService._rate_window[email] = recent

            email_sent = await EmailService.send_otp_email(email, plaintext_code, purpose)
            if not email_sent:
                async with OTPService._lock:
                    OTPService._records.pop(OTPService._record_key(email, purpose), None)
                logger.warning(f"Email delivery failed for OTP to {email}")
                return False, "Failed to send OTP email. Please try again."

            logger.info(f"OTP sent (in-memory only) - email={email} purpose={purpose}")
            return True, "OTP sent successfully"

        except Exception as exc:
            logger.error(f"create_and_send_otp error for {email}: {exc}")
            return False, "An error occurred. Please try again."

    @staticmethod
    async def verify_otp(
        email: str,
        code: str,
        purpose: str = "registration",
    ) -> tuple[bool, str]:
        """
        Verify a submitted OTP code against in-memory hashed record.

        Returns:
            (success: bool, message: str)
        """
        email = email.strip().lower()
        code = code.strip()
        now = datetime.utcnow()
        record_key = OTPService._record_key(email, purpose)

        try:
            async with OTPService._lock:
                OTPService._purge_expired_locked(now)
                record = OTPService._records.get(record_key)

                if not record:
                    return False, "Invalid or expired OTP code."

                if now >= record["expires_at"]:
                    OTPService._records.pop(record_key, None)
                    return False, "OTP has expired. Please request a new one."

                record["attempts"] += 1
                if record["attempts"] > OTPService.MAX_ATTEMPTS:
                    OTPService._records.pop(record_key, None)
                    return False, "Too many failed attempts. Please request a new OTP."

                expected_hash = OTPService._hash_code(code)
                if not hmac.compare_digest(record["code_hash"], expected_hash):
                    remaining = OTPService.MAX_ATTEMPTS - record["attempts"]
                    if remaining <= 0:
                        OTPService._records.pop(record_key, None)
                        return (
                            False,
                            "Too many failed attempts. Please request a new OTP.",
                        )
                    return False, f"Invalid OTP code. {remaining} attempt(s) remaining."

                OTPService._records.pop(record_key, None)

            logger.info(f"OTP verified (in-memory only) - email={email} purpose={purpose}")
            return True, "OTP verified successfully."

        except Exception as exc:
            logger.error(f"verify_otp error for {email}: {exc}")
            return False, "An error occurred during verification. Please try again."

    @staticmethod
    async def cleanup_expired_otps() -> int:
        """Manually purge expired in-memory OTP records."""
        now = datetime.utcnow()
        async with OTPService._lock:
            deleted = OTPService._purge_expired_locked(now)
        if deleted:
            logger.info(f"Manually cleaned up {deleted} expired OTP records")
        return deleted
