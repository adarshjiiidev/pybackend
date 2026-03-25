"""
OTP (One-Time Password) service — MongoDB-backed, multi-worker safe.

Key improvements over the old in-memory implementation:
  • OTPs survive server restarts and work across multiple Uvicorn workers.
  • Codes are stored as HMAC-SHA256 hashes (never plaintext at rest).
  • Code generation uses secrets.randbelow() (CSPRNG, not random.choices).
  • MongoDB TTL index auto-deletes expired records — no manual cleanup needed.
  • Rate-limit state is also stored in MongoDB so it is worker-agnostic.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from ..models.db_models import LoginAttempt, OTPRecord
from .email_service import EmailService

logger = logging.getLogger(__name__)


class OTPService:
    """Service for generating, storing, and validating OTPs via MongoDB."""

    OTP_LENGTH: int = 6
    OTP_EXPIRY_MINUTES: int = 10
    MAX_ATTEMPTS: int = 5
    RATE_LIMIT_MINUTES: int = 15
    MAX_OTPS_PER_PERIOD: int = 3

    # ── Internal HMAC secret for hashing OTP codes ─────────────────────────
    # This salt makes it impossible to reverse the stored hash even if the
    # MongoDB collection is compromised.  It does NOT need to be rotated
    # because OTPs expire in 10 minutes anyway.
    _HASH_SECRET: bytes = b"otp-hmac-salt-daddysai-v1"

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_otp_code() -> str:
        """
        Generate a cryptographically secure 6-digit OTP code.

        Uses secrets.randbelow() (CSPRNG) instead of random.choices()
        to prevent statistical guessing of generated codes.
        """
        code = secrets.randbelow(10**OTPService.OTP_LENGTH)
        return f"{code:0{OTPService.OTP_LENGTH}d}"

    @staticmethod
    async def check_rate_limit(email: str) -> bool:
        """
        Return True if the email is within the allowed OTP-request rate.

        Counts OTPRecords created in the last RATE_LIMIT_MINUTES window.
        Works across all workers because the count is read from MongoDB.
        """
        try:
            window_start = datetime.utcnow() - timedelta(
                minutes=OTPService.RATE_LIMIT_MINUTES
            )
            recent_count = await OTPRecord.find(
                OTPRecord.email == email.lower(),
                OTPRecord.created_at >= window_start,
            ).count()
            return recent_count < OTPService.MAX_OTPS_PER_PERIOD
        except Exception as exc:
            # Fail open — don't block users if DB is temporarily unavailable
            logger.error(f"OTP rate-limit check failed for {email}: {exc}")
            return True

    @staticmethod
    async def create_and_send_otp(
        email: str,
        purpose: str = "registration",
    ) -> tuple[bool, str]:
        """
        Create a new OTP record in MongoDB and send the plaintext code by email.

        Returns:
            (success: bool, message: str)
        """
        email = email.strip().lower()

        try:
            # ── Rate-limit check ───────────────────────────────────────────
            if not await OTPService.check_rate_limit(email):
                return (
                    False,
                    f"Too many OTP requests. Please wait {OTPService.RATE_LIMIT_MINUTES} "
                    "minutes before trying again.",
                )

            # ── Generate plaintext code (only ever kept in memory) ─────────
            plaintext_code = OTPService.generate_otp_code()
            code_hash = OTPRecord.hash_code(plaintext_code, OTPService._HASH_SECRET)

            expires_at = datetime.utcnow() + timedelta(
                minutes=OTPService.OTP_EXPIRY_MINUTES
            )

            # ── Upsert: replace any existing OTP for this email+purpose ────
            # This prevents the user from having multiple valid OTPs at once.
            existing = await OTPRecord.find_one(
                OTPRecord.email == email,
                OTPRecord.purpose == purpose,
            )
            if existing:
                existing.code_hash = code_hash
                existing.expires_at = expires_at
                existing.attempts = 0
                existing.created_at = datetime.utcnow()
                await existing.save()
                logger.debug(f"Replaced existing OTP for {email} / {purpose}")
            else:
                new_record = OTPRecord(
                    email=email,
                    purpose=purpose,
                    code_hash=code_hash,
                    expires_at=expires_at,
                )
                await new_record.insert()
                logger.debug(f"Created new OTP record for {email} / {purpose}")

            # ── Send the plaintext code by email ───────────────────────────
            email_sent = await EmailService.send_otp_email(
                email, plaintext_code, purpose
            )
            if not email_sent:
                # Best-effort rollback: delete the record we just wrote
                await OTPRecord.find_one(
                    OTPRecord.email == email, OTPRecord.purpose == purpose
                ).delete()
                logger.warning(
                    f"Email delivery failed for OTP to {email}; record rolled back"
                )
                return False, "Failed to send OTP email. Please try again."

            logger.info(f"OTP sent — email={email}  purpose={purpose}")
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
        Verify a submitted OTP code against the stored hash.

        On success the record is immediately deleted (one-time use).
        On failure the attempt counter is incremented; after MAX_ATTEMPTS
        the record is deleted to force the user to request a fresh OTP.

        Returns:
            (success: bool, message: str)
        """
        email = email.strip().lower()
        code = code.strip()

        try:
            record: Optional[OTPRecord] = await OTPRecord.find_one(
                OTPRecord.email == email,
                OTPRecord.purpose == purpose,
            )

            if not record:
                return False, "Invalid or expired OTP code."

            # ── Expiry check (belt-and-suspenders; TTL index handles cleanup) ─
            if record.is_expired():
                await record.delete()
                return False, "OTP has expired. Please request a new one."

            # ── Increment attempt counter first to prevent timing side-channels ─
            record.attempts += 1

            if record.attempts > OTPService.MAX_ATTEMPTS:
                await record.delete()
                return (
                    False,
                    "Too many failed attempts. Please request a new OTP.",
                )

            await record.save()  # Persist incremented counter

            # ── Constant-time code verification ───────────────────────────
            if not record.verify_code(code, OTPService._HASH_SECRET):
                remaining = OTPService.MAX_ATTEMPTS - record.attempts
                if remaining <= 0:
                    await record.delete()
                    return (
                        False,
                        "Too many failed attempts. Please request a new OTP.",
                    )
                return False, f"Invalid OTP code. {remaining} attempt(s) remaining."

            # ── Success: delete the record immediately (one-time use) ──────
            await record.delete()
            logger.info(f"OTP verified — email={email}  purpose={purpose}")
            return True, "OTP verified successfully."

        except Exception as exc:
            logger.error(f"verify_otp error for {email}: {exc}")
            return False, "An error occurred during verification. Please try again."

    # ─────────────────────────────────────────────────────────────────────────
    # Maintenance helpers (called on demand or from background tasks)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def cleanup_expired_otps() -> int:
        """
        Manually delete expired OTP records.

        NOTE: Under normal operation the MongoDB TTL index handles this
        automatically.  Call this only from a maintenance script or if the
        TTL index was accidentally dropped.

        Returns:
            Number of records deleted.
        """
        try:
            result = await OTPRecord.find(
                OTPRecord.expires_at < datetime.utcnow()
            ).delete()
            count = result.deleted_count if result else 0
            if count:
                logger.info(f"Manually cleaned up {count} expired OTP records")
            return count
        except Exception as exc:
            logger.error(f"cleanup_expired_otps error: {exc}")
            return 0
