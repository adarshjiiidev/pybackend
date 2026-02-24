"""
OTP (One-Time Password) service for authentication.
"""

import random
import string
import logging
from datetime import datetime, timedelta
from typing import Optional
from ..models import OTP
from .email_service import EmailService
from ..auth.cache import otp_cache

logger = logging.getLogger(__name__)


class OTPService:
    """Service for generating and validating OTPs."""
    
    OTP_LENGTH = 6
    OTP_EXPIRY_MINUTES = 10
    MAX_ATTEMPTS = 5
    RATE_LIMIT_MINUTES = 15
    MAX_OTPS_PER_PERIOD = 3
    
    @staticmethod
    def generate_otp_code() -> str:
        """Generate a 6-digit OTP code."""
        return ''.join(random.choices(string.digits, k=OTPService.OTP_LENGTH))
    
    @staticmethod
    async def check_rate_limit(email: str) -> bool:
        """
        Check if user has exceeded OTP request rate limit.
        Uses high-speed cache for rate limiting.
        """
        cache_key = f"otp_limit:{email}"
        requests = otp_cache.get(cache_key) or []
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=OTPService.RATE_LIMIT_MINUTES)
        # Filter requests within window
        requests = [r for r in requests if r > cutoff_time]
        
        if len(requests) >= OTPService.MAX_OTPS_PER_PERIOD:
            return False

        return True

    @staticmethod
    def _update_rate_limit_cache(email: str):
        """Update rate limit cache with new request."""
        cache_key = f"otp_limit:{email}"
        requests = otp_cache.get(cache_key) or []
        requests.append(datetime.utcnow())
        otp_cache.set(cache_key, requests, ttl_seconds=OTPService.RATE_LIMIT_MINUTES * 60)
    
    @staticmethod
    async def create_and_send_otp(
        email: str,
        purpose: str = "registration"
    ) -> tuple[bool, str]:
        """
        Create a new OTP and send it via email.
        
        Returns:
            (success, message) tuple
        """
        try:
            # Check rate limit
            if not await OTPService.check_rate_limit(email):
                return False, "Too many OTP requests. Please try again in 15 minutes."
            
            # Invalidate any existing unused OTPs for this email and purpose
            await OTP.find(
                OTP.email == email,
                OTP.purpose == purpose,
                OTP.is_used == False
            ).update({"$set": {"is_used": True}})
            
            # Generate new OTP
            otp_code = OTPService.generate_otp_code()
            expires_at = datetime.utcnow() + timedelta(minutes=OTPService.OTP_EXPIRY_MINUTES)
            
            # Save to database
            otp = OTP(
                email=email,
                code=otp_code,
                purpose=purpose,
                expires_at=expires_at
            )
            await otp.insert()
            
            # Update cache
            OTPService._update_rate_limit_cache(email)
            otp_cache.set(f"otp_verify:{email}:{purpose}", otp, ttl_seconds=OTPService.OTP_EXPIRY_MINUTES * 60)

            # Send email
            email_sent = await EmailService.send_otp_email(email, otp_code, purpose)
            
            if not email_sent:
                logger.warning(f"Failed to send OTP email to {email}")
                return False, "Failed to send OTP email. Please try again."
            
            logger.info(f"OTP created and sent for {email} (purpose: {purpose})")
            return True, "OTP sent successfully"
            
        except Exception as e:
            logger.error(f"Error creating OTP for {email}: {e}")
            return False, "An error occurred. Please try again."
    
    @staticmethod
    async def verify_otp(
        email: str,
        code: str,
        purpose: str = "registration"
    ) -> tuple[bool, str]:
        """
        Verify an OTP code.
        Optimized with cache for instant verification.
        
        Returns:
            (success, message) tuple
        """
        try:
            # 1. Try Cache first
            cache_key = f"otp_verify:{email}:{purpose}"
            otp = otp_cache.get(cache_key)

            # 2. Fallback to DB
            if not otp or otp.code != code:
                otp = await OTP.find_one(
                    OTP.email == email,
                    OTP.code == code,
                    OTP.purpose == purpose,
                    OTP.is_used == False
                )
            
            if not otp:
                return False, "Invalid or expired OTP code"
            
            # Increment attempts
            otp.attempts += 1
            await otp.save()
            
            # Check if too many attempts
            if otp.attempts > OTPService.MAX_ATTEMPTS:
                otp.is_used = True
                await otp.save()
                return False, "Too many verification attempts. Please request a new OTP."
            
            # Check if expired
            if otp.is_expired():
                otp.is_used = True
                await otp.save()
                return False, "OTP has expired. Please request a new one."
            
            # Valid OTP - mark as used
            otp.is_used = True
            await otp.save()
            otp_cache.delete(cache_key) # Invalidate cache
            
            logger.info(f"OTP verified successfully for {email} (purpose: {purpose})")
            return True, "OTP verified successfully"
            
        except Exception as e:
            logger.error(f"Error verifying OTP for {email}: {e}")
            return False, "An error occurred during verification"
    
    @staticmethod
    async def cleanup_expired_otps():
        """Clean up expired OTPs from database (run periodically)."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            result = await OTP.find(
                OTP.created_at < cutoff_time
            ).delete()
            
            logger.info(f"Cleaned up {result.deleted_count} expired OTPs")
            
        except Exception as e:
            logger.error(f"Error cleaning up OTPs: {e}")
