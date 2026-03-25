"""
Production Authentication API with OTP verification.
Handles OTP-based registration, login, and verification.
"""

import asyncio
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr

from ..auth.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    TOKEN_TYPE_REFRESH,
    blacklist_token,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_current_user_with_token,
    hash_password,
    verify_password,
)
from ..config import settings
from ..models import User
from ..models.db_models import LoginAttempt
from ..services.email_service import EmailService
from ..services.otp_service import OTPService
from ..utils.sanitizer import SanitizationError, sanitize_query_param

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


# Request/Response Models
class SendOTPRequest(BaseModel):
    email: EmailStr
    purpose: str = "registration"  # registration, login, password_reset


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    code: str
    purpose: str = "registration"


class CompleteRegistrationRequest(BaseModel):
    email: EmailStr
    otp_code: str
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    is_verified: bool
    is_premium: bool
    storage_used_mb: float
    storage_limit_mb: float
    storage_percentage: float
    created_at: datetime


class MessageResponse(BaseModel):
    message: str
    success: bool = True


# OTP Endpoints
@router.post("/send-otp", response_model=MessageResponse)
async def send_otp(request: SendOTPRequest):
    """
    Send OTP to email for verification.
    Purpose can be: registration, login, password_reset
    """
    # Validate purpose
    valid_purposes = ["registration", "login", "password_reset"]
    if request.purpose not in valid_purposes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid purpose. Must be one of: {', '.join(valid_purposes)}",
        )

    # For registration, check if email is already registered
    if request.purpose == "registration":
        existing_user = await User.find_one(User.email == request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    # For login, check if user exists
    if request.purpose == "login":
        user = await User.find_one(User.email == request.email)
        if not user:
            # Don't reveal if email doesn't exist for security
            return {
                "message": "If the email is registered, an OTP has been sent",
                "success": True,
            }

    # Create and send OTP
    success, message = await OTPService.create_and_send_otp(
        request.email, request.purpose
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    return {"message": "OTP sent successfully to your email", "success": True}


@router.post("/verify-otp", response_model=MessageResponse)
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP code (doesn't complete login/registration, just verifies code).
    """
    success, message = await OTPService.verify_otp(
        request.email, request.code, request.purpose
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    return {"message": message, "success": True}


@router.post(
    "/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
async def register(request: CompleteRegistrationRequest):
    """
    Complete registration after OTP verification.
    Verifies OTP, creates user account, sends welcome email, and returns JWT tokens.
    """
    # Verify OTP first
    otp_valid, otp_message = await OTPService.verify_otp(
        request.email, request.otp_code, "registration"
    )

    if not otp_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=otp_message)

    # Check if user already exists (double check)
    existing_user = await User.find_one(User.email == request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Sanitize full_name to prevent injection in stored data
    if request.full_name:
        try:
            request.full_name = sanitize_query_param(
                request.full_name, field="full_name"
            )
        except SanitizationError as se:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid name", "reason": se.reason},
            )

    # Validate password strength
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters",
        )

    # Prevent injection patterns in password field (passwords are hashed,
    # but raw value still passes through logs on failure paths)
    try:
        sanitize_query_param(request.password, field="password")
    except SanitizationError as se:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid password",
                "reason": "Password contains disallowed characters",
            },
        )

    # Create user (verified since they passed OTP)
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        is_verified=True,  # Auto-verified via OTP
    )
    await user.insert()

    logger.info(f"New user registered via OTP: {user.email}")

    # Send welcome email (async, don't wait)
    try:
        await EmailService.send_welcome_email(
            user.email, user.full_name or user.email.split("@")[0]
        )
    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")

    # Generate JWT tokens
    access_token = create_access_token({"sub": user.user_id, "email": user.email})
    refresh_token = create_refresh_token({"sub": user.user_id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "is_verified": user.is_verified,
            "is_premium": user.is_premium,
            "storage_used_mb": user.get_storage_used_mb(),
            "storage_limit_mb": user.get_storage_limit_mb(),
            "storage_percentage": user.get_storage_percentage(),
        },
    }


# ── Brute-force helpers ───────────────────────────────────────────────────────


async def _check_brute_force(email: str, ip: str | None = None) -> None:
    """
    Raise HTTP 429 if the email has exceeded the failed-login threshold.

    Reads the LoginAttempt collection (MongoDB) so the check works across
    all Uvicorn workers.  The TTL index on attempt_at auto-cleans old records.
    """
    window_start = datetime.utcnow() - timedelta(minutes=LoginAttempt.WINDOW_MINUTES)
    recent_failures = await LoginAttempt.find(
        LoginAttempt.email == email.lower(),
        LoginAttempt.attempt_at >= window_start,
    ).count()

    if recent_failures >= LoginAttempt.MAX_ATTEMPTS:
        logger.warning(
            f"🔒 Brute-force lockout triggered for {email} "
            f"({recent_failures} failures in {LoginAttempt.WINDOW_MINUTES} min)"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many failed login attempts. "
                f"Please wait {LoginAttempt.LOCKOUT_MINUTES} minutes before trying again."
            ),
            headers={"Retry-After": str(LoginAttempt.LOCKOUT_MINUTES * 60)},
        )


async def _record_failed_login(email: str, ip: str | None = None) -> None:
    """Persist a failed login attempt to MongoDB for brute-force tracking."""
    try:
        attempt = LoginAttempt(email=email.lower(), ip_address=ip)
        await attempt.insert()
    except Exception as exc:
        logger.warning(f"Failed to record login attempt for {email}: {exc}")


async def _clear_failed_logins(email: str) -> None:
    """
    Delete all LoginAttempt records for an email after a *successful* login.
    This prevents a valid user from being locked out after a forgotten-password
    sequence.
    """
    try:
        await LoginAttempt.find(LoginAttempt.email == email.lower()).delete()
    except Exception as exc:
        logger.warning(f"Failed to clear login attempts for {email}: {exc}")


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Login with email and password.

    Brute-force protection: after MAX_ATTEMPTS failures within WINDOW_MINUTES,
    the endpoint returns 429 until the window expires (tracked in MongoDB so
    it works across all workers).
    """
    email_lower = request.email.strip().lower()
    client_ip: str | None = None
    if http_request:
        forwarded = http_request.headers.get("X-Forwarded-For")
        client_ip = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (http_request.client.host if http_request.client else None)
        )

    # ── Brute-force check (before any DB user lookup to avoid user enumeration) ──
    await _check_brute_force(email_lower, client_ip)

    # ── Find user ──────────────────────────────────────────────────────────────
    user = await User.find_one(User.email == email_lower)
    if not user:
        # Record failure even for unknown emails (prevents timing-based enumeration)
        await _record_failed_login(email_lower, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Check if password exists (for legacy / OAuth-only users)
    if not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please reset your password or sign up again",
        )

    # ── Verify password ────────────────────────────────────────────────────────
    if not verify_password(request.password, user.password_hash):
        await _record_failed_login(email_lower, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account has been deactivated"
        )

    # ── Success: clear failure history, update last_login ──────────────────────
    await _clear_failed_logins(email_lower)
    user.last_login = datetime.utcnow()
    await user.save()

    logger.info(f"User logged in via password: {user.email} from {client_ip}")

    # ── Generate JWT tokens ────────────────────────────────────────────────────
    access_token = create_access_token({"sub": user.user_id, "email": user.email})
    refresh_token = create_refresh_token({"sub": user.user_id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "is_verified": user.is_verified,
            "is_premium": user.is_premium,
            "storage_used_mb": user.get_storage_used_mb(),
            "storage_limit_mb": user.get_storage_limit_mb(),
            "storage_percentage": user.get_storage_percentage(),
        },
    }


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return {
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "is_verified": user.is_verified,
        "is_premium": user.is_premium,
        "storage_used_mb": user.get_storage_used_mb(),
        "storage_limit_mb": user.get_storage_limit_mb(),
        "storage_percentage": user.get_storage_percentage(),
        "created_at": user.created_at,
    }


@router.post("/logout")
async def logout(
    auth_info: tuple = Depends(get_current_user_with_token),
):
    """
    Logout the current user.

    Blacklists the presented JWT in MongoDB so it cannot be reused even
    before its natural expiry.  Subsequent requests with the same token
    will receive HTTP 401 "Token has been revoked".
    """
    user, jti, expires_at = auth_info
    if jti:
        await blacklist_token(jti, expires_at)
        logger.info(
            f"🔒 User logged out: {user.email} — "
            f"jti={jti[:8]}… blacklisted until {expires_at.isoformat()}"
        )
    else:
        logger.warning(f"Logout for {user.email}: token had no jti — cannot blacklist")
    return {"message": "Logged out successfully"}


# ── Password Reset ──────────────────────────────────────────────────────


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request: ResetPasswordRequest):
    """
    Complete a password reset after OTP verification.

    Flow:
      1. User calls POST /auth/send-otp with purpose=password_reset.
      2. User enters OTP received by email.
      3. This endpoint verifies the OTP, hashes the new password, and updates the user.
    """
    # Validate new password strength
    if len(request.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters",
        )

    # Verify OTP
    otp_valid, otp_message = await OTPService.verify_otp(
        request.email, request.otp_code, "password_reset"
    )
    if not otp_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=otp_message
        )

    # Find user
    user = await User.find_one(User.email == request.email.strip().lower())
    if not user:
        # Keep error generic to avoid email enumeration
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset failed — please request a new OTP",
        )

    # Hash and save new password
    user.password_hash = hash_password(request.new_password)
    user.updated_at = datetime.utcnow()
    await user.save()

    # Invalidate user cache so next request reads fresh data
    from ..auth.security import invalidate_user_cache
    invalidate_user_cache(user.user_id)

    logger.info(f"✅ Password reset completed for: {user.email}")
    return {"message": "Password reset successfully", "success": True}



@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Exchange a valid refresh token for a new access + refresh token pair.

    Validates:
      • JWT signature & expiry
      • token_type == "refresh"  (access tokens are rejected)
      • User exists and is active
    """
    from ..auth.security import _is_token_blacklisted, verify_token

    payload = verify_token(refresh_token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    # ── Token-type guard: only accept refresh tokens here ─────────────────────
    token_type = payload.get("token_type")
    if token_type and token_type != TOKEN_TYPE_REFRESH:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="An access token cannot be used to refresh. Please provide a refresh token.",
        )

    # ── Blacklist check (covers logged-out refresh tokens too) ─────────────────
    jti = payload.get("jti")
    if jti and await _is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )

    user_id = payload.get("sub")
    user = await User.find_one(User.user_id == user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user"
        )

    # ── Rotate tokens: blacklist the old refresh token, issue new pair ─────────
    if jti:
        exp_epoch = payload.get("exp", 0)
        old_expires = (
            datetime.utcfromtimestamp(exp_epoch)
            if exp_epoch
            else datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )
        await blacklist_token(jti, old_expires)

    new_access_token = create_access_token({"sub": user.user_id, "email": user.email})
    new_refresh_token = create_refresh_token({"sub": user.user_id})

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "is_verified": user.is_verified,
        },
    }


# ── OAuth Configuration ────────────────────────────────────────────────────
# Endpoints hardcoded to skip the ~1s metadata fetch on every request
oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
    client_kwargs={"scope": "openid email profile"},
)

# ── Stateless CSRF State Helpers ─────────────────────────────────────────────
# The OAuth state is a self-contained HMAC-signed token: "<nonce>.<timestamp>.<sig>"
# No server-side storage needed — survives reloads, restarts, and multi-process deploys.
import hashlib
import hmac

_OAUTH_STATE_TTL = 600  # 10 minutes — matches Google's auth code expiry
_STATE_SECRET = os.environ.get(
    "SESSION_SECRET_KEY", "change-me-in-production-must-be-32-chars-min-xxxxxxxxxxx"
).encode()  # reuses existing env var


def _make_oauth_state() -> str:
    """Create a short-lived HMAC-signed state token."""
    nonce = secrets.token_urlsafe(16)
    ts = str(
        int(time.monotonic())
    )  # relative timestamp (not wall clock, avoids clock skew)
    payload = f"{nonce}.{ts}"
    sig = hmac.new(_STATE_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}.{sig}"


def _verify_oauth_state(state: str) -> bool:
    """Verify state signature and expiry. Returns True if valid."""
    try:
        parts = state.split(".")
        if len(parts) != 3:
            return False
        nonce, ts_str, received_sig = parts
        payload = f"{nonce}.{ts_str}"
        expected_sig = hmac.new(
            _STATE_SECRET, payload.encode(), hashlib.sha256
        ).hexdigest()[:16]
        # Constant-time comparison prevents timing attacks
        if not hmac.compare_digest(received_sig, expected_sig):
            return False
        # Check expiry
        age = time.monotonic() - float(ts_str)
        return 0 <= age <= _OAUTH_STATE_TTL
    except Exception:
        return False


@router.get("/google")
async def google_auth(request: Request):
    """
    Initiate Google OAuth flow.
    Redirects user to Google consent screen.
    """
    # Stateless signed state — no server-side storage required
    state = _make_oauth_state()
    redirect_uri = request.url_for("google_callback")
    logger.info(f"Initiating Google OAuth — redirect: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)


@router.get("/callback/google")
async def google_callback(request: Request):
    """
    Handle Google OAuth callback.
    Verifies: CSRF state, auth code exchange, and Google id_token signature via tokeninfo.
    """
    try:
        # ── CSRF state verification — reject tampered/replayed flows ──────────
        state = request.query_params.get("state", "")
        if not _verify_oauth_state(state):
            logger.warning("Google OAuth callback: invalid or expired state parameter")
            return RedirectResponse(
                url=f"{settings.frontend_url}/?error=invalid_state"
            )

        code = request.query_params.get("code")
        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing authorization code",
            )

        async with httpx.AsyncClient(timeout=10) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": str(request.url_for("google_callback")),
                    "grant_type": "authorization_code",
                },
            )

        if token_resp.status_code != 200:
            logger.error(f"Google token exchange failed: {token_resp.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange authorization code",
            )

        token_data = token_resp.json()
        id_token_str = token_data.get("id_token")
        if not id_token_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No ID token from Google",
            )

        # ── Verify id_token via Google's tokeninfo endpoint ─────────────────
        # This validates: signature, expiry, issuer, and audience.
        # Never decode manually — that bypasses all cryptographic verification.
        import httpx
        async with httpx.AsyncClient(timeout=8) as verify_client:
            verify_resp = await verify_client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": id_token_str},
            )

        if verify_resp.status_code != 200:
            logger.error(f"Google tokeninfo verification failed: {verify_resp.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to verify Google identity token",
            )

        user_info = verify_resp.json()

        # Confirm audience matches our client ID (prevents token substitution attacks)
        if user_info.get("aud") != settings.google_client_id:
            logger.error(
                f"id_token aud mismatch: got {user_info.get('aud')!r}, "
                f"expected {settings.google_client_id!r}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Identity token audience mismatch",
            )

        email = user_info.get("email")
        full_name = user_info.get("name")
        email_verified = user_info.get("email_verified") in ("true", True)
        if not email or not email_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verified email not provided by Google",
            )

        logger.info(f"Google OAuth callback: {email}")

        # Find or create user
        user = await User.find_one(
            User.email == email
        )  # Needs index on email (see database.py)

        if not user:
            # New user — insert once, no extra save()
            user = User(
                email=email,
                full_name=full_name,
                is_verified=True,
                password_hash=None,
            )
            await user.insert()
            logger.info(f"New user via Google OAuth: {email}")
        else:
            # Existing user — update last_login FIRE-AND-FORGET (don't block the redirect)
            async def _update_login(u):
                try:
                    if full_name and not u.full_name:
                        u.full_name = full_name
                    u.is_verified = True
                    u.last_login = datetime.utcnow()
                    await u.save()
                except Exception as ex:
                    logger.warning(f"last_login update failed (non-critical): {ex}")

            asyncio.create_task(_update_login(user))
            logger.info(f"Existing user logged in via Google: {email}")

        # Generate JWT tokens (CPU-only, instant)
        access_token = create_access_token({"sub": user.user_id, "email": user.email})
        refresh_token = create_refresh_token({"sub": user.user_id})

        callback_url = (
            f"{settings.frontend_url}/auth/callback"
            f"?access_token={access_token}&refresh_token={refresh_token}"
        )
        return RedirectResponse(url=callback_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return RedirectResponse(url=f"{settings.frontend_url}/?error=oauth_failed")
