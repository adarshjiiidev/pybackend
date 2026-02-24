"""
Production Authentication API with OTP verification.
Handles OTP-based registration, login, and verification.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
import logging
import secrets
from authlib.integrations.starlette_client import OAuth
from ..config import settings

from ..models import User, OTP
from ..auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    REFRESH_TOKEN_EXPIRE_DAYS
)
from ..services.otp_service import OTPService
from ..services.email_service import EmailService

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
            detail=f"Invalid purpose. Must be one of: {', '.join(valid_purposes)}"
        )
    
    # For registration, check if email is already registered
    if request.purpose == "registration":
        existing_user = await User.find_one(User.email == request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # For login, check if user exists
    if request.purpose == "login":
        user = await User.find_one(User.email == request.email)
        if not user:
            # Don't reveal if email doesn't exist for security
            return {
                "message": "If the email is registered, an OTP has been sent",
                "success": True
            }
    
    # Create and send OTP
    success, message = await OTPService.create_and_send_otp(
        request.email,
        request.purpose
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {
        "message": "OTP sent successfully to your email",
        "success": True
    }


@router.post("/verify-otp", response_model=MessageResponse)
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP code (doesn't complete login/registration, just verifies code).
    """
    success, message = await OTPService.verify_otp(
        request.email,
        request.code,
        request.purpose
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {
        "message": message,
        "success": True
    }


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: CompleteRegistrationRequest, background_tasks: BackgroundTasks):
    """
    Complete registration after OTP verification.
    Verifies OTP, creates user account, sends welcome email, and returns JWT tokens.
    """
    # Verify OTP first
    otp_valid, otp_message = await OTPService.verify_otp(
        request.email,
        request.otp_code,
        "registration"
    )
    
    if not otp_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=otp_message
        )
    
    # Check if user already exists (double check)
    existing_user = await User.find_one(User.email == request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate password strength
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    
    # Create user (verified since they passed OTP)
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        is_verified=True  # Auto-verified via OTP
    )
    await user.insert()
    
    logger.info(f"New user registered via OTP: {user.email}")
    
    # Send welcome email (Bolt: ⚡ Background task for faster registration)
    background_tasks.add_task(
        EmailService.send_welcome_email,
        user.email,
        user.full_name or user.email.split('@')[0]
    )
    
    # Generate JWT tokens
    access_token = create_access_token({"sub": user.user_id, "email": user.email})
    refresh_token = create_access_token(
        {"sub": user.user_id},
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
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
            "storage_percentage": user.get_storage_percentage()
        }
    }


async def _update_last_login(user_id: str):
    """Background task to update last login timestamp."""
    user = await User.find_one(User.user_id == user_id)
    if user:
        user.last_login = datetime.utcnow()
        await user.save()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, background_tasks: BackgroundTasks):
    """
    Login with email and password.
    Returns JWT tokens on successful authentication.
    """
    # Find user
    user = await User.find_one(User.email == request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if password exists (for legacy users)
    if not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please reset your password or sign up again"
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been deactivated"
        )
    
    # Update last login timestamp (Bolt: ⚡ Background task for faster response)
    background_tasks.add_task(_update_last_login, user.user_id)
    
    logger.info(f"User logged in via password: {user.email}")
    
    # Generate JWT tokens
    access_token = create_access_token({"sub": user.user_id, "email": user.email})
    refresh_token = create_access_token(
        {"sub": user.user_id},
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
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
            "storage_percentage": user.get_storage_percentage()
        }
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
        "created_at": user.created_at
    }


@router.post("/logout")
async def logout(user: User = Depends(get_current_user)):
    """Logout user."""
    logger.info(f"User logged out: {user.email}")
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token."""
    from ..auth.security import verify_token
    
    payload = verify_token(refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    user = await User.find_one(User.user_id == user_id)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user"
        )
    
    # Generate new tokens
    access_token = create_access_token({"sub": user.user_id, "email": user.email})
    new_refresh_token = create_access_token(
        {"sub": user.user_id},
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "is_verified": user.is_verified
        }
    }


# OAuth Configuration
oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Store for OAuth state (in production, use Redis)
oauth_states = {}


@router.get("/google")
async def google_auth(request: Request):
    """
    Initiate Google OAuth flow.
    Redirects user to Google consent screen.
    """
    # Generate random state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = True
    
    # Build redirect URI
    redirect_uri = request.url_for('google_callback')
    
    logger.info(f"Initiating Google OAuth with redirect: {redirect_uri}")
    
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)


@router.get("/callback/google")
async def google_callback(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Google OAuth callback.
    Exchanges code for token, creates/logs in user, and redirects to frontend.
    """
    try:
        # Verify state parameter
        state = request.query_params.get('state')
        if not state or state not in oauth_states:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )
        
        # Remove used state
        oauth_states.pop(state, None)
        
        # Exchange authorization code for access token
        token = await oauth.google.authorize_access_token(request)
        
        # Get user info from Google
        user_info = token.get('userinfo')
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user information from Google"
            )
        
        email = user_info.get('email')
        full_name = user_info.get('name')
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        logger.info(f"Google OAuth callback for email: {email}")
        
        # Find or create user
        user = await User.find_one(User.email == email)
        
        if not user:
            # Create new user from Google account
            user = User(
                email=email,
                full_name=full_name,
                is_verified=True,  # Verified by Google
                password_hash=None  # OAuth user, no password
            )
            await user.save()
            logger.info(f"Created new user via Google OAuth: {email}")
        else:
            # Update existing user info
            if full_name and not user.full_name:
                user.full_name = full_name
            user.is_verified = True
            # Update last login in background
            background_tasks.add_task(_update_last_login, user.user_id)
            logger.info(f"Logged in existing user via Google OAuth: {email}")
        
        # Generate JWT tokens
        access_token = create_access_token({"sub": user.user_id, "email": user.email})
        refresh_token = create_access_token(
            {"sub": user.user_id},
            expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # Redirect to frontend with tokens
        callback_url = f"{settings.frontend_url}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
        
        return RedirectResponse(url=callback_url)
        
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        # Redirect to frontend with error
        return RedirectResponse(url=f"{settings.frontend_url}/?error=oauth_failed")
