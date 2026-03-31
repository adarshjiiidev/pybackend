"""
Payments API — Razorpay integration.

Endpoints:
  POST /payments/create-order     → Create Razorpay order, return order_id
  POST /payments/verify-payment   → Verify HMAC signature, activate subscription
  GET  /payments/subscription     → Get current user's subscription status
  GET  /payments/history          → Get user's payment history

Security:
  - Secret key is NEVER sent to client — all crypto on server.
  - Signature verified with HMAC-SHA256 before touching DB.
  - Duplicate payment prevention via unique razorpay_order_id constraint.
  - All endpoints require authenticated user (Bearer JWT).
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import razorpay
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.security import get_current_user
from ..models.db_models import (
    PaymentTransaction,
    PlanType,
    Subscription,
    SubscriptionStatus,
    User,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["payments"])

# ── Razorpay client (lazy-initialised once) ───────────────────────────────────

_razorpay_client: Optional[razorpay.Client] = None


def _get_razorpay() -> razorpay.Client:
    global _razorpay_client
    if _razorpay_client is None:
        key_id = os.getenv("RAZORPAY_KEY_ID", "")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
        if not key_id or not key_secret:
            raise HTTPException(
                status_code=503,
                detail="Payment gateway not configured. Contact support.",
            )
        _razorpay_client = razorpay.Client(auth=(key_id, key_secret))
    return _razorpay_client


# ── Plan pricing (amounts in paise — INR × 100) ───────────────────────────────

PLAN_PRICING = {
    PlanType.PRO: {
        "monthly": 49900,   # ₹499/mo
        "yearly":  449900,  # ₹4,499/yr
    },
    PlanType.PREMIUM: {
        "monthly": 149900,  # ₹1,499/mo
        "yearly":  1399900, # ₹13,999/yr
    },
}

PLAN_DURATION_DAYS = {
    "monthly": 30,
    "yearly":  365,
}


# ── Request / Response schemas ────────────────────────────────────────────────

class CreateOrderRequest(BaseModel):
    plan: PlanType = Field(..., description="pro | premium")
    is_yearly: bool = Field(default=False)


class CreateOrderResponse(BaseModel):
    order_id: str
    amount: int          # paise
    currency: str
    key_id: str          # Public key — safe to send to client
    plan: str
    is_yearly: bool


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class SubscriptionResponse(BaseModel):
    plan: str
    status: str
    is_yearly: bool
    started_at: datetime
    expires_at: Optional[datetime]
    days_remaining: Optional[int]
    is_active: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Validate Razorpay HMAC-SHA256 signature.
    Expected: HMAC_SHA256(key_secret, f"{order_id}|{payment_id}")
    """
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_secret:
        return False
    body = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(key_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _get_or_create_subscription(user_id: str) -> Subscription:
    """Fetch existing subscription or create a free-plan default."""
    sub = await Subscription.find_one(Subscription.user_id == user_id)
    if not sub:
        sub = Subscription(user_id=user_id, plan=PlanType.FREE)
        await sub.insert()
    return sub


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(
    body: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Create a Razorpay order for the selected plan.

    Returns order_id + public key_id — the frontend opens Razorpay checkout
    with these values. Secret key never leaves the server.
    """
    if body.plan == PlanType.FREE:
        raise HTTPException(
            status_code=400, detail="Cannot create an order for the free plan."
        )

    billing = "yearly" if body.is_yearly else "monthly"
    amount_paise = PLAN_PRICING[body.plan][billing]
    key_id = os.getenv("RAZORPAY_KEY_ID", "")

    # Check for existing pending order (basic idempotency guard)
    existing = await PaymentTransaction.find_one(
        PaymentTransaction.user_id == current_user.user_id,
        PaymentTransaction.plan == body.plan,
        PaymentTransaction.status == "created",
    )
    if existing:
        # Return the existing order so the user can re-open checkout
        return CreateOrderResponse(
            order_id=existing.razorpay_order_id,
            amount=existing.amount_paise,
            currency=existing.currency,
            key_id=key_id,
            plan=body.plan.value,
            is_yearly=body.is_yearly,
        )

    # Create order via Razorpay SDK
    try:
        client = _get_razorpay()
        order_data = client.order.create(
            {
                "amount": amount_paise,
                "currency": "INR",
                "receipt": f"receipt_{current_user.user_id[:8]}",
                "notes": {
                    "user_id": current_user.user_id,
                    "plan": body.plan.value,
                    "billing": billing,
                },
            }
        )
    except Exception as exc:
        logger.error(f"Razorpay order creation failed: {exc}")
        raise HTTPException(
            status_code=502, detail="Failed to create payment order. Try again."
        )

    # Persist transaction record
    txn = PaymentTransaction(
        user_id=current_user.user_id,
        plan=body.plan,
        is_yearly=body.is_yearly,
        amount_paise=amount_paise,
        currency="INR",
        razorpay_order_id=order_data["id"],
    )
    await txn.insert()

    logger.info(
        f"[payments] Order created: {order_data['id']} "
        f"user={current_user.user_id[:8]} plan={body.plan} billing={billing}"
    )

    return CreateOrderResponse(
        order_id=order_data["id"],
        amount=amount_paise,
        currency="INR",
        key_id=key_id,
        plan=body.plan.value,
        is_yearly=body.is_yearly,
    )


@router.post("/verify-payment")
async def verify_payment(
    body: VerifyPaymentRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Verify Razorpay payment signature and activate subscription.

    Flow:
      1. Validate HMAC-SHA256 signature (tamper-proof)
      2. Look up existing transaction (prevents replay attacks)
      3. Mark transaction as captured
      4. Upsert subscription with new plan + expiry
      5. Set user.is_premium = True if plan != FREE
    """
    # 1. Signature check — most critical security step
    sig_valid = _verify_signature(
        body.razorpay_order_id,
        body.razorpay_payment_id,
        body.razorpay_signature,
    )
    if not sig_valid:
        logger.warning(
            f"[payments] ⚠️ Invalid signature from user={current_user.user_id[:8]} "
            f"order={body.razorpay_order_id}"
        )
        raise HTTPException(
            status_code=400,
            detail="Payment verification failed. Signature mismatch.",
        )

    # 2. Find transaction
    txn = await PaymentTransaction.find_one(
        PaymentTransaction.razorpay_order_id == body.razorpay_order_id,
        PaymentTransaction.user_id == current_user.user_id,
    )
    if not txn:
        raise HTTPException(
            status_code=404, detail="Transaction not found."
        )

    # Replay attack: already captured
    if txn.status == "captured":
        logger.warning(
            f"[payments] Duplicate verify attempt order={body.razorpay_order_id}"
        )
        return {"success": True, "plan": txn.plan.value, "message": "Already activated"}

    # 3. Mark captured
    txn.razorpay_payment_id = body.razorpay_payment_id
    txn.razorpay_signature = body.razorpay_signature
    txn.status = "captured"
    txn.updated_at = datetime.utcnow()
    await txn.save()

    # 4. Activate subscription
    billing_days = PLAN_DURATION_DAYS["yearly" if txn.is_yearly else "monthly"]
    expires_at = datetime.utcnow() + timedelta(days=billing_days)

    sub = await _get_or_create_subscription(current_user.user_id)
    sub.plan = txn.plan
    sub.status = SubscriptionStatus.ACTIVE
    sub.is_yearly = txn.is_yearly
    sub.started_at = datetime.utcnow()
    sub.expires_at = expires_at
    sub.latest_payment_id = body.razorpay_payment_id
    sub.updated_at = datetime.utcnow()
    await sub.save()

    # 5. Flip is_premium on User document
    current_user.is_premium = txn.plan != PlanType.FREE
    current_user.updated_at = datetime.utcnow()
    await current_user.save()

    logger.info(
        f"[payments] ✅ Subscription activated: user={current_user.user_id[:8]} "
        f"plan={txn.plan} expires={expires_at.date()}"
    )

    return {
        "success": True,
        "plan": txn.plan.value,
        "expires_at": expires_at.isoformat(),
        "message": f"Successfully subscribed to {txn.plan.value.title()} plan!",
    }


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(current_user: User = Depends(get_current_user)):
    """Return the current user's subscription details."""
    sub = await _get_or_create_subscription(current_user.user_id)

    # Auto-expire if past expiry date
    if (
        sub.expires_at
        and datetime.utcnow() > sub.expires_at
        and sub.status == SubscriptionStatus.ACTIVE
    ):
        sub.status = SubscriptionStatus.EXPIRED
        sub.updated_at = datetime.utcnow()
        await sub.save()

    return SubscriptionResponse(
        plan=sub.plan.value,
        status=sub.status.value,
        is_yearly=sub.is_yearly,
        started_at=sub.started_at,
        expires_at=sub.expires_at,
        days_remaining=sub.days_remaining(),
        is_active=sub.is_active_plan(),
    )


@router.get("/history")
async def payment_history(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
):
    """Return last N payment transactions for the user."""
    txns = (
        await PaymentTransaction.find(
            PaymentTransaction.user_id == current_user.user_id,
        )
        .sort(-PaymentTransaction.created_at)
        .limit(limit)
        .to_list()
    )

    return {
        "transactions": [
            {
                "transaction_id": t.transaction_id,
                "plan": t.plan.value,
                "is_yearly": t.is_yearly,
                "amount_inr": t.amount_paise / 100,
                "status": t.status,
                "razorpay_payment_id": t.razorpay_payment_id,
                "created_at": t.created_at.isoformat(),
            }
            for t in txns
        ]
    }
