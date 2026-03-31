"""
QuickReplyAgent — Fires BEFORE the router.

Handles purely conversational messages (greetings, thanks, small talk)
with an instant one-liner reply. No KB, no search, no deep reasoning.

If the message is NOT conversational, returns state unchanged so the normal
router → agent pipeline runs as usual.
"""

import asyncio
import logging
import re

from ..models.agent_state import AgentState, AgentMode

logger = logging.getLogger(__name__)

# ── Conversational patterns ────────────────────────────────────────────────────
# Short pure-greeting phrases → match whole (stripped) query
_QUICK_PHRASES: set[str] = {
    "hi", "hii", "hiii", "hiiii", "hello", "hey", "yo", "sup",
    "namaste", "namaskar", "hola", "hye", "helo",
    "thanks", "thank you", "thx", "ty", "thank u",
    "ok", "okay", "k", "cool", "great", "nice", "wow", "awesome", "noted",
    "bye", "goodbye", "gn", "good night", "cya", "see ya", "later",
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "how r u", "how are u", "how ru",
    "what's up", "whats up", "wassup", "wsp",
    "you there", "u there", "you there?",
}

# Short regex for stretched greetings (hiiii, heyyyy, hellloooo etc.)
_GREETING_RE = re.compile(
    r"^(h+[aeiou]+[ley]*[oy]*|namaste|hola|howdy|yo+|sup)\W*$",
    re.IGNORECASE,
)

# Finance markers — if ANY of these appear, it's NOT small talk
_FINANCE_MARKERS = {
    "nifty", "sensex", "banknifty", "bank nifty", "stock", "market",
    "price", "rsi", "macd", "sip", "mutual", "crypto", "bitcoin",
    "option", "future", "call", "put", "strike", "expiry", "trade",
    "invest", "portfolio", "analyse", "analyze", "research", "analysis",
    "wtb", "wtt", "soc", "coa", "eor", "eos", "ltp", "pcr", "vix",
    "weekly", "range", "support", "resistance", "bullish", "bearish",
    "chart", "indicator", "candle", "volume", "open interest",
}

_REPLIES = [
    "Hey! 👋 How can I help you with markets or investing today?",
    "Hello! 😊 Ask me anything about stocks, Nifty, or financial concepts.",
    "Hi there! Ready to dive into markets or any trading concept. 📈",
    "Hey! What's on your mind — stocks, options, or something else? 🙌",
    "Hello! 😊 I'm here to help with anything finance-related. Fire away!",
]


def _is_conversational(query: str) -> bool:
    """Return True only when the message is pure small talk with no finance intent."""
    q = query.strip().lower()
    if not q:
        return True

    # Finance markers override everything → not conversational
    if any(marker in q for marker in _FINANCE_MARKERS):
        return False

    # Exact phrase match
    q_clean = re.sub(r"[^a-z0-9\s]", " ", q).strip()
    q_clean = re.sub(r"\s+", " ", q_clean)
    if q_clean in _QUICK_PHRASES:
        return True

    # Regex match for stretched greetings
    if _GREETING_RE.match(q_clean):
        return True

    # Very short (≤3 words) with no finance terms
    tokens = q_clean.split()
    if len(tokens) <= 3:
        greeting_words = {"hi", "hey", "hello", "yo", "hola", "namaste"}
        if tokens and tokens[0] in greeting_words:
            return True

    return False


class QuickReplyAgent:
    """
    Pre-router agent. Checks if the query is pure small talk.
    If yes → replies instantly and sets skip flags.
    If no  → passes state through unchanged.
    """

    async def handle(self, state: AgentState) -> AgentState:
        query = state.get("query", "")

        if not _is_conversational(query):
            # Not small talk — let the router handle it
            return state

        import random
        reply = random.choice(_REPLIES)

        # Build a personalised reply using a tiny fast model, with a hard token cap
        try:
            from ..config.key_rotator import get_groq_client
            groq = get_groq_client()
            resp = await asyncio.wait_for(
                groq.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Daddy's AI, a warm Indian financial assistant. "
                                "The user sent a greeting or small talk message. "
                                "Reply in exactly 1 short sentence. "
                                "Be friendly. Mention you're ready to help with markets or finance. "
                                "No bullet points. No markdown. No long paragraphs. "
                                "Example: \"Hey! 😊 Ask me anything about Nifty, stocks, or options!\""
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.9,
                    max_tokens=60,
                ),
                timeout=8.0,
            )
            reply = (resp.choices[0].message.content or reply).strip()
        except Exception as e:
            logger.debug(f"QuickReply LLM skipped ({e}), using static reply")

        state["final_response"] = reply
        state["selected_mode"] = AgentMode.EXPLAINER.value
        state["enable_research_loop"] = False
        state["use_kb"] = False
        state["is_conversational"] = True
        state["execution_metadata"] = {
            "agent": "quick_reply",
            "mode": "conversational",
            "skip_verifier": True,
        }
        logger.info(f"💬 QuickReply → instant reply for: {query[:50]!r}")
        return state
