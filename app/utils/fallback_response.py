"""
Response fallback helpers.
Avoids static hardcoded failure strings by generating query-aware fallback text.
"""

import re
from typing import Any


_DEFINITION_PREFIX = re.compile(
    r"^(what is|what are|define|explain|meaning of|tell me about|how does)\b",
    re.IGNORECASE,
)


def _extract_term(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", (query or "").strip()).strip("?.! ")
    if not cleaned:
        return ""
    return _DEFINITION_PREFIX.sub("", cleaned).strip("?.! ") or cleaned


def _fallback_for_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "I'm Daddy's AI. Ask me anything about markets, stocks, or finance."

    q_lower = q.lower()
    if re.match(r"^(hi|hello|hey|namaste|yo|sup)\b", q_lower):
        return "Hey! I'm Daddy's AI, ready to help with markets and finance."

    if _DEFINITION_PREFIX.match(q):
        term = _extract_term(q)
        if term:
            return (
                f"I couldn't complete this turn for \"{term}\" yet, but I can still help. "
                "Send the exact term spelling once and I'll explain it using KB + web context."
            )
        return (
            "I couldn't complete this explanation in this turn, but I can still help. "
            "Send the term again and I'll explain it clearly with KB + web context."
        )

    return (
        "I hit a temporary issue while generating this reply. "
        "Please send the same message once more and I'll continue."
    )


def resolve_response_or_fallback(final_state: Any, query: str) -> str:
    """
    Resolve model output to a safe non-empty response.
    """
    if isinstance(final_state, dict):
        value = final_state.get("final_response")
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return _fallback_for_query(query)
