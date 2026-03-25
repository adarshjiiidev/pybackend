"""
Input Sanitization Module
Blocks SQL injection, shell injection, binary data, prompt injection,
and other malicious inputs before they reach the AI pipeline or database.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL injection patterns
# ---------------------------------------------------------------------------
SQL_PATTERNS = [
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b.*\b(FROM|INTO|TABLE|DATABASE|WHERE)\b",
    r"(?i)(--|\#|\/\*|\*\/)",          # SQL comment syntax
    r"(?i)\bUNION\b.*\bSELECT\b",
    r"(?i)\bOR\b\s+\d+\s*=\s*\d+",    # OR 1=1
    r"(?i)\bAND\b\s+\d+\s*=\s*\d+",   # AND 1=1
    r"(?i)\bxp_cmdshell\b",
    r"(?i)\bINFORMATION_SCHEMA\b",
    r"(?i)\bsys\.\w+\b",               # sys.tables, sys.columns etc.
    r"';",                              # SQL string terminator
    r'";',                              # SQL string terminator with double quote
]

# ---------------------------------------------------------------------------
# Shell / command injection patterns
# ---------------------------------------------------------------------------
SHELL_PATTERNS = [
    r"(?:\||;|&|`|\$\(|\${)",           # pipe, semicolon, backtick, $(), ${}
    r"(?i)\b(eval|exec|os\.system|subprocess|popen|system)\s*\(",
    r"(?i)(\.\.\/|\.\.\\)",            # path traversal
    r"(?i)\/etc\/passwd",
    r"(?i)\/bin\/(bash|sh|zsh|dash)",
    r"(?i)(wget|curl)\s+http",
    r"(?i)base64\s+-d",                # base64 decode piped to shell
]

# ---------------------------------------------------------------------------
# Binary / null byte patterns
# ---------------------------------------------------------------------------
BINARY_PATTERNS = [
    r"\x00",                            # null byte
    r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",  # control chars outside normal range
]

# ---------------------------------------------------------------------------
# Prompt injection / jailbreak patterns (heuristic)
# ---------------------------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(previous|all|above|prior)\s+instructions?",
    r"(?i)forget\s+(everything|all|previous)",
    r"(?i)you\s+are\s+now\s+(?:a\s+)?(DAN|jailbroken|unrestricted|unethical)",
    r"(?i)act\s+as\s+(?:an?\s+)?(evil|malicious|unrestricted|DAN)",
    r"(?i)pretend\s+(you\s+are|to be)\s+(?:an?\s+)?(evil|malicious|unrestricted)",
    r"(?i)system\s*:\s*you\s+are",     # fake system prompt injection
    r"(?i)<\s*system\s*>",             # XML/HTML system tag injection
]

# Compile all patterns once at module load
_SQL_COMPILED = [re.compile(p) for p in SQL_PATTERNS]
_SHELL_COMPILED = [re.compile(p) for p in SHELL_PATTERNS]
_BINARY_COMPILED = [re.compile(p) for p in BINARY_PATTERNS]
_PROMPT_COMPILED = [re.compile(p) for p in PROMPT_INJECTION_PATTERNS]

MAX_MESSAGE_LENGTH = 10_000
MAX_QUERY_PARAM_LENGTH = 500


class SanitizationError(ValueError):
    """Raised when user input fails sanitization checks."""
    def __init__(self, reason: str, field: str = "input"):
        self.reason = reason
        self.field = field
        super().__init__(f"Input rejected [{field}]: {reason}")


def sanitize_user_message(text: str, field: str = "message") -> str:
    """
    Full sanitization pipeline for user chat messages.
    Returns the cleaned text (stripped of leading/trailing whitespace),
    or raises SanitizationError if the input looks malicious.

    Steps:
      1. Length check
      2. Binary / control-char strip (non-printable bytes except newline/tab)
      3. SQL injection detection
      4. Shell injection detection
      5. Prompt injection heuristics
    """
    if not isinstance(text, str):
        raise SanitizationError("Input must be a string", field)

    # 1. Length guard
    if len(text) > MAX_MESSAGE_LENGTH:
        raise SanitizationError(
            f"Message too long ({len(text)} chars, max {MAX_MESSAGE_LENGTH})", field
        )

    # 2. Remove binary / control characters
    text = _strip_control_chars(text)

    if not text.strip():
        raise SanitizationError("Message is empty after sanitization", field)

    # 3. SQL injection check
    _check_patterns(text, _SQL_COMPILED, "SQL injection", field)

    # 4. Shell injection check
    _check_patterns(text, _SHELL_COMPILED, "shell injection", field)

    # 5. Prompt injection (log a warning but also block to protect the system)
    _check_patterns(text, _PROMPT_COMPILED, "prompt injection attempt", field)

    return text.strip()


def sanitize_query_param(value: str, field: str = "query") -> str:
    """
    Lighter sanitization for URL query parameters, search terms, symbols, etc.
    Blocks SQL/shell injection, strips control chars, enforces short length.
    """
    if not isinstance(value, str):
        raise SanitizationError("Parameter must be a string", field)

    if len(value) > MAX_QUERY_PARAM_LENGTH:
        raise SanitizationError(
            f"Parameter too long ({len(value)} chars, max {MAX_QUERY_PARAM_LENGTH})", field
        )

    value = _strip_control_chars(value)
    _check_patterns(value, _SQL_COMPILED, "SQL injection", field)
    _check_patterns(value, _SHELL_COMPILED, "shell injection", field)

    return value.strip()


def sanitize_symbol(symbol: str) -> str:
    """
    Strict whitelist for stock/index symbols: only uppercase letters, digits, hyphens, dots.
    E.g. RELIANCE, NIFTY, TCS, HDFC-BANK, NIFTY50.
    """
    if not isinstance(symbol, str):
        raise SanitizationError("Symbol must be a string", "symbol")

    symbol = symbol.strip().upper()

    if not symbol:
        raise SanitizationError("Symbol cannot be empty", "symbol")

    if len(symbol) > 30:
        raise SanitizationError("Symbol too long", "symbol")

    if not re.match(r'^[A-Z0-9\-\.&]+$', symbol):
        raise SanitizationError(
            f"Symbol '{symbol}' contains invalid characters (only A-Z, 0-9, -, . allowed)", "symbol"
        )

    return symbol


def validate_email_format(email: str) -> str:
    """Basic email format check beyond Pydantic's EmailStr."""
    if not isinstance(email, str):
        raise SanitizationError("Email must be a string", "email")

    email = email.strip().lower()

    # Check for injection in email field
    _check_patterns(email, _SQL_COMPILED, "SQL injection", "email")

    # Simple regex validation
    if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
        raise SanitizationError("Invalid email format", "email")

    if len(email) > 254:  # RFC 5321 max
        raise SanitizationError("Email too long", "email")

    return email


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_control_chars(text: str) -> str:
    """Remove binary and non-printable control characters, keeping newlines and tabs."""
    # Allow: printable ASCII + \n (0x0A) + \t (0x09)
    # Remove: null bytes, other control chars, high surrogates
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return cleaned


def _check_patterns(text: str, patterns: list, attack_type: str, field: str) -> None:
    """Check text against a list of compiled regex patterns. Raises on match."""
    for pattern in patterns:
        if pattern.search(text):
            logger.warning(
                f"🚨 {attack_type} blocked in field '{field}': "
                f"matched pattern '{pattern.pattern[:60]}'"
            )
            raise SanitizationError(
                f"Input contains potentially malicious {attack_type} content", field
            )
