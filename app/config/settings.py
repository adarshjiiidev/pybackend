"""
Application settings using Pydantic Settings.
Dual-provider: OpenRouter (primary LLM) + Groq (web search + GPT-OSS deep reasoning).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    """Available model types for different tasks."""
    REASONING_DEEP = "reasoning_deep"  # GPT-OSS-120B for deep reasoning (Groq)
    REASONING_FAST = "reasoning_fast"  # GPT-OSS-20B for faster reasoning (Groq)
    COMPOUND = "compound"  # Compound AI with built-in tools (Groq only)
    COMPOUND_MINI = "compound_mini"  # Lightweight compound (Groq only)
    FAST = "fast"  # Quick responses, simple tasks
    ANALYSIS = "analysis"  # Data analysis, technical tasks
    CREATIVE = "creative"  # Educational content, explanations
    ROUTER = "router"  # Intent classification
    KB_RAG = "kb_rag"  # Knowledge base / RAG synthesis
    VISION = "vision"  # Vision & multimodal (Groq only)


class Settings(BaseSettings):
    """Application configuration settings optimized for Groq API."""
    
    # ── Groq API Keys (for web search + GPT-OSS deep reasoning) ──────────
    groq_api_key: str
    groq_api_key_2: Optional[str] = None
    groq_api_key_3: Optional[str] = None
    groq_api_key_4: Optional[str] = None
    groq_api_key_5: Optional[str] = None
    
    # ── OpenRouter API Keys (primary LLM provider — free models) ─────────
    openrouter_api_key: Optional[str] = None
    openrouter_api_key_2: Optional[str] = None
    openrouter_api_key_3: Optional[str] = None
    openrouter_api_key_4: Optional[str] = None
    openrouter_api_key_5: Optional[str] = None

    # ── NVIDIA NIM API Key (DeepSeek-V3.2 + other NIM models) ───────────
    nvidia_api_key: Optional[str] = None
    
    # ── Groq Model Selection (kept for web search + deep reasoning) ──────
    model_reasoning_deep: str = "openai/gpt-oss-120b"  # Deep reasoning (Groq)
    model_reasoning_fast: str = "openai/gpt-oss-20b"  # Fast reasoning (Groq)
    model_compound: str = "groq/compound"  # Web search (Groq only)
    model_compound_mini: str = "groq/compound-mini"  # Lightweight compound (Groq)
    model_vision: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision (Groq)
    
    # ── OpenRouter Model Selection (primary LLM — all free) ──────────────
    or_model_deep: str = "arcee-ai/trinity-large-preview:free"     # 400B MoE, 131K ctx, tool calling
    or_model_fast: str = "arcee-ai/trinity-mini:free"              # Quick greetings, routing
    or_model_analysis: str = "stepfun/step-3.5-flash:free"         # 196B MoE, 256K ctx, strong tool use
    or_model_kb_rag: str = "liquid/lfm-2.5-1.2b-thinking:free"    # Reasoning traces, RAG synthesis
    or_model_router: str = "arcee-ai/trinity-mini:free"            # Fast intent classification
    or_model_creative: str = "stepfun/step-3.5-flash:free"         # Educational content
    
    # Legacy aliases (keep for backward compat)
    model_fast: str = "llama-3.1-8b-instant"
    model_analysis: str = "llama-3.3-70b-versatile"
    model_creative: str = "llama-3.3-70b-versatile"
    model_router: str = "llama-3.1-8b-instant"
    
    # Groq Reasoning Parameters (for GPT-OSS models)
    reasoning_effort_deep: Literal["low", "medium", "high"] = "high"  # Max reasoning for 120B
    reasoning_effort_fast: Literal["low", "medium", "high"] = "medium"  # Balanced for 20B
    include_reasoning: bool = True  # FORCE: Always show reasoning process for transparency
    
    # Model Parameters (following Groq docs recommendations)
    temperature_reasoning: float = 0.4  # Lower for focused, accurate research
    temperature_fast: float = 0.3  # Low for classification
    temperature_creative: float = 0.8  # Higher for explanations
    
    # Max tokens - aligned with Groq rate limits
    max_tokens_reasoning_deep: int = 4096  # 120B: 8K TPM, 200K TPD
    max_tokens_reasoning_fast: int = 4096  # 20B: 8K TPM, 200K TPD
    max_tokens_compound: int = 4096  # Compound: 70K TPM
    max_tokens_default: int = 4096
    max_tokens_fast: int = 1024
    
    # Advanced Features - Groq-specific
    enable_tool_calling: bool = True  # Function calling
    enable_compound_ai: bool = True  # Use Compound for real-time web search
    enable_web_search: bool = True  # Via Compound built-in tool
    enable_code_execution: bool = True  # Via Compound built-in tool
    enable_browser: bool = True  # Via Compound built-in tool
    max_parallel_tools: int = 1  # GPT-OSS doesn't support parallel tools
    
    # Compound AI Settings
    compound_use_cases: list[str] = ["news", "real_time_data", "web_search", "calculations"]
    
    # External API Keys (Optional)
    serpapi_key: Optional[str] = None  # For custom web search (if not using Compound)
    news_api_key: Optional[str] = None  # For news fetching

    # Qdrant Vector DB — KB semantic search
    qdrant_url: Optional[str] = None      # Cloud URL (set in .env)
    qdrant_api_key: Optional[str] = None  # Cloud API key (set in .env)
    qdrant_collection: str = "daddys_kb"  # Collection name
    
    # Email/SMTP Configuration (supports both formats)
    email_server: bool = True
    email_server_host: str = "smtp.gmail.com"
    email_server_port: int = 587
    email_server_user: str = ""
    email_server_password: str = ""
    email_from: str = ""
    email_server_secure: bool = False
    
    # Legacy SMTP settings (fallback)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_email: str = ""
    smtp_password: str = ""
    smtp_from_name: str = "Daddy's AI"
    
    # Google OAuth
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    
    # JWT Secret — REQUIRED, minimum 32 characters
    jwt_secret: str = ""

    # OTP HMAC Secret — used to hash OTP codes before storing in MongoDB.
    # Must be set in .env for production. Generate with:
    # python -c "import secrets; print(secrets.token_hex(32))"
    otp_secret: str = ""

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Ensure JWT secret meets minimum security requirements."""
        if not v:
            import os
            if os.environ.get("ENVIRONMENT", "development") == "production":
                raise ValueError(
                    "JWT_SECRET is required in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            # Dev fallback — warn loudly
            import logging
            logging.getLogger(__name__).warning(
                "⚠️  JWT_SECRET not set! Using insecure dev fallback. "
                "Set JWT_SECRET in your .env (min 32 chars)."
            )
            return "dev-insecure-fallback-change-me-in-production-32chars"
        if len(v.encode()) < 32:
            raise ValueError(
                f"JWT_SECRET must be at least 32 characters (got {len(v)}). "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v
    
    # Frontend URL (for OAuth redirects)
    frontend_url: str = "http://localhost:3000"
    
    # Database Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "daddys_ai"
    mongodb_max_pool_size: int = 100  # Scaled for 5000+ concurrent users
    mongodb_min_pool_size: int = 10  # Higher minimum for better performance
    
    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"
    cache_ttl_seconds: int = 60  # 1 minute (faster refresh, less DB load)
    max_conversation_history: int = 50
    
    # API Settings - Scaled for high concurrency
    api_timeout_seconds: int = 30
    rate_limit_per_minute: int = 100  # Increased for multiple concurrent users
    # CORS origins — set ALLOWED_ORIGINS in .env for production, e.g.:
    # ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
    # Do NOT use ["*"] in production — it disables credentials-based auth.
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Agent Settings
    default_agent_mode: str = "auto"
    enable_streaming: bool = True
    enable_autonomous_reasoning: bool = True
    prefer_compound_for_realtime: bool = True  # Use Compound for news/real-time queries
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()  # Allow model_ prefix for our settings
    )
    
    def get_model_for_task(self, task_type: ModelType) -> str:
        """Get the best Groq model for a specific task type."""
        model_map = {
            ModelType.REASONING_DEEP: self.model_reasoning_deep,
            ModelType.REASONING_FAST: self.model_reasoning_fast,
            ModelType.COMPOUND: self.model_compound,
            ModelType.COMPOUND_MINI: self.model_compound_mini,
            ModelType.FAST: self.model_fast,
            ModelType.ANALYSIS: self.model_analysis,
            ModelType.CREATIVE: self.model_creative,
            ModelType.ROUTER: self.model_router
        }
        return model_map.get(task_type, self.model_reasoning_deep)
    
    def get_temperature_for_task(self, task_type: ModelType) -> float:
        """Get optimal temperature for task type."""
        temp_map = {
            ModelType.REASONING_DEEP: self.temperature_reasoning,
            ModelType.REASONING_FAST: self.temperature_reasoning,
            ModelType.COMPOUND: self.temperature_reasoning,
            ModelType.COMPOUND_MINI: self.temperature_reasoning,
            ModelType.FAST: self.temperature_fast,
            ModelType.ANALYSIS: self.temperature_reasoning,
            ModelType.CREATIVE: self.temperature_creative,
            ModelType.ROUTER: self.temperature_fast
        }
        return temp_map.get(task_type, 0.6)
    
    def get_max_tokens_for_task(self, task_type: ModelType) -> int:
        """Get optimal max tokens for task type."""
        if task_type in [ModelType.REASONING_DEEP, ModelType.REASONING_FAST]:
            return self.max_tokens_reasoning_deep
        elif task_type in [ModelType.COMPOUND, ModelType.COMPOUND_MINI]:
            return self.max_tokens_compound
        elif task_type in [ModelType.FAST, ModelType.ROUTER]:
            return self.max_tokens_fast
        return self.max_tokens_default
    
    def get_reasoning_effort(self, task_type: ModelType) -> Optional[Literal["low", "medium", "high"]]:
        """Get reasoning effort for GPT-OSS models."""
        if task_type == ModelType.REASONING_DEEP:
            return self.reasoning_effort_deep
        elif task_type == ModelType.REASONING_FAST:
            return self.reasoning_effort_fast
        return None
    
    def get_all_api_keys(self) -> list[str]:
        """Get all configured Groq API keys for rotation."""
        keys = [self.groq_api_key]
        if self.groq_api_key_2:
            keys.append(self.groq_api_key_2)
        if self.groq_api_key_3:
            keys.append(self.groq_api_key_3)
        if self.groq_api_key_4:
            keys.append(self.groq_api_key_4)
        if self.groq_api_key_5:
            keys.append(self.groq_api_key_5)
        return keys
    
    def get_all_openrouter_keys(self) -> list[str]:
        """Get all configured OpenRouter API keys for rotation."""
        keys = []
        for attr in ["openrouter_api_key", "openrouter_api_key_2",
                     "openrouter_api_key_3", "openrouter_api_key_4",
                     "openrouter_api_key_5"]:
            val = getattr(self, attr, None)
            if val:
                keys.append(val)
        return keys
    
    def get_openrouter_model(self, task_type: ModelType) -> str:
        """Get the best OpenRouter model for a task type."""
        model_map = {
            ModelType.REASONING_DEEP: self.or_model_deep,
            ModelType.REASONING_FAST: self.or_model_analysis,
            ModelType.FAST: self.or_model_fast,
            ModelType.ANALYSIS: self.or_model_analysis,
            ModelType.CREATIVE: self.or_model_creative,
            ModelType.ROUTER: self.or_model_router,
            ModelType.KB_RAG: self.or_model_kb_rag,
        }
        return model_map.get(task_type, self.or_model_analysis)
    
    @property
    def openrouter_available(self) -> bool:
        """Whether OpenRouter is configured."""
        return bool(self.openrouter_api_key)

    @property
    def nvidia_available(self) -> bool:
        """Whether NVIDIA NIM is configured."""
        return bool(self.nvidia_api_key)


# Global settings instance
settings = Settings()

