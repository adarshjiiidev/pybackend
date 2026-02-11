"""
Application settings using Pydantic Settings.
Optimized for Groq's GPT-OSS-120B reasoning model and Compound AI.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    """Available model types for different tasks."""
    REASONING_DEEP = "reasoning_deep"  # GPT-OSS-120B for deep reasoning
    REASONING_FAST = "reasoning_fast"  # GPT-OSS-20B for faster reasoning
    COMPOUND = "compound"  # Compound AI with built-in tools
    COMPOUND_MINI = "compound_mini"  # Lightweight compound
    FAST = "fast"  # Quick responses, simple tasks
    ANALYSIS = "analysis"  # Data analysis, technical tasks
    CREATIVE = "creative"  # Educational content, explanations
    ROUTER = "router"  # Intent classification


class Settings(BaseSettings):
    """Application configuration settings optimized for Groq API."""
    
    # LLM Configuration - Groq Multi-Model System
    groq_api_key: str
    
    # Model Selection - Using Groq's best models
    model_reasoning_deep: str = "openai/gpt-oss-120b"  # Deep reasoning with high effort
    model_reasoning_fast: str = "openai/gpt-oss-20b"  # Fast reasoning with medium effort
    model_compound: str = "groq/compound"  # Built-in web search, browser, code exec
    model_compound_mini: str = "groq/compound-mini"  # Lightweight compound
    model_fast: str = "llama-3.1-8b-instant"  # Quick routing (14.4K RPD, 6K TPM)
    model_analysis: str = "llama-3.3-70b-versatile"  # Market analysis (1K RPD, 12K TPM)
    model_creative: str = "llama-3.3-70b-versatile"  # Educational content (UPDATED: 3.1 decommissioned)
    model_creative: str = "llama-3.3-70b-versatile"  # Educational content (UPDATED: 3.1 decommissioned)
    model_router: str = "llama-3.1-8b-instant"  # Fast intent classification
    model_vision: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision & Multimodal tasks
    
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
    
    # JWT Secret
    jwt_secret: str = ""
    
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
    cors_origins: list[str] = ["*"]
    
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


# Global settings instance
settings = Settings()

