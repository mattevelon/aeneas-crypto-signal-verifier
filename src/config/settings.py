"""
Application settings using Pydantic for validation and environment variable loading.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
import warnings


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )
    
    # Application Settings
    app_env: str = Field(default="development", description="Application environment")
    app_port: int = Field(default=8000, description="Application port")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://crypto_user:crypto_password@localhost:5432/crypto_signals",
        description="PostgreSQL connection URL",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=40, description="Maximum overflow connections")
    database_pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Default TTL in seconds")
    redis_max_connections: int = Field(default=50, description="Max Redis connections")
    
    # Telegram Configuration
    telegram_api_id: Optional[int] = Field(default=None, description="Telegram API ID")
    telegram_api_hash: Optional[str] = Field(default=None, description="Telegram API Hash")
    telegram_phone_number: Optional[str] = Field(default=None, description="Telegram phone number")
    telegram_session_name: str = Field(default="crypto_signals_bot", description="Session name")
    telegram_channels: str = Field(default="", description="Comma-separated channel list")
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider")
    llm_model: str = Field(default="gpt-4-turbo-preview", description="LLM model")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    llm_max_tokens: int = Field(default=4000, description="Max tokens")
    llm_timeout: int = Field(default=30, description="LLM timeout")
    llm_max_retries: int = Field(default=3, description="Max retries")
    
    # Market Data APIs
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_api_secret: Optional[str] = Field(default=None, description="Binance API secret")
    kucoin_api_key: Optional[str] = Field(default=None, description="KuCoin API key")
    kucoin_api_secret: Optional[str] = Field(default=None, description="KuCoin API secret")
    kucoin_api_passphrase: Optional[str] = Field(default=None, description="KuCoin passphrase")
    
    # Vector Database Configuration
    vector_db_url: str = Field(default="http://localhost:6333", description="Vector DB URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    vector_collection_name: str = Field(default="signals", description="Collection name")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka servers")
    kafka_topic_signals: str = Field(default="crypto-signals", description="Signals topic")
    kafka_topic_validation: str = Field(default="signal-validation", description="Validation topic")
    kafka_consumer_group: str = Field(default="signal-processor", description="Consumer group")
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    cors_origins: str = Field(default="*", description="CORS origins")
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", description="JWT secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=1440, description="JWT expiration")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, description="Rate limit")
    rate_limit_burst_size: int = Field(default=100, description="Burst size")
    
    # Feature Flags
    enable_websocket: bool = Field(default=True, description="Enable WebSocket")
    enable_backtesting: bool = Field(default=True, description="Enable backtesting")
    enable_paper_trading: bool = Field(default=False, description="Enable paper trading")
    enable_notifications: bool = Field(default=True, description="Enable notifications")
    
    # Performance Settings
    signal_processing_timeout: int = Field(default=2, description="Processing timeout")
    max_concurrent_signals: int = Field(default=10, description="Max concurrent signals")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL")
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces", description="Jaeger endpoint")
    jaeger_service_name: str = Field(default="crypto-signals-api", description="Service name")
    
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if isinstance(self.cors_origins, str):
            if self.cors_origins == "*":
                return ["*"]
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins
    
    @property
    def telegram_channels_list(self) -> List[str]:
        """Get Telegram channels as a list."""
        if isinstance(self.telegram_channels, str) and self.telegram_channels:
            return [channel.strip() for channel in self.telegram_channels.split(",")]
        return []
    
    @model_validator(mode='after')
    def validate_credentials(self):
        """Validate and warn about missing credentials."""
        missing_critical = []
        missing_optional = []
        
        # Critical credentials
        if not self.database_url:
            missing_critical.append("DATABASE_URL")
        
        # Important but not critical (app can run without them)
        if not self.telegram_api_id or not self.telegram_api_hash:
            missing_optional.append("Telegram credentials (telegram_api_id, telegram_api_hash)")
            
        if not self.llm_api_key:
            missing_optional.append("LLM API key (llm_api_key)")
            
        if not self.binance_api_key and not self.kucoin_api_key:
            missing_optional.append("Exchange API keys (binance or kucoin)")
        
        # Raise error for critical missing
        if missing_critical:
            raise ValueError(f"Critical configuration missing: {', '.join(missing_critical)}")
        
        # Warn for optional missing
        if missing_optional:
            warnings.warn(
                f"Optional services not configured (app will run with limited functionality): {', '.join(missing_optional)}",
                UserWarning
            )
        
        return self
    
    @property
    def has_telegram_credentials(self) -> bool:
        """Check if Telegram credentials are available."""
        return bool(self.telegram_api_id and self.telegram_api_hash and self.telegram_phone_number)
    
    @property
    def has_llm_credentials(self) -> bool:
        """Check if LLM credentials are available."""
        return bool(self.llm_api_key)
    
    @property
    def has_exchange_credentials(self) -> bool:
        """Check if any exchange credentials are available."""
        return bool(self.binance_api_key or self.kucoin_api_key)


# Create global settings instance
settings = Settings()

def get_settings():
    """Get settings instance."""
    return settings
