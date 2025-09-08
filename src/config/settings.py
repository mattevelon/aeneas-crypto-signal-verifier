"""
Application settings using Pydantic for validation and environment variable loading.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application Settings
    app_env: str = Field(default="development", description="Application environment")
    app_port: int = Field(default=8000, description="Application port")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/crypto_signals",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=40, description="Maximum overflow connections")
    database_pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Default TTL in seconds")
    redis_max_connections: int = Field(default=50, description="Maximum Redis connections")
    
    # Telegram Configuration
    telegram_api_id: Optional[int] = Field(default=None, description="Telegram API ID")
    telegram_api_hash: Optional[str] = Field(default=None, description="Telegram API Hash")
    telegram_phone_number: Optional[str] = Field(default=None, description="Telegram phone number")
    telegram_session_name: str = Field(default="crypto_signals_bot", description="Session name")
    telegram_channels: List[str] = Field(default_factory=list, description="Telegram channels to monitor")
    
    @validator("telegram_channels", pre=True)
    def parse_channels(cls, v):
        if isinstance(v, str):
            return [ch.strip() for ch in v.split(",") if ch.strip()]
        return v
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider (openai/anthropic)")
    llm_model: str = Field(default="gpt-4-turbo-preview", description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    llm_max_tokens: int = Field(default=4000, description="Maximum tokens per request")
    llm_timeout: int = Field(default=30, description="LLM request timeout")
    llm_max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Fallback LLM Configuration
    fallback_llm_provider: Optional[str] = Field(default=None, description="Fallback LLM provider")
    fallback_llm_model: Optional[str] = Field(default=None, description="Fallback LLM model")
    fallback_llm_api_key: Optional[str] = Field(default=None, description="Fallback LLM API key")
    
    # Market Data APIs
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_api_secret: Optional[str] = Field(default=None, description="Binance API secret")
    kucoin_api_key: Optional[str] = Field(default=None, description="KuCoin API key")
    kucoin_api_secret: Optional[str] = Field(default=None, description="KuCoin API secret")
    kucoin_api_passphrase: Optional[str] = Field(default=None, description="KuCoin API passphrase")
    
    # Vector Database Configuration
    vector_db_url: str = Field(default="http://localhost:6333", description="Vector DB URL")
    vector_collection: str = Field(default="signals", description="Vector collection name")
    vector_embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    vector_dimension: int = Field(default=1536, description="Vector dimension")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    kafka_signal_topic: str = Field(default="crypto-signals", description="Signal topic")
    kafka_validation_topic: str = Field(default="signal-validation", description="Validation topic")
    kafka_alert_topic: str = Field(default="signal-alerts", description="Alert topic")
    kafka_consumer_group: str = Field(default="signal-processor", description="Consumer group")
    
    # OCR Configuration
    ocr_provider: str = Field(default="google", description="OCR provider (google/aws/tesseract)")
    google_vision_api_key: Optional[str] = Field(default=None, description="Google Vision API key")
    aws_textract_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    
    # Security Configuration
    jwt_secret_key: str = Field(default="change-this-secret-key", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(default=15, description="Access token expiry")
    jwt_refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], description="CORS origins")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces", description="Jaeger endpoint")
    jaeger_service_name: str = Field(default="crypto-signals-api", description="Service name")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=100, description="Rate limit per minute")
    rate_limit_burst_size: int = Field(default=150, description="Burst size")
    
    # Feature Flags
    enable_websocket: bool = Field(default=True, description="Enable WebSocket support")
    enable_backtesting: bool = Field(default=True, description="Enable backtesting")
    enable_paper_trading: bool = Field(default=False, description="Enable paper trading")
    enable_signal_feedback: bool = Field(default=True, description="Enable signal feedback")
    
    # Performance Settings
    worker_count: int = Field(default=4, description="Number of workers")
    worker_timeout: int = Field(default=120, description="Worker timeout in seconds")
    batch_size: int = Field(default=5, description="Batch processing size")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    
    # Notification Settings
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    telegram_alert_chat_id: Optional[int] = Field(default=None, description="Telegram alert chat ID")
    
    # External Services
    coingecko_api_key: Optional[str] = Field(default=None, description="CoinGecko API key")
    newsapi_key: Optional[str] = Field(default=None, description="NewsAPI key")
    cryptocompare_api_key: Optional[str] = Field(default=None, description="CryptoCompare API key")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True


# Create global settings instance
settings = Settings()
