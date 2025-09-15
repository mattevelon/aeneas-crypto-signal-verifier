"""
Unit tests for configuration settings.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import warnings
from pydantic import ValidationError

from src.config.settings import Settings


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        """Test that default settings load correctly."""
        settings = Settings()
        
        assert settings.app_env == "development"
        assert settings.app_port == 8000
        assert settings.log_level == "INFO"
        assert settings.debug is False
    
    def test_database_settings(self):
        """Test database configuration."""
        settings = Settings()
        
        assert "postgresql://" in settings.database_url
        assert settings.database_pool_size == 20
        assert settings.database_max_overflow == 40
        assert settings.database_pool_timeout == 30
    
    def test_redis_settings(self):
        """Test Redis configuration."""
        settings = Settings()
        
        assert "redis://" in settings.redis_url
        assert settings.redis_ttl == 3600
        assert settings.redis_max_connections == 50
    
    @patch.dict(os.environ, {
        "TELEGRAM_API_ID": "123456", 
        "TELEGRAM_API_HASH": "abc123",
        "TELEGRAM_PHONE_NUMBER": "+1234567890"
    })
    def test_telegram_credentials_complete(self):
        """Test complete Telegram credentials detection."""
        settings = Settings()
        
        assert settings.telegram_api_id == 123456
        assert settings.telegram_api_hash == "abc123"
        assert settings.telegram_phone_number == "+1234567890"
        assert settings.has_telegram_credentials is True
    
    @patch.dict(os.environ, {"TELEGRAM_API_ID": "123456", "TELEGRAM_API_HASH": "abc123"})
    def test_telegram_credentials_incomplete(self):
        """Test incomplete Telegram credentials detection."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settings = Settings()
        
        assert settings.telegram_api_id == 123456
        assert settings.telegram_api_hash == "abc123"
        assert settings.has_telegram_credentials is False  # Missing phone number
    
    def test_cors_origins_list(self):
        """Test CORS origins parsing."""
        settings = Settings()
        
        # Test default wildcard
        assert settings.cors_origins_list == ["*"]
        
        # Test comma-separated list
        settings.cors_origins = "http://localhost:3000,https://example.com"
        assert settings.cors_origins_list == ["http://localhost:3000", "https://example.com"]
    
    def test_telegram_channels_list(self):
        """Test Telegram channels parsing."""
        settings = Settings()
        
        # Test empty
        assert settings.telegram_channels_list == []
        
        # Test comma-separated list
        settings.telegram_channels = "channel1,channel2,channel3"
        assert settings.telegram_channels_list == ["channel1", "channel2", "channel3"]
    
    def test_feature_flags(self):
        """Test feature flags."""
        settings = Settings()
        
        assert settings.enable_websocket is True
        assert settings.enable_backtesting is True
        assert settings.enable_paper_trading is False
        assert settings.enable_notifications is True
    
    def test_performance_settings(self):
        """Test performance configuration."""
        settings = Settings()
        
        assert settings.signal_processing_timeout == 2
        assert settings.max_concurrent_signals == 10
        assert settings.cache_ttl_seconds == 3600
    
    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    def test_llm_credentials(self):
        """Test LLM credentials detection."""
        settings = Settings()
        
        assert settings.llm_api_key == "test-key"
        assert settings.has_llm_credentials is True
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4-turbo-preview"
    
    @patch.dict(os.environ, {"BINANCE_API_KEY": "binance-key"})
    def test_exchange_credentials_binance(self):
        """Test Binance credentials detection."""
        settings = Settings()
        
        assert settings.binance_api_key == "binance-key"
        assert settings.has_exchange_credentials is True
    
    @patch.dict(os.environ, {"KUCOIN_API_KEY": "kucoin-key"})
    def test_exchange_credentials_kucoin(self):
        """Test KuCoin credentials detection."""
        settings = Settings()
        
        assert settings.kucoin_api_key == "kucoin-key"
        assert settings.has_exchange_credentials is True
    
    def test_kafka_settings(self):
        """Test Kafka configuration."""
        settings = Settings()
        
        assert settings.kafka_bootstrap_servers == "localhost:9092"
        assert settings.kafka_topic_signals == "crypto-signals"
        assert settings.kafka_topic_validation == "signal-validation"
        assert settings.kafka_consumer_group == "signal-processor"
    
    def test_monitoring_settings(self):
        """Test monitoring configuration."""
        settings = Settings()
        
        assert settings.prometheus_port == 9090
        assert settings.jaeger_service_name == "crypto-signals-api"
        assert "14268" in settings.jaeger_endpoint
    
    def test_api_settings(self):
        """Test API configuration."""
        settings = Settings()
        
        assert settings.api_prefix == "/api/v1"
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_expiration_minutes == 1440
        assert settings.rate_limit_requests_per_minute == 60
        assert settings.rate_limit_burst_size == 100
    
    def test_vector_db_settings(self):
        """Test vector database configuration."""
        settings = Settings()
        
        assert settings.vector_db_url == "http://localhost:6333"
        assert settings.vector_collection_name == "signals"
        assert settings.qdrant_api_key is None  # Default for local deployment
    
    def test_validation_warnings(self):
        """Test that missing optional credentials trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            settings = Settings()
            
            # Should have warnings for missing optional services
            warning_messages = [str(warning.message) for warning in w]
            assert any("Optional services not configured" in msg for msg in warning_messages)
    
    @patch.dict(os.environ, {"DATABASE_URL": ""})
    def test_critical_missing_raises_error(self):
        """Test that missing critical configuration raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Check that DATABASE_URL is mentioned in the error
        assert "DATABASE_URL" in str(exc_info.value) or "Critical configuration missing" in str(exc_info.value)
