"""Configuration validation unit tests."""

import pytest
from unittest.mock import patch
import os
from src.config.settings import Settings


class TestConfiguration:
    """Test configuration settings validation."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()
        assert settings.app_env == "development"
        assert settings.app_port == 8000
        assert settings.log_level == "INFO"

    def test_database_url_validation(self):
        """Test database URL configuration."""
        settings = Settings(
            database_url="postgresql://user:pass@localhost:5432/test_db"
        )
        assert "test_db" in settings.database_url

    def test_telegram_channels_parsing(self):
        """Test Telegram channels string parsing."""
        settings = Settings(
            telegram_channels="channel1,channel2,channel3"
        )
        assert settings.telegram_channels == "channel1,channel2,channel3"
        assert settings.telegram_channels_list == ["channel1", "channel2", "channel3"]

    def test_cors_origins_parsing(self):
        """Test CORS origins string parsing."""
        settings = Settings(
            cors_origins="http://localhost:3000,http://localhost:8080"
        )
        assert settings.cors_origins == "http://localhost:3000,http://localhost:8080"
        assert len(settings.cors_origins_list) == 2
        assert "http://localhost:3000" in settings.cors_origins_list

    @patch.dict(os.environ, {"APP_ENV": "production", "APP_PORT": "9000"})
    def test_env_variable_override(self):
        """Test environment variable override."""
        settings = Settings()
        assert settings.app_env == "production"
        assert settings.app_port == 9000

    def test_llm_configuration(self):
        """Test LLM configuration settings."""
        settings = Settings(
            llm_provider="openai",
            llm_model="gpt-4-turbo",
            llm_temperature=0.3,
            llm_max_tokens=4000
        )
        assert settings.llm_provider == "openai"
        assert settings.llm_temperature == 0.3
        assert settings.llm_max_tokens == 4000

    def test_rate_limiting_config(self):
        """Test rate limiting configuration."""
        settings = Settings(
            rate_limit_requests_per_minute=100,
            rate_limit_burst_size=150
        )
        assert settings.rate_limit_requests_per_minute == 100
        assert settings.rate_limit_burst_size == 150

    def test_feature_flags(self):
        """Test feature flag configuration."""
        settings = Settings(
            enable_websocket=True,
            enable_backtesting=True,
            enable_paper_trading=False
        )
        assert settings.enable_websocket is True
        assert settings.enable_backtesting is True
        assert settings.enable_paper_trading is False
