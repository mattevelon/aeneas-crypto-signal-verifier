"""
Unit tests for signal validation system.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.core.signal_validator import SignalValidator
from src.models import SignalDirection, RiskLevel


@pytest.fixture
def signal_validator():
    """Create signal validator instance."""
    return SignalValidator()


@pytest.fixture
def sample_signal():
    """Create sample signal data."""
    return {
        "pair": "BTC/USDT",
        "direction": SignalDirection.LONG,
        "entry_price": 42000,
        "stop_loss": 41000,
        "take_profits": [43000, 44000, 45000],
        "leverage": 5,
        "confidence_score": 70
    }


@pytest.mark.asyncio
async def test_validate_signal_valid(signal_validator, sample_signal):
    """Test validation of valid signal."""
    with patch('src.core.signal_validator.market_data') as mock_market:
        mock_market.validate_price = AsyncMock(return_value={
            "valid": True,
            "current_price": 42100,
            "deviation_percentage": 0.24
        })
        mock_market.get_24h_stats = AsyncMock(return_value={
            "quote_volume": 5000000
        })
        
        result = await signal_validator.validate_signal(sample_signal)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert result["risk_level"] == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_validate_market_data(signal_validator, sample_signal):
    """Test market data validation."""
    with patch('src.core.signal_validator.market_data') as mock_market:
        mock_market.validate_price = AsyncMock(return_value={
            "valid": False,
            "current_price": 45000,
            "deviation_percentage": 7.14,
            "reason": "Price deviation: 7.14%"
        })
        
        result = await signal_validator._validate_market_data(sample_signal)
        
        assert result["market_validation"]["valid"] is False
        assert "deviation" in result["market_validation"]["reason"]


@pytest.mark.asyncio
async def test_validate_risk_parameters_high_leverage(signal_validator):
    """Test risk validation with high leverage."""
    signal_data = {
        "entry_price": 100,
        "stop_loss": 95,
        "direction": SignalDirection.LONG,
        "leverage": 25  # Too high
    }
    
    result = await signal_validator._validate_risk_parameters(signal_data)
    
    assert result["risk_validation"]["valid"] is False
    assert "Leverage too high" in result["risk_validation"]["reason"]


@pytest.mark.asyncio
async def test_validate_risk_parameters_stop_loss_too_far(signal_validator):
    """Test risk validation with stop loss too far."""
    signal_data = {
        "entry_price": 100,
        "stop_loss": 85,  # 15% stop loss
        "direction": SignalDirection.LONG,
        "leverage": 5
    }
    
    result = await signal_validator._validate_risk_parameters(signal_data)
    
    assert result["risk_validation"]["valid"] is False
    assert "Stop loss too far" in result["risk_validation"]["reason"]


@pytest.mark.asyncio
async def test_validate_risk_parameters_good_risk_reward(signal_validator):
    """Test risk validation with good risk/reward ratio."""
    signal_data = {
        "entry_price": 100,
        "stop_loss": 98,
        "take_profits": [104, 106],  # 2:1 risk/reward
        "direction": SignalDirection.LONG,
        "leverage": 3
    }
    
    result = await signal_validator._validate_risk_parameters(signal_data)
    
    assert result["risk_validation"]["valid"] is True
    assert result["risk_validation"]["stop_loss_percentage"] == 2.0


@pytest.mark.asyncio
async def test_validate_technical_levels_long(signal_validator):
    """Test technical level validation for long position."""
    # Invalid long signal (SL above entry)
    signal_data = {
        "entry_price": 100,
        "stop_loss": 105,  # Invalid
        "take_profits": [110, 115],
        "direction": SignalDirection.LONG
    }
    
    result = await signal_validator._validate_technical_levels(signal_data)
    
    assert result["technical_validation"]["valid"] is False
    assert "Stop loss above entry" in result["technical_validation"]["reason"]


@pytest.mark.asyncio
async def test_validate_technical_levels_short(signal_validator):
    """Test technical level validation for short position."""
    # Valid short signal
    signal_data = {
        "entry_price": 100,
        "stop_loss": 105,
        "take_profits": [95, 90],
        "direction": SignalDirection.SHORT
    }
    
    result = await signal_validator._validate_technical_levels(signal_data)
    
    assert result["technical_validation"]["valid"] is True


@pytest.mark.asyncio
async def test_validate_liquidity_low_volume(signal_validator):
    """Test liquidity validation with low volume."""
    signal_data = {"pair": "BTC/USDT"}
    
    with patch('src.core.signal_validator.market_data') as mock_market:
        mock_market.get_24h_stats = AsyncMock(return_value={
            "quote_volume": 50000  # Below minimum
        })
        
        result = await signal_validator._validate_liquidity(signal_data)
        
        assert result["liquidity_validation"]["valid"] is True  # Warning, not error
        assert "Low liquidity" in result["liquidity_validation"]["warning"]


def test_adjust_confidence(signal_validator):
    """Test confidence score adjustment."""
    base_confidence = 70
    
    # With warnings and errors
    validation_results = {
        "warnings": ["Warning 1", "Warning 2"],
        "errors": ["Error 1"],
        "validations": {
            "risk_validation": {"risk_reward_ratio": 4}  # Good ratio
        }
    }
    
    result = signal_validator._adjust_confidence(base_confidence, validation_results)
    
    # 70 - (2*5) - (1*10) + 10 = 60
    assert result == 60


def test_assess_risk_level_high(signal_validator):
    """Test risk level assessment - high risk."""
    signal_data = {"leverage": 15}
    validation_results = {
        "validations": {
            "risk_validation": {"stop_loss_percentage": 8},
            "liquidity_validation": {"volume_usd": 200000}
        }
    }
    
    result = signal_validator._assess_risk_level(signal_data, validation_results)
    assert result == RiskLevel.HIGH


def test_assess_risk_level_low(signal_validator):
    """Test risk level assessment - low risk."""
    signal_data = {"leverage": 2}
    validation_results = {
        "validations": {
            "risk_validation": {"stop_loss_percentage": 1.5},
            "liquidity_validation": {"volume_usd": 2000000}
        }
    }
    
    result = signal_validator._assess_risk_level(signal_data, validation_results)
    assert result == RiskLevel.LOW
