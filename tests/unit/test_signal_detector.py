"""
Unit tests for signal detection system.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.core.signal_detector import SignalDetector
from src.models import SignalDirection, RiskLevel


@pytest.fixture
def signal_detector():
    """Create signal detector instance."""
    return SignalDetector()


@pytest.mark.asyncio
async def test_detect_valid_signal(signal_detector):
    """Test detection of valid trading signal."""
    text = """
    BTC/USDT
    Entry: $42,500
    Target 1: $43,200
    Target 2: $44,000
    Stop Loss: $41,800
    """
    
    result = await signal_detector.detect(text)
    assert result is True


@pytest.mark.asyncio
async def test_detect_invalid_signal(signal_detector):
    """Test detection of invalid text."""
    text = "This is just random text about crypto"
    
    result = await signal_detector.detect(text)
    assert result is False


@pytest.mark.asyncio
async def test_extract_long_signal(signal_detector):
    """Test extraction of long signal details."""
    text = """
    LONG BTC/USDT
    Entry: 42500
    TP1: 43200
    TP2: 44000
    SL: 41800
    Leverage: 5x
    """
    
    with patch.object(signal_detector, '_analyze_with_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "confidence_score": 75,
            "risk_assessment": "medium",
            "justification": {"analysis": "Good signal"}
        }
        
        result = await signal_detector.extract(text)
        
        assert result is not None
        assert result["pair"] == "BTC/USDT"
        assert result["direction"] == SignalDirection.LONG
        assert result["entry_price"] == 42500
        assert result["stop_loss"] == 41800
        assert result["take_profits"] == [43200, 44000]
        assert result["leverage"] == 5


@pytest.mark.asyncio
async def test_extract_short_signal(signal_detector):
    """Test extraction of short signal details."""
    text = """
    SHORT ETH/USDT
    Entry: $2200
    Stop Loss: $2250
    Take Profit: $2100
    """
    
    with patch.object(signal_detector, '_analyze_with_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "confidence_score": 60,
            "risk_assessment": "low",
            "justification": {"analysis": "Bearish setup"}
        }
        
        result = await signal_detector.extract(text)
        
        assert result is not None
        assert result["pair"] == "ETH/USDT"
        assert result["direction"] == SignalDirection.SHORT
        assert result["entry_price"] == 2200
        assert result["stop_loss"] == 2250
        assert result["take_profits"] == [2100]


def test_extract_pair(signal_detector):
    """Test cryptocurrency pair extraction."""
    test_cases = [
        ("BTC/USDT", "BTC/USDT"),
        ("ETH-USDT", "ETH/USDT"),
        ("BNB USDT", "BNB/USDT"),
        ("btc/usdt", "BTC/USDT"),
    ]
    
    for text, expected in test_cases:
        result = signal_detector._extract_pair(text)
        assert result == expected


def test_extract_direction(signal_detector):
    """Test signal direction extraction."""
    assert signal_detector._extract_direction("LONG BTC") == SignalDirection.LONG
    assert signal_detector._extract_direction("SHORT ETH") == SignalDirection.SHORT
    assert signal_detector._extract_direction("Buy signal") == SignalDirection.LONG
    assert signal_detector._extract_direction("Sell position") == SignalDirection.SHORT
    assert signal_detector._extract_direction("Random text") is None


def test_extract_entry_price(signal_detector):
    """Test entry price extraction."""
    test_cases = [
        ("Entry: $42,500", 42500),
        ("Buy at 1234.56", 1234.56),
        ("Entry price: 999", 999),
        ("Long @ 50000", 50000),
    ]
    
    for text, expected in test_cases:
        result = signal_detector._extract_entry_price(text)
        assert result == expected


def test_extract_stop_loss(signal_detector):
    """Test stop loss extraction."""
    test_cases = [
        ("Stop Loss: $41,800", 41800),
        ("SL: 1234.56", 1234.56),
        ("Stop: 999", 999),
        ("stop-loss @ 50000", 50000),
    ]
    
    for text, expected in test_cases:
        result = signal_detector._extract_stop_loss(text)
        assert result == expected


def test_extract_take_profits(signal_detector):
    """Test take profit extraction."""
    text = """
    TP1: 43200
    TP2: 44000
    Target 3: 45000
    """
    
    result = signal_detector._extract_take_profits(text)
    assert result == [43200, 44000, 45000]


def test_determine_risk_level(signal_detector):
    """Test risk level determination."""
    # High risk scenario
    signal_data = {
        "entry_price": 100,
        "stop_loss": 90,  # 10% stop loss
        "leverage": 15
    }
    llm_analysis = {"risk_assessment": "high"}
    
    result = signal_detector._determine_risk_level(signal_data, llm_analysis)
    assert result == RiskLevel.HIGH
    
    # Low risk scenario
    signal_data = {
        "entry_price": 100,
        "stop_loss": 98,  # 2% stop loss
        "leverage": 2
    }
    llm_analysis = {"risk_assessment": "low"}
    
    result = signal_detector._determine_risk_level(signal_data, llm_analysis)
    assert result == RiskLevel.LOW
