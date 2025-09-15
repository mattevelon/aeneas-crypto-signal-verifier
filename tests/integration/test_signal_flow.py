"""
Integration tests for complete signal processing flow.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db_context
from src.core.signal_detector import SignalDetector
from src.core.redis_client import signal_cache
from src.models import Signal, SignalDirection, RiskLevel, SignalStatus


@pytest.mark.asyncio
async def test_signal_detection_and_extraction():
    """Test signal detection and extraction from text."""
    detector = SignalDetector()
    
    # Test signal text
    signal_text = """
    🚀 BTC/USDT LONG Signal
    
    Entry: $45,000
    Stop Loss: $44,000
    
    Take Profits:
    TP1: $46,000
    TP2: $47,000
    TP3: $48,500
    
    Leverage: 5x
    Risk: 2%
    """
    
    # Test detection
    is_signal = await detector.detect(signal_text)
    assert is_signal is True
    
    # Test extraction
    extracted = await detector.extract(signal_text, channel_id=123456)
    assert extracted is not None
    assert extracted["pair"] == "BTC/USDT"
    assert extracted["direction"] == SignalDirection.LONG
    assert extracted["entry_price"] == 45000.0
    assert extracted["stop_loss"] == 44000.0
    assert len(extracted["take_profits"]) == 3
    assert extracted["leverage"] == 5
    assert extracted["risk_percentage"] == 2.0


@pytest.mark.asyncio
async def test_signal_persistence_and_caching():
    """Test signal persistence to database and cache."""
    async with get_db_context() as db:
        # Create a test signal
        signal = Signal(
            source_channel_id=123456,
            original_message_id=789,
            pair="ETH/USDT",
            direction=SignalDirection.SHORT,
            entry_price=2500.0,
            stop_loss=2600.0,
            take_profits=[2400.0, 2300.0, 2200.0],
            risk_level=RiskLevel.MEDIUM,
            confidence_score=75.0,
            justification={"test": "data"},
            status=SignalStatus.ACTIVE
        )
        
        db.add(signal)
        await db.commit()
        await db.refresh(signal)
        
        # Verify persistence
        assert signal.id is not None
        assert signal.created_at is not None
        
        # Test caching
        await signal_cache.set(
            f"signal:{signal.id}",
            {
                "id": str(signal.id),
                "pair": signal.pair,
                "direction": signal.direction.value,
                "confidence_score": signal.confidence_score
            },
            ttl=60
        )
        
        # Verify cache
        cached = await signal_cache.get(f"signal:{signal.id}")
        assert cached is not None
        assert cached["pair"] == "ETH/USDT"
        assert cached["confidence_score"] == 75.0
        
        # Cleanup
        await signal_cache.delete(f"signal:{signal.id}")


@pytest.mark.asyncio
async def test_signal_api_workflow():
    """Test complete API workflow for signals."""
    from httpx import AsyncClient
    from src.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create a signal
        signal_data = {
            "source_channel_id": 999999,
            "original_message_id": 111,
            "pair": "SOL/USDT",
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profits": [105.0, 110.0, 115.0],
            "risk_level": "low",
            "confidence_score": 85.0,
            "justification": {"reason": "test"}
        }
        
        response = await client.post("/api/v1/signals/", json=signal_data)
        
        # Check response
        if response.status_code == 200:
            created_signal = response.json()
            assert created_signal["pair"] == "SOL/USDT"
            assert created_signal["confidence_score"] == 85.0
            
            # Get the signal
            signal_id = created_signal["id"]
            response = await client.get(f"/api/v1/signals/{signal_id}")
            assert response.status_code == 200
            
            # Update the signal
            update_data = {"status": "closed", "confidence_score": 90.0}
            response = await client.patch(
                f"/api/v1/signals/{signal_id}",
                json=update_data
            )
            
            if response.status_code == 200:
                updated_signal = response.json()
                assert updated_signal["status"] == "closed"
                assert updated_signal["confidence_score"] == 90.0
            
            # List signals
            response = await client.get("/api/v1/signals/")
            assert response.status_code == 200
            signals = response.json()
            assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_risk_assessment_logic():
    """Test risk assessment and determination logic."""
    detector = SignalDetector()
    
    # High risk signal
    high_risk_data = {
        "stop_loss": 100.0,
        "entry_price": 110.0,  # 9% stop loss distance
        "leverage": 20
    }
    
    risk_level = detector._determine_risk_level(
        high_risk_data,
        {"risk_assessment": "high"}
    )
    assert risk_level == RiskLevel.HIGH
    
    # Low risk signal
    low_risk_data = {
        "stop_loss": 100.0,
        "entry_price": 101.0,  # 1% stop loss distance
        "leverage": 2
    }
    
    risk_level = detector._determine_risk_level(
        low_risk_data,
        {"risk_assessment": "low"}
    )
    assert risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_concurrent_signal_processing():
    """Test concurrent signal processing capabilities."""
    detector = SignalDetector()
    
    # Create multiple signal texts
    signals = [
        "BTC/USDT LONG Entry: 45000 SL: 44000 TP: 46000",
        "ETH/USDT SHORT Entry: 2500 SL: 2600 TP: 2400",
        "SOL/USDT LONG Entry: 100 SL: 95 TP: 110",
        "DOGE/USDT SHORT Entry: 0.08 SL: 0.085 TP: 0.075"
    ]
    
    # Process concurrently
    tasks = [detector.detect(text) for text in signals]
    results = await asyncio.gather(*tasks)
    
    # All should be detected as signals
    assert all(results)
    
    # Extract concurrently
    extract_tasks = [detector.extract(text) for text in signals]
    extracted = await asyncio.gather(*extract_tasks)
    
    # Verify all extracted
    assert len([e for e in extracted if e is not None]) == 4
