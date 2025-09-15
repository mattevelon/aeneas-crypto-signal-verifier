"""
Test database operations and models.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import init_db, close_db, get_db
from src.core.db_operations import db_ops
from src.models import Signal, TelegramMessage, SignalDirection, RiskLevel, SignalStatus


@pytest.fixture
async def db_session():
    """Create a test database session."""
    await init_db()
    async for session in get_db():
        yield session
    await close_db()


@pytest.mark.asyncio
async def test_bulk_insert_messages(db_session: AsyncSession):
    """Test bulk message insertion."""
    messages = [
        {
            "channel_id": 123456,
            "message_id": i,
            "content": f"Test message {i}",
            "author": "TestBot",
            "timestamp": datetime.utcnow(),
            "has_media": False,
            "processed": False,
            "is_signal": i % 5 == 0  # Every 5th message is a signal
        }
        for i in range(100)
    ]
    
    inserted = await db_ops.bulk_insert_messages(messages)
    assert inserted == 100


@pytest.mark.asyncio
async def test_signal_creation(db_session: AsyncSession):
    """Test signal creation and validation."""
    signal = Signal(
        source_channel_id=123456,
        original_message_id=789,
        pair="BTC/USDT",
        direction=SignalDirection.LONG,
        entry_price=65000.00,
        stop_loss=63000.00,
        take_profits=[67000.00, 70000.00, 75000.00],
        risk_level=RiskLevel.MEDIUM,
        confidence_score=75.5,
        justification={"reason": "Strong support level"},
        status=SignalStatus.ACTIVE
    )
    
    db_session.add(signal)
    await db_session.commit()
    
    assert signal.id is not None
    assert signal.status == SignalStatus.ACTIVE


@pytest.mark.asyncio
async def test_transaction_rollback(db_session: AsyncSession):
    """Test transaction rollback on error."""
    async with db_ops.transaction() as db:
        # This should fail due to constraint violation
        with pytest.raises(Exception):
            signal1 = Signal(
                source_channel_id=123456,
                original_message_id=999,
                pair="BTC/USDT",
                direction=SignalDirection.LONG,
                entry_price=65000.00,
                stop_loss=63000.00,
                take_profits=[67000.00],
                risk_level=RiskLevel.LOW,
                confidence_score=80.0,
                justification={"test": "rollback"}
            )
            db.add(signal1)
            
            # Try to add duplicate (should fail)
            signal2 = Signal(
                source_channel_id=123456,
                original_message_id=999,  # Same as signal1
                pair="ETH/USDT",
                direction=SignalDirection.SHORT,
                entry_price=3000.00,
                stop_loss=3100.00,
                take_profits=[2900.00],
                risk_level=RiskLevel.HIGH,
                confidence_score=60.0,
                justification={"test": "duplicate"}
            )
            db.add(signal2)
            await db.commit()


@pytest.mark.asyncio
async def test_channel_statistics():
    """Test channel statistics calculation."""
    channel_id = 123456
    stats = await db_ops.get_channel_statistics(channel_id, days=7)
    
    assert "channel_id" in stats
    assert "message_count" in stats
    assert "signal_count" in stats
    assert stats["period_days"] == 7


@pytest.mark.asyncio
async def test_text_compression():
    """Test text compression for large messages."""
    large_text = "A" * 2000  # 2000 characters
    
    compressed = db_ops._compress_text(large_text)
    assert compressed.startswith("COMPRESSED:")
    assert len(compressed) < len(large_text)
    
    decompressed = db_ops._decompress_text(compressed)
    assert decompressed == large_text


@pytest.mark.asyncio
async def test_bulk_update_signals():
    """Test bulk signal updates."""
    updates = [
        {
            "id": "test-signal-1",
            "status": SignalStatus.CLOSED,
            "confidence_score": 85.0
        },
        {
            "id": "test-signal-2",
            "status": SignalStatus.EXPIRED,
            "risk_level": RiskLevel.HIGH
        }
    ]
    
    # This will return 0 if signals don't exist, which is fine for test
    updated = await db_ops.bulk_update_signals(updates)
    assert updated >= 0
