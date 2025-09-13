"""
Signal management API endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from pydantic import BaseModel, Field
import structlog

from src.core.database import get_db
from src.core.redis_client import signal_cache
from src.core.kafka_client import KafkaEventPublisher
from src.models import Signal, SignalDirection, RiskLevel, SignalStatus

router = APIRouter()
logger = structlog.get_logger()


class SignalCreate(BaseModel):
    """Signal creation request."""
    source_channel_id: int
    original_message_id: int
    pair: str = Field(..., max_length=20)
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    risk_level: Optional[RiskLevel] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=100)
    justification: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class SignalResponse(BaseModel):
    """Signal response model."""
    id: UUID
    source_channel_id: int
    original_message_id: int
    pair: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    risk_level: Optional[RiskLevel]
    confidence_score: Optional[float]
    justification: Dict[str, Any]
    status: SignalStatus
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class SignalUpdate(BaseModel):
    """Signal update request."""
    status: Optional[SignalStatus] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = None


@router.post("/", response_model=SignalResponse)
async def create_signal(
    signal_data: SignalCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Signal:
    """Create a new signal."""
    try:
        # Check for duplicate
        existing = await db.execute(
            select(Signal).where(
                and_(
                    Signal.source_channel_id == signal_data.source_channel_id,
                    Signal.original_message_id == signal_data.original_message_id
                )
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Signal already exists")
        
        # Create signal
        signal = Signal(
            **signal_data.dict(),
            take_profits=signal_data.take_profits  # Store as JSONB
        )
        
        db.add(signal)
        await db.commit()
        await db.refresh(signal)
        
        # Cache the signal
        await signal_cache.set(
            f"signal:{signal.id}",
            signal_data.dict(),
            ttl=3600
        )
        
        # Publish event
        background_tasks.add_task(
            KafkaEventPublisher.publish_signal,
            {"signal_id": str(signal.id), "pair": signal.pair}
        )
        
        logger.info("Signal created", signal_id=signal.id, pair=signal.pair)
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create signal", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create signal")


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Signal:
    """Get signal by ID."""
    # Try cache first
    cached = await signal_cache.get(f"signal:{signal_id}")
    if cached:
        return SignalResponse(**cached)
    
    # Get from database
    result = await db.execute(
        select(Signal).where(Signal.id == signal_id)
    )
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return signal


@router.get("/", response_model=List[SignalResponse])
async def list_signals(
    pair: Optional[str] = None,
    status: Optional[SignalStatus] = None,
    direction: Optional[SignalDirection] = None,
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    channel_id: Optional[int] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
) -> List[Signal]:
    """List signals with filters."""
    query = select(Signal)
    
    # Apply filters
    conditions = []
    if pair:
        conditions.append(Signal.pair == pair.upper())
    if status:
        conditions.append(Signal.status == status)
    if direction:
        conditions.append(Signal.direction == direction)
    if min_confidence is not None:
        conditions.append(Signal.confidence_score >= min_confidence)
    if channel_id:
        conditions.append(Signal.source_channel_id == channel_id)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    # Order and paginate
    query = query.order_by(Signal.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.patch("/{signal_id}", response_model=SignalResponse)
async def update_signal(
    signal_id: UUID,
    update_data: SignalUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Signal:
    """Update signal status or metadata."""
    result = await db.execute(
        select(Signal).where(Signal.id == signal_id)
    )
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Update fields
    if update_data.status is not None:
        signal.status = update_data.status
    if update_data.confidence_score is not None:
        signal.confidence_score = update_data.confidence_score
    if update_data.metadata is not None:
        signal.metadata = {**(signal.metadata or {}), **update_data.metadata}
    
    signal.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(signal)
    
    # Invalidate cache
    await signal_cache.delete(f"signal:{signal_id}")
    
    # Publish update event
    if update_data.status:
        background_tasks.add_task(
            KafkaEventPublisher.publish_validation,
            {"signal_id": str(signal_id), "status": update_data.status}
        )
    
    return signal


@router.delete("/{signal_id}")
async def delete_signal(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Delete a signal (soft delete by changing status)."""
    result = await db.execute(
        select(Signal).where(Signal.id == signal_id)
    )
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal.status = SignalStatus.CANCELLED
    signal.updated_at = datetime.utcnow()
    
    await db.commit()
    
    # Invalidate cache
    await signal_cache.delete(f"signal:{signal_id}")
    
    return {"message": "Signal cancelled successfully"}


@router.get("/stats/summary")
async def get_signal_stats(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get signal statistics summary."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get counts by status
    status_counts = await db.execute(
        select(
            Signal.status,
            func.count(Signal.id).label('count')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.status)
    )
    
    # Get average confidence by direction
    confidence_stats = await db.execute(
        select(
            Signal.direction,
            func.avg(Signal.confidence_score).label('avg_confidence'),
            func.count(Signal.id).label('count')
        ).where(
            and_(
                Signal.created_at >= since,
                Signal.confidence_score.isnot(None)
            )
        ).group_by(Signal.direction)
    )
    
    # Get top pairs
    top_pairs = await db.execute(
        select(
            Signal.pair,
            func.count(Signal.id).label('count')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.pair).order_by(func.count(Signal.id).desc()).limit(10)
    )
    
    return {
        "period_days": days,
        "status_distribution": {row.status: row.count for row in status_counts},
        "direction_stats": [
            {
                "direction": row.direction,
                "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
                "count": row.count
            }
            for row in confidence_stats
        ],
        "top_pairs": [{"pair": row.pair, "count": row.count} for row in top_pairs]
    }
