"""
Channel management API endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from pydantic import BaseModel, Field
import structlog

from src.core.database import get_db
from src.models import ChannelStatistics

router = APIRouter()
logger = structlog.get_logger()


class ChannelStatsResponse(BaseModel):
    """Channel statistics response."""
    channel_id: int
    channel_name: Optional[str]
    total_signals: int
    successful_signals: int
    failed_signals: int
    average_confidence: Optional[float]
    reputation_score: Optional[float]
    last_signal_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    success_rate: Optional[float] = None
    
    class Config:
        from_attributes = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_signals > 0:
            self.success_rate = (self.successful_signals / self.total_signals) * 100


class ChannelUpdate(BaseModel):
    """Channel update request."""
    channel_name: Optional[str] = Field(None, max_length=255)
    reputation_score: Optional[float] = Field(None, ge=0, le=100)


@router.get("/{channel_id}/stats", response_model=ChannelStatsResponse)
async def get_channel_stats(
    channel_id: int,
    db: AsyncSession = Depends(get_db)
) -> ChannelStatistics:
    """Get channel statistics."""
    result = await db.execute(
        select(ChannelStatistics).where(ChannelStatistics.channel_id == channel_id)
    )
    stats = result.scalar_one_or_none()
    
    if not stats:
        # Create default stats if not exist
        stats = ChannelStatistics(channel_id=channel_id)
        db.add(stats)
        await db.commit()
        await db.refresh(stats)
    
    return stats


@router.get("/", response_model=List[ChannelStatsResponse])
async def list_channels(
    min_signals: Optional[int] = Query(None, ge=0),
    min_reputation: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
) -> List[ChannelStatistics]:
    """List channels with statistics."""
    query = select(ChannelStatistics)
    
    # Apply filters
    if min_signals is not None:
        query = query.where(ChannelStatistics.total_signals >= min_signals)
    if min_reputation is not None:
        query = query.where(ChannelStatistics.reputation_score >= min_reputation)
    
    # Order by reputation and total signals
    query = query.order_by(
        ChannelStatistics.reputation_score.desc().nullslast(),
        ChannelStatistics.total_signals.desc()
    ).limit(limit).offset(offset)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.patch("/{channel_id}", response_model=ChannelStatsResponse)
async def update_channel(
    channel_id: int,
    update_data: ChannelUpdate,
    db: AsyncSession = Depends(get_db)
) -> ChannelStatistics:
    """Update channel information."""
    result = await db.execute(
        select(ChannelStatistics).where(ChannelStatistics.channel_id == channel_id)
    )
    stats = result.scalar_one_or_none()
    
    if not stats:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Update fields
    if update_data.channel_name is not None:
        stats.channel_name = update_data.channel_name
    if update_data.reputation_score is not None:
        stats.reputation_score = update_data.reputation_score
    
    stats.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(stats)
    
    return stats


@router.get("/top-performers")
async def get_top_performers(
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get top performing channels."""
    result = await db.execute(
        select(
            ChannelStatistics.channel_id,
            ChannelStatistics.channel_name,
            ChannelStatistics.total_signals,
            ChannelStatistics.successful_signals,
            ChannelStatistics.reputation_score,
            (ChannelStatistics.successful_signals * 100.0 / 
             func.nullif(ChannelStatistics.total_signals, 0)).label('success_rate')
        ).where(
            ChannelStatistics.total_signals > 0
        ).order_by(
            func.coalesce(ChannelStatistics.reputation_score, 0).desc()
        ).limit(limit)
    )
    
    return [
        {
            "channel_id": row.channel_id,
            "channel_name": row.channel_name,
            "total_signals": row.total_signals,
            "successful_signals": row.successful_signals,
            "reputation_score": row.reputation_score,
            "success_rate": float(row.success_rate) if row.success_rate else 0
        }
        for row in result
    ]
