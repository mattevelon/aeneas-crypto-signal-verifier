"""
Performance tracking API endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import BaseModel, Field
import structlog

from src.core.database import get_db
from src.models import SignalPerformance, Signal

router = APIRouter()
logger = structlog.get_logger()


class PerformanceCreate(BaseModel):
    """Performance record creation."""
    signal_id: UUID
    actual_entry: float
    actual_exit: float
    pnl_percentage: float
    pnl_amount: float
    hit_stop_loss: bool = False
    hit_take_profit: Optional[int] = None
    duration_hours: int
    closed_at: datetime


class PerformanceResponse(BaseModel):
    """Performance response model."""
    id: UUID
    signal_id: UUID
    actual_entry: float
    actual_exit: float
    pnl_percentage: float
    pnl_amount: float
    hit_stop_loss: bool
    hit_take_profit: Optional[int]
    duration_hours: int
    closed_at: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


@router.post("/", response_model=PerformanceResponse)
async def record_performance(
    performance_data: PerformanceCreate,
    db: AsyncSession = Depends(get_db)
) -> SignalPerformance:
    """Record signal performance."""
    # Check if signal exists
    signal_result = await db.execute(
        select(Signal).where(Signal.id == performance_data.signal_id)
    )
    signal = signal_result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Check if performance already recorded
    existing = await db.execute(
        select(SignalPerformance).where(
            SignalPerformance.signal_id == performance_data.signal_id
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Performance already recorded")
    
    # Create performance record
    performance = SignalPerformance(**performance_data.dict())
    db.add(performance)
    
    # Update signal status
    signal.status = "closed"
    
    await db.commit()
    await db.refresh(performance)
    
    logger.info("Performance recorded", signal_id=performance_data.signal_id)
    return performance


@router.get("/{signal_id}", response_model=PerformanceResponse)
async def get_signal_performance(
    signal_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> SignalPerformance:
    """Get performance for a specific signal."""
    result = await db.execute(
        select(SignalPerformance).where(SignalPerformance.signal_id == signal_id)
    )
    performance = result.scalar_one_or_none()
    
    if not performance:
        raise HTTPException(status_code=404, detail="Performance not found")
    
    return performance


@router.get("/", response_model=List[PerformanceResponse])
async def list_performance(
    min_pnl: Optional[float] = None,
    max_pnl: Optional[float] = None,
    hit_stop_loss: Optional[bool] = None,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
) -> List[SignalPerformance]:
    """List performance records with filters."""
    since = datetime.utcnow() - timedelta(days=days)
    query = select(SignalPerformance).where(SignalPerformance.closed_at >= since)
    
    # Apply filters
    conditions = []
    if min_pnl is not None:
        conditions.append(SignalPerformance.pnl_percentage >= min_pnl)
    if max_pnl is not None:
        conditions.append(SignalPerformance.pnl_percentage <= max_pnl)
    if hit_stop_loss is not None:
        conditions.append(SignalPerformance.hit_stop_loss == hit_stop_loss)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    # Order and paginate
    query = query.order_by(SignalPerformance.closed_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/stats/summary")
async def get_performance_stats(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance statistics summary."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get overall stats
    overall_stats = await db.execute(
        select(
            func.count(SignalPerformance.id).label('total_trades'),
            func.sum(func.case((SignalPerformance.pnl_percentage > 0, 1), else_=0)).label('winning_trades'),
            func.sum(func.case((SignalPerformance.pnl_percentage < 0, 1), else_=0)).label('losing_trades'),
            func.avg(SignalPerformance.pnl_percentage).label('avg_pnl'),
            func.max(SignalPerformance.pnl_percentage).label('max_pnl'),
            func.min(SignalPerformance.pnl_percentage).label('min_pnl'),
            func.avg(SignalPerformance.duration_hours).label('avg_duration')
        ).where(SignalPerformance.closed_at >= since)
    )
    stats = overall_stats.one()
    
    # Get stop loss/take profit stats
    sl_tp_stats = await db.execute(
        select(
            func.sum(func.case((SignalPerformance.hit_stop_loss == True, 1), else_=0)).label('hit_stop_loss'),
            func.sum(func.case((SignalPerformance.hit_take_profit.isnot(None), 1), else_=0)).label('hit_take_profit')
        ).where(SignalPerformance.closed_at >= since)
    )
    sl_tp = sl_tp_stats.one()
    
    # Calculate win rate
    win_rate = 0
    if stats.total_trades:
        win_rate = (stats.winning_trades / stats.total_trades) * 100
    
    # Calculate profit factor
    winning_pnl = await db.execute(
        select(func.sum(SignalPerformance.pnl_amount)).where(
            and_(
                SignalPerformance.closed_at >= since,
                SignalPerformance.pnl_amount > 0
            )
        )
    )
    losing_pnl = await db.execute(
        select(func.abs(func.sum(SignalPerformance.pnl_amount))).where(
            and_(
                SignalPerformance.closed_at >= since,
                SignalPerformance.pnl_amount < 0
            )
        )
    )
    
    total_wins = winning_pnl.scalar() or 0
    total_losses = losing_pnl.scalar() or 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    return {
        "period_days": days,
        "total_trades": stats.total_trades or 0,
        "winning_trades": stats.winning_trades or 0,
        "losing_trades": stats.losing_trades or 0,
        "win_rate": round(win_rate, 2),
        "avg_pnl_percentage": round(float(stats.avg_pnl or 0), 2),
        "max_pnl_percentage": round(float(stats.max_pnl or 0), 2),
        "min_pnl_percentage": round(float(stats.min_pnl or 0), 2),
        "avg_duration_hours": round(float(stats.avg_duration or 0), 1),
        "hit_stop_loss_count": sl_tp.hit_stop_loss or 0,
        "hit_take_profit_count": sl_tp.hit_take_profit or 0,
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else None
    }
