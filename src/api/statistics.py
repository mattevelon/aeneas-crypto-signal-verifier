"""
Statistics and analytics API endpoints.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import structlog

from src.core.database import get_db
from src.core.redis_client import get_redis
from src.models import Signal, SignalStatus, SignalDirection, RiskLevel
from src.api.auth import get_current_user
from src.performance_tracking.performance_dashboard import PerformanceDashboard

router = APIRouter()
logger = structlog.get_logger()


@router.get("/overview")
async def get_system_overview(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get system overview statistics."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Total signals
    total_signals = await db.execute(
        select(func.count(Signal.id)).where(Signal.created_at >= since)
    )
    
    # Active signals
    active_signals = await db.execute(
        select(func.count(Signal.id)).where(
            and_(
                Signal.created_at >= since,
                Signal.status == SignalStatus.ACTIVE
            )
        )
    )
    
    # Average confidence
    avg_confidence = await db.execute(
        select(func.avg(Signal.confidence_score)).where(
            and_(
                Signal.created_at >= since,
                Signal.confidence_score.isnot(None)
            )
        )
    )
    
    # Success rate (signals with positive outcome)
    success_rate = await db.execute(
        select(
            func.count(Signal.id).filter(Signal.status == SignalStatus.COMPLETED),
            func.count(Signal.id)
        ).where(Signal.created_at >= since)
    )
    
    success_count, total_count = success_rate.one()
    
    # Get performance metrics from dashboard
    dashboard = PerformanceDashboard(db, get_redis())
    performance_metrics = await dashboard.get_aggregated_metrics(
        time_range=timedelta(days=days)
    )
    
    return {
        "period_days": days,
        "total_signals": total_signals.scalar() or 0,
        "active_signals": active_signals.scalar() or 0,
        "average_confidence": float(avg_confidence.scalar() or 0),
        "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
        "performance_metrics": performance_metrics
    }


@router.get("/channels")
async def get_channel_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get statistics by channel."""
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            Signal.source_channel_id,
            func.count(Signal.id).label('total_signals'),
            func.avg(Signal.confidence_score).label('avg_confidence'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.COMPLETED).label('successful'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.FAILED).label('failed'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.ACTIVE).label('active')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.source_channel_id)
    )
    
    channels = []
    for row in result:
        success_rate = 0
        if row.total_signals > 0:
            success_rate = (row.successful / row.total_signals) * 100
        
        channels.append({
            "channel_id": row.source_channel_id,
            "total_signals": row.total_signals,
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
            "successful": row.successful,
            "failed": row.failed,
            "active": row.active,
            "success_rate": success_rate
        })
    
    return sorted(channels, key=lambda x: x['total_signals'], reverse=True)


@router.get("/performance")
async def get_performance_metrics(
    days: int = Query(30, ge=1, le=365),
    pair: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed performance metrics."""
    dashboard = PerformanceDashboard(db, get_redis())
    
    # Get performance metrics
    metrics = await dashboard.get_aggregated_metrics(
        time_range=timedelta(days=days),
        trading_pair=pair
    )
    
    # Get P&L breakdown
    pnl_breakdown = await dashboard.get_portfolio_analytics()
    
    # Get top performers
    top_performers = await dashboard.get_top_performers(limit=10)
    
    # Get performance trends
    trends = await dashboard.get_performance_trends(days=days)
    
    return {
        "period_days": days,
        "pair": pair,
        "metrics": metrics,
        "pnl_breakdown": pnl_breakdown,
        "top_performers": top_performers,
        "trends": trends
    }


@router.get("/risk-metrics")
async def get_risk_metrics(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get risk management metrics."""
    since = datetime.utcnow() - timedelta(days=days)
    
    # Risk distribution
    risk_distribution = await db.execute(
        select(
            Signal.risk_level,
            func.count(Signal.id).label('count')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.risk_level)
    )
    
    # Average risk by direction
    avg_risk_by_direction = await db.execute(
        select(
            Signal.direction,
            func.avg(
                func.case(
                    (Signal.risk_level == RiskLevel.LOW, 1),
                    (Signal.risk_level == RiskLevel.MEDIUM, 2),
                    (Signal.risk_level == RiskLevel.HIGH, 3),
                    else_=0
                )
            ).label('avg_risk_score')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.direction)
    )
    
    # Get stop loss statistics
    stop_loss_stats = await db.execute(
        select(
            func.avg(
                func.abs((Signal.entry_price - Signal.stop_loss) / Signal.entry_price * 100)
            ).label('avg_stop_loss_pct'),
            func.min(
                func.abs((Signal.entry_price - Signal.stop_loss) / Signal.entry_price * 100)
            ).label('min_stop_loss_pct'),
            func.max(
                func.abs((Signal.entry_price - Signal.stop_loss) / Signal.entry_price * 100)
            ).label('max_stop_loss_pct')
        ).where(
            Signal.created_at >= since
        )
    )
    
    stop_loss_data = stop_loss_stats.one()
    
    return {
        "period_days": days,
        "risk_distribution": {
            str(row.risk_level): row.count for row in risk_distribution
        },
        "avg_risk_by_direction": [
            {
                "direction": str(row.direction),
                "avg_risk_score": float(row.avg_risk_score) if row.avg_risk_score else 0
            }
            for row in avg_risk_by_direction
        ],
        "stop_loss_statistics": {
            "avg_stop_loss_pct": float(stop_loss_data.avg_stop_loss_pct) if stop_loss_data.avg_stop_loss_pct else 0,
            "min_stop_loss_pct": float(stop_loss_data.min_stop_loss_pct) if stop_loss_data.min_stop_loss_pct else 0,
            "max_stop_loss_pct": float(stop_loss_data.max_stop_loss_pct) if stop_loss_data.max_stop_loss_pct else 0
        }
    }


@router.get("/pairs")
async def get_pair_statistics(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get statistics by trading pair."""
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            Signal.pair,
            func.count(Signal.id).label('total_signals'),
            func.avg(Signal.confidence_score).label('avg_confidence'),
            func.count(Signal.id).filter(Signal.direction == SignalDirection.LONG).label('long_count'),
            func.count(Signal.id).filter(Signal.direction == SignalDirection.SHORT).label('short_count'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.COMPLETED).label('successful')
        ).where(
            Signal.created_at >= since
        ).group_by(Signal.pair).order_by(func.count(Signal.id).desc()).limit(limit)
    )
    
    pairs = []
    for row in result:
        success_rate = 0
        if row.total_signals > 0:
            success_rate = (row.successful / row.total_signals) * 100
        
        pairs.append({
            "pair": row.pair,
            "total_signals": row.total_signals,
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
            "long_count": row.long_count,
            "short_count": row.short_count,
            "long_short_ratio": row.long_count / row.short_count if row.short_count > 0 else float('inf'),
            "success_rate": success_rate
        })
    
    return pairs


@router.get("/daily")
async def get_daily_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get daily signal statistics."""
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            func.date(Signal.created_at).label('date'),
            func.count(Signal.id).label('total_signals'),
            func.avg(Signal.confidence_score).label('avg_confidence'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.COMPLETED).label('successful'),
            func.count(Signal.id).filter(Signal.status == SignalStatus.FAILED).label('failed')
        ).where(
            Signal.created_at >= since
        ).group_by(func.date(Signal.created_at)).order_by(func.date(Signal.created_at))
    )
    
    daily_stats = []
    for row in result:
        success_rate = 0
        if row.total_signals > 0:
            success_rate = (row.successful / row.total_signals) * 100
        
        daily_stats.append({
            "date": row.date.isoformat() if row.date else None,
            "total_signals": row.total_signals,
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
            "successful": row.successful,
            "failed": row.failed,
            "success_rate": success_rate
        })
    
    return daily_stats


@router.get("/cache-metrics")
async def get_cache_metrics(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache performance metrics."""
    redis_client = get_redis()
    
    # Get cache info
    info = await redis_client.info()
    
    # Get memory stats
    memory_stats = await redis_client.memory_stats()
    
    return {
        "cache_hits": info.get("keyspace_hits", 0),
        "cache_misses": info.get("keyspace_misses", 0),
        "hit_rate": (
            info.get("keyspace_hits", 0) / 
            (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
        ) * 100 if info.get("keyspace_hits", 0) > 0 else 0,
        "memory_used": memory_stats.get("used_memory_human", "0"),
        "memory_peak": memory_stats.get("used_memory_peak_human", "0"),
        "connected_clients": info.get("connected_clients", 0),
        "evicted_keys": info.get("evicted_keys", 0)
    }
