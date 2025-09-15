"""
Cache warming strategies for improved performance.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.database import get_db_context
from src.core.redis_client import signal_cache, market_cache
from src.models import Signal, SignalStatus, ChannelStatistics

logger = structlog.get_logger()


class CacheWarmer:
    """Cache warming strategies for different data types."""
    
    def __init__(self):
        self.warming_interval = 300  # 5 minutes
        self.is_running = False
    
    async def start(self):
        """Start cache warming tasks."""
        self.is_running = True
        
        # Start warming tasks
        asyncio.create_task(self._warm_active_signals())
        asyncio.create_task(self._warm_channel_stats())
        asyncio.create_task(self._warm_top_pairs())
        
        logger.info("Cache warming started")
    
    async def stop(self):
        """Stop cache warming tasks."""
        self.is_running = False
        logger.info("Cache warming stopped")
    
    async def _warm_active_signals(self):
        """Warm cache with active signals."""
        while self.is_running:
            try:
                async with get_db_context() as db:
                    # Get active signals
                    result = await db.execute(
                        select(Signal)
                        .where(Signal.status == SignalStatus.ACTIVE)
                        .order_by(Signal.created_at.desc())
                        .limit(100)
                    )
                    signals = result.scalars().all()
                    
                    # Cache each signal
                    for signal in signals:
                        await signal_cache.set(
                            f"signal:{signal.id}",
                            {
                                "id": str(signal.id),
                                "pair": signal.pair,
                                "direction": signal.direction.value,
                                "entry_price": float(signal.entry_price),
                                "stop_loss": float(signal.stop_loss),
                                "take_profits": signal.take_profits,
                                "confidence_score": signal.confidence_score,
                                "status": signal.status.value,
                                "created_at": signal.created_at.isoformat()
                            },
                            ttl=3600
                        )
                    
                    logger.debug(f"Warmed {len(signals)} active signals")
                    
            except Exception as e:
                logger.error(f"Error warming active signals: {e}")
            
            await asyncio.sleep(self.warming_interval)
    
    async def _warm_channel_stats(self):
        """Warm cache with channel statistics."""
        while self.is_running:
            try:
                async with get_db_context() as db:
                    # Get channel statistics
                    result = await db.execute(
                        select(ChannelStatistics)
                        .order_by(ChannelStatistics.reputation_score.desc())
                        .limit(50)
                    )
                    channels = result.scalars().all()
                    
                    # Cache channel stats
                    for channel in channels:
                        await signal_cache.set(
                            f"channel_stats:{channel.channel_id}",
                            {
                                "channel_id": channel.channel_id,
                                "channel_name": channel.channel_name,
                                "total_signals": channel.total_signals,
                                "successful_signals": channel.successful_signals,
                                "failed_signals": channel.failed_signals,
                                "average_confidence": channel.average_confidence,
                                "reputation_score": channel.reputation_score,
                                "last_signal_at": channel.last_signal_at.isoformat() if channel.last_signal_at else None
                            },
                            ttl=7200  # 2 hours
                        )
                    
                    logger.debug(f"Warmed {len(channels)} channel statistics")
                    
            except Exception as e:
                logger.error(f"Error warming channel stats: {e}")
            
            await asyncio.sleep(self.warming_interval * 2)  # Less frequent
    
    async def _warm_top_pairs(self):
        """Warm cache with top trading pairs."""
        while self.is_running:
            try:
                async with get_db_context() as db:
                    # Get top pairs from last 24 hours
                    since = datetime.utcnow() - timedelta(hours=24)
                    result = await db.execute(
                        select(
                            Signal.pair,
                            func.count(Signal.id).label('count'),
                            func.avg(Signal.confidence_score).label('avg_confidence')
                        )
                        .where(Signal.created_at >= since)
                        .group_by(Signal.pair)
                        .order_by(func.count(Signal.id).desc())
                        .limit(20)
                    )
                    
                    top_pairs = []
                    for row in result:
                        pair_data = {
                            "pair": row.pair,
                            "count": row.count,
                            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0
                        }
                        top_pairs.append(pair_data)
                        
                        # Cache individual pair stats
                        await market_cache.set(
                            f"pair_stats:{row.pair}",
                            pair_data,
                            ttl=1800  # 30 minutes
                        )
                    
                    # Cache the top pairs list
                    await market_cache.set(
                        "top_pairs_24h",
                        top_pairs,
                        ttl=1800
                    )
                    
                    logger.debug(f"Warmed {len(top_pairs)} top pairs")
                    
            except Exception as e:
                logger.error(f"Error warming top pairs: {e}")
            
            await asyncio.sleep(self.warming_interval)
    
    async def warm_on_demand(self, data_type: str, key: str = None):
        """Warm specific data on demand."""
        try:
            if data_type == "signal" and key:
                async with get_db_context() as db:
                    result = await db.execute(
                        select(Signal).where(Signal.id == key)
                    )
                    signal = result.scalar_one_or_none()
                    if signal:
                        await signal_cache.set(
                            f"signal:{signal.id}",
                            {
                                "id": str(signal.id),
                                "pair": signal.pair,
                                "direction": signal.direction.value,
                                "entry_price": float(signal.entry_price),
                                "stop_loss": float(signal.stop_loss),
                                "take_profits": signal.take_profits,
                                "confidence_score": signal.confidence_score,
                                "status": signal.status.value
                            },
                            ttl=3600
                        )
                        logger.debug(f"Warmed signal {key} on demand")
            
            elif data_type == "recent_signals":
                await self._warm_active_signals()
                
            elif data_type == "channels":
                await self._warm_channel_stats()
                
            elif data_type == "pairs":
                await self._warm_top_pairs()
                
        except Exception as e:
            logger.error(f"Error in on-demand cache warming: {e}")


# Global cache warmer instance
cache_warmer = CacheWarmer()


async def start_cache_warming():
    """Start cache warming tasks."""
    await cache_warmer.start()


async def stop_cache_warming():
    """Stop cache warming tasks."""
    await cache_warmer.stop()


async def warm_cache(data_type: str, key: str = None):
    """Warm specific cache on demand."""
    await cache_warmer.warm_on_demand(data_type, key)
