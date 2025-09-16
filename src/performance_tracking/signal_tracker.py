"""
Signal Outcome Tracker

Tracks the real-world performance of signals from initiation to completion,
monitoring entry/exit prices, timing, and actual vs predicted outcomes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid
from decimal import Decimal
import logging
import json

from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from pydantic import BaseModel, Field

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.market_data import MarketDataClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SignalOutcome(str, Enum):
    """Signal outcome types"""
    PENDING = "pending"
    IN_POSITION = "in_position"
    PROFIT = "profit"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIALLY_FILLED = "partially_filled"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"


class PositionStatus(str, Enum):
    """Position status types"""
    WAITING_ENTRY = "waiting_entry"
    ACTIVE = "active"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"


class TrackingMetadata(BaseModel):
    """Metadata for signal tracking"""
    signal_id: str
    start_time: datetime
    entry_attempts: int = 0
    last_check: Optional[datetime] = None
    price_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    execution_notes: List[str] = Field(default_factory=list)
    risk_adjustments: List[Dict[str, Any]] = Field(default_factory=list)


class SignalOutcomeTracker:
    """
    Tracks signal outcomes and performance metrics
    """
    
    def __init__(self, market_client: Optional[MarketDataClient] = None):
        self.redis = get_redis()
        self.market_client = market_client or MarketDataClient()
        self.tracking_interval = 60  # Check every minute
        self.max_tracking_duration = timedelta(days=7)  # Stop tracking after 7 days
        self._tracking_tasks = {}
        
    async def start_tracking(self, signal_id: str) -> bool:
        """
        Start tracking a signal's performance
        
        Args:
            signal_id: Unique signal identifier
            
        Returns:
            bool: Success status
        """
        try:
            # Check if already tracking
            if signal_id in self._tracking_tasks:
                logger.warning(f"Signal {signal_id} is already being tracked")
                return False
            
            # Fetch signal from database
            async with get_async_session() as session:
                signal = await self._get_signal(session, signal_id)
                if not signal:
                    logger.error(f"Signal {signal_id} not found")
                    return False
            
            # Initialize tracking metadata
            metadata = TrackingMetadata(
                signal_id=signal_id,
                start_time=datetime.utcnow()
            )
            
            # Store in Redis
            await self._save_metadata(metadata)
            
            # Start background tracking task
            task = asyncio.create_task(self._track_signal(signal_id))
            self._tracking_tasks[signal_id] = task
            
            logger.info(f"Started tracking signal {signal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting tracking for {signal_id}: {str(e)}")
            return False
    
    async def stop_tracking(self, signal_id: str) -> bool:
        """
        Stop tracking a signal
        
        Args:
            signal_id: Signal to stop tracking
            
        Returns:
            bool: Success status
        """
        try:
            if signal_id in self._tracking_tasks:
                task = self._tracking_tasks[signal_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._tracking_tasks[signal_id]
                
            # Clear Redis metadata
            await self.redis.delete(f"tracking:{signal_id}")
            
            logger.info(f"Stopped tracking signal {signal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping tracking for {signal_id}: {str(e)}")
            return False
    
    async def _track_signal(self, signal_id: str):
        """
        Background task to continuously track a signal
        """
        try:
            while True:
                # Load metadata
                metadata = await self._load_metadata(signal_id)
                if not metadata:
                    break
                
                # Check if exceeded max tracking duration
                if datetime.utcnow() - metadata.start_time > self.max_tracking_duration:
                    await self._finalize_tracking(signal_id, SignalOutcome.EXPIRED)
                    break
                
                # Update signal status
                outcome = await self._check_signal_status(signal_id, metadata)
                
                # Update metadata
                metadata.last_check = datetime.utcnow()
                await self._save_metadata(metadata)
                
                # Check if signal completed
                if outcome in [SignalOutcome.PROFIT, SignalOutcome.LOSS, 
                              SignalOutcome.STOP_LOSS_HIT, SignalOutcome.TAKE_PROFIT_HIT,
                              SignalOutcome.CANCELLED, SignalOutcome.EXPIRED]:
                    await self._finalize_tracking(signal_id, outcome)
                    break
                
                # Wait for next check
                await asyncio.sleep(self.tracking_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Tracking cancelled for signal {signal_id}")
        except Exception as e:
            logger.error(f"Error in tracking loop for {signal_id}: {str(e)}")
    
    async def _check_signal_status(self, signal_id: str, metadata: TrackingMetadata) -> SignalOutcome:
        """
        Check current status of a signal
        
        Args:
            signal_id: Signal identifier
            metadata: Tracking metadata
            
        Returns:
            SignalOutcome: Current outcome status
        """
        try:
            async with get_async_session() as session:
                # Get signal and performance data
                signal = await self._get_signal(session, signal_id)
                if not signal:
                    return SignalOutcome.CANCELLED
                
                performance = await self._get_or_create_performance(session, signal_id)
                
                # Get current market price
                current_price = await self.market_client.get_current_price(signal.pair)
                if not current_price:
                    logger.warning(f"Could not get current price for {signal.pair}")
                    return SignalOutcome.PENDING
                
                # Check entry conditions
                if not performance.actual_entry:
                    if await self._check_entry_conditions(signal, current_price, metadata):
                        performance.actual_entry = current_price
                        performance.entry_time = datetime.utcnow()
                        metadata.entry_attempts += 1
                        await session.commit()
                        return SignalOutcome.IN_POSITION
                    return SignalOutcome.PENDING
                
                # Check exit conditions
                outcome = await self._check_exit_conditions(
                    signal, performance, current_price, session
                )
                
                # Update performance metrics
                await self._update_performance_metrics(
                    performance, signal, current_price, outcome, session
                )
                
                await session.commit()
                return outcome
                
        except Exception as e:
            logger.error(f"Error checking signal status: {str(e)}")
            return SignalOutcome.PENDING
    
    async def _check_entry_conditions(self, signal: Signal, current_price: float, 
                                      metadata: TrackingMetadata) -> bool:
        """
        Check if entry conditions are met
        
        Args:
            signal: Signal object
            current_price: Current market price
            metadata: Tracking metadata
            
        Returns:
            bool: True if should enter position
        """
        # Price within acceptable range of entry price (2% default)
        entry_tolerance = 0.02
        price_diff = abs(current_price - float(signal.entry_price)) / float(signal.entry_price)
        
        if price_diff <= entry_tolerance:
            # Check volume conditions
            volume_data = await self.market_client.get_24h_volume(signal.pair)
            min_volume = 100000  # $100k minimum
            
            if volume_data and volume_data.get('volume', 0) >= min_volume:
                # Record entry conditions
                metadata.market_conditions = {
                    'entry_price': current_price,
                    'target_price': float(signal.entry_price),
                    'price_diff': price_diff,
                    'volume': volume_data.get('volume', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
                return True
        
        return False
    
    async def _check_exit_conditions(self, signal: Signal, performance: SignalPerformance,
                                     current_price: float, session: AsyncSession) -> SignalOutcome:
        """
        Check exit conditions for an active position
        
        Args:
            signal: Signal object
            performance: Performance tracking object
            current_price: Current market price
            session: Database session
            
        Returns:
            SignalOutcome: Current outcome
        """
        entry_price = float(performance.actual_entry)
        
        # Calculate P&L
        if signal.direction == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check stop loss
        if signal.stop_loss:
            sl_price = float(signal.stop_loss)
            if (signal.direction == 'long' and current_price <= sl_price) or \
               (signal.direction == 'short' and current_price >= sl_price):
                performance.actual_exit = current_price
                performance.hit_stop_loss = True
                performance.pnl_percentage = pnl_pct
                return SignalOutcome.STOP_LOSS_HIT
        
        # Check take profit targets
        take_profits = signal.take_profits or []
        for i, tp in enumerate(take_profits):
            tp_price = tp.get('price')
            if tp_price:
                if (signal.direction == 'long' and current_price >= tp_price) or \
                   (signal.direction == 'short' and current_price <= tp_price):
                    performance.actual_exit = current_price
                    performance.hit_take_profit = i + 1
                    performance.pnl_percentage = pnl_pct
                    return SignalOutcome.TAKE_PROFIT_HIT
        
        # Determine current status based on P&L
        if pnl_pct > 0.1:  # > 0.1% profit
            return SignalOutcome.IN_POSITION  # Still profitable, keep tracking
        elif pnl_pct < -0.1:  # < -0.1% loss
            return SignalOutcome.IN_POSITION  # In loss but not stopped out
        else:
            return SignalOutcome.IN_POSITION  # Near breakeven
    
    async def _update_performance_metrics(self, performance: SignalPerformance, signal: Signal,
                                          current_price: float, outcome: SignalOutcome,
                                          session: AsyncSession):
        """
        Update performance metrics for a signal
        
        Args:
            performance: Performance object to update
            signal: Signal object
            current_price: Current market price
            outcome: Current outcome status
            session: Database session
        """
        if performance.actual_entry:
            entry_price = float(performance.actual_entry)
            
            # Calculate current P&L
            if signal.direction == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                pnl_amount = (current_price - entry_price) * performance.position_size if performance.position_size else 0
            else:  # short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                pnl_amount = (entry_price - current_price) * performance.position_size if performance.position_size else 0
            
            # Update metrics
            performance.pnl_percentage = pnl_pct
            performance.pnl_amount = Decimal(str(pnl_amount))
            performance.max_profit = max(performance.max_profit or 0, pnl_pct)
            performance.max_drawdown = min(performance.max_drawdown or 0, pnl_pct)
            
            # Calculate duration if closed
            if outcome in [SignalOutcome.PROFIT, SignalOutcome.LOSS, 
                          SignalOutcome.STOP_LOSS_HIT, SignalOutcome.TAKE_PROFIT_HIT]:
                if performance.entry_time:
                    duration = datetime.utcnow() - performance.entry_time
                    performance.duration_hours = int(duration.total_seconds() / 3600)
                performance.closed_at = datetime.utcnow()
    
    async def _finalize_tracking(self, signal_id: str, outcome: SignalOutcome):
        """
        Finalize tracking for a completed signal
        
        Args:
            signal_id: Signal identifier
            outcome: Final outcome
        """
        try:
            # Update signal status
            async with get_async_session() as session:
                await session.execute(
                    update(Signal)
                    .where(Signal.id == uuid.UUID(signal_id))
                    .values(
                        status='closed' if outcome != SignalOutcome.CANCELLED else 'cancelled',
                        updated_at=datetime.utcnow()
                    )
                )
                
                # Update performance with final outcome
                await session.execute(
                    update(SignalPerformance)
                    .where(SignalPerformance.signal_id == uuid.UUID(signal_id))
                    .values(
                        outcome=outcome.value,
                        closed_at=datetime.utcnow()
                    )
                )
                
                await session.commit()
            
            # Generate performance report
            report = await self.generate_performance_report(signal_id)
            
            # Cache report
            await self.redis.setex(
                f"performance_report:{signal_id}",
                86400,  # 24 hours
                json.dumps(report, default=str)
            )
            
            # Clean up tracking
            await self.stop_tracking(signal_id)
            
            logger.info(f"Finalized tracking for signal {signal_id} with outcome {outcome}")
            
        except Exception as e:
            logger.error(f"Error finalizing tracking: {str(e)}")
    
    async def generate_performance_report(self, signal_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive performance report for a signal
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Dict containing performance metrics
        """
        try:
            async with get_async_session() as session:
                # Get signal and performance data
                signal = await self._get_signal(session, signal_id)
                performance = await self._get_performance(session, signal_id)
                
                if not signal or not performance:
                    return {}
                
                # Calculate metrics
                report = {
                    'signal_id': signal_id,
                    'pair': signal.pair,
                    'direction': signal.direction,
                    'entry_price': float(signal.entry_price) if signal.entry_price else None,
                    'actual_entry': float(performance.actual_entry) if performance.actual_entry else None,
                    'actual_exit': float(performance.actual_exit) if performance.actual_exit else None,
                    'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                    'hit_stop_loss': performance.hit_stop_loss,
                    'hit_take_profit': performance.hit_take_profit,
                    'pnl_percentage': float(performance.pnl_percentage) if performance.pnl_percentage else 0,
                    'pnl_amount': float(performance.pnl_amount) if performance.pnl_amount else 0,
                    'duration_hours': performance.duration_hours,
                    'max_profit': float(performance.max_profit) if performance.max_profit else 0,
                    'max_drawdown': float(performance.max_drawdown) if performance.max_drawdown else 0,
                    'outcome': performance.outcome,
                    'entry_time': performance.entry_time.isoformat() if performance.entry_time else None,
                    'exit_time': performance.closed_at.isoformat() if performance.closed_at else None,
                    'tracking_metadata': await self._load_metadata_dict(signal_id)
                }
                
                # Calculate additional metrics
                if performance.actual_entry and performance.actual_exit:
                    report['slippage_entry'] = abs(float(signal.entry_price) - float(performance.actual_entry)) / float(signal.entry_price) * 100
                    report['risk_reward_achieved'] = abs(float(performance.pnl_percentage)) / abs(float(signal.stop_loss) - float(signal.entry_price)) * float(signal.entry_price) / 100 if signal.stop_loss else None
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {}
    
    async def get_tracking_summary(self) -> Dict[str, Any]:
        """
        Get summary of all currently tracked signals
        
        Returns:
            Dict containing tracking summary
        """
        try:
            active_signals = list(self._tracking_tasks.keys())
            
            summaries = []
            for signal_id in active_signals:
                metadata = await self._load_metadata(signal_id)
                if metadata:
                    summaries.append({
                        'signal_id': signal_id,
                        'start_time': metadata.start_time.isoformat(),
                        'last_check': metadata.last_check.isoformat() if metadata.last_check else None,
                        'entry_attempts': metadata.entry_attempts,
                        'alerts_count': len(metadata.price_alerts)
                    })
            
            return {
                'total_tracking': len(active_signals),
                'signals': summaries,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting tracking summary: {str(e)}")
            return {}
    
    # Helper methods
    async def _get_signal(self, session: AsyncSession, signal_id: str) -> Optional[Signal]:
        """Get signal from database"""
        result = await session.execute(
            select(Signal).where(Signal.id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_performance(self, session: AsyncSession, signal_id: str) -> Optional[SignalPerformance]:
        """Get performance record from database"""
        result = await session.execute(
            select(SignalPerformance).where(SignalPerformance.signal_id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_or_create_performance(self, session: AsyncSession, signal_id: str) -> SignalPerformance:
        """Get or create performance record"""
        performance = await self._get_performance(session, signal_id)
        if not performance:
            performance = SignalPerformance(
                signal_id=uuid.UUID(signal_id),
                outcome=SignalOutcome.PENDING.value,
                created_at=datetime.utcnow()
            )
            session.add(performance)
        return performance
    
    async def _save_metadata(self, metadata: TrackingMetadata):
        """Save tracking metadata to Redis"""
        await self.redis.setex(
            f"tracking:{metadata.signal_id}",
            86400,  # 24 hours
            metadata.json()
        )
    
    async def _load_metadata(self, signal_id: str) -> Optional[TrackingMetadata]:
        """Load tracking metadata from Redis"""
        data = await self.redis.get(f"tracking:{signal_id}")
        if data:
            return TrackingMetadata.parse_raw(data)
        return None
    
    async def _load_metadata_dict(self, signal_id: str) -> Dict[str, Any]:
        """Load tracking metadata as dictionary"""
        metadata = await self._load_metadata(signal_id)
        if metadata:
            return metadata.dict()
        return {}
