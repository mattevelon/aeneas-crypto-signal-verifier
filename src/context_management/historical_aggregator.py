"""Historical Data Aggregator for collecting and processing past signal data."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_async_session

logger = logging.getLogger(__name__)


@dataclass
class HistoricalContext:
    """Historical context data for signal analysis."""
    similar_signals: List[Dict[str, Any]]
    channel_performance: Dict[str, float]
    pair_statistics: Dict[str, Any]
    time_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]


class HistoricalDataAggregator:
    """
    Aggregates historical data within a sliding window for context building.
    Implements 24-hour sliding window with efficient data sampling.
    """
    
    def __init__(self, window_hours: int = 24, sample_size: int = 100):
        """
        Initialize historical data aggregator.
        
        Args:
            window_hours: Hours to look back for historical data
            sample_size: Maximum number of samples for large datasets
        """
        self.window_hours = window_hours
        self.sample_size = sample_size
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        logger.info(f"Initialized HistoricalDataAggregator with {window_hours}h window")
    
    async def aggregate_context(self, 
                               pair: str,
                               channel_id: Optional[int] = None,
                               timestamp: Optional[datetime] = None) -> HistoricalContext:
        """
        Aggregate historical context for signal analysis.
        
        Args:
            pair: Trading pair to analyze
            channel_id: Source channel ID
            timestamp: Reference timestamp (default: now)
            
        Returns:
            Aggregated historical context
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check cache
        cache_key = f"{pair}_{channel_id}_{timestamp.date()}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        # Aggregate data components
        tasks = [
            self._get_similar_signals(pair, timestamp),
            self._get_channel_performance(channel_id, timestamp) if channel_id else self._get_default_performance(),
            self._get_pair_statistics(pair, timestamp),
            self._analyze_time_patterns(pair, timestamp),
            self._detect_anomalies(pair, timestamp),
            self._calculate_performance_metrics(pair, timestamp)
        ]
        
        results = await asyncio.gather(*tasks)
        
        context = HistoricalContext(
            similar_signals=results[0],
            channel_performance=results[1],
            pair_statistics=results[2],
            time_patterns=results[3],
            anomalies=results[4],
            performance_metrics=results[5]
        )
        
        # Cache result
        self.cache[cache_key] = (context, datetime.now())
        
        return context
    
    async def _get_similar_signals(self, pair: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get similar historical signals within the time window."""
        window_start = timestamp - timedelta(hours=self.window_hours)
        
        async with get_async_session() as session:
            # Query similar signals
            query = select('signals').where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start,
                    'signals.created_at' <= timestamp
                )
            ).order_by('signals.created_at.desc()').limit(self.sample_size)
            
            result = await session.execute(query)
            signals = result.fetchall()
            
            # Process and format signals
            processed_signals = []
            for signal in signals:
                processed_signals.append({
                    'id': signal.id,
                    'entry_price': float(signal.entry_price),
                    'stop_loss': float(signal.stop_loss),
                    'take_profits': signal.take_profits,
                    'direction': signal.direction,
                    'confidence_score': signal.confidence_score,
                    'created_at': signal.created_at.isoformat(),
                    'status': signal.status,
                    'performance': await self._get_signal_performance(signal.id, session)
                })
            
            return processed_signals
    
    async def _get_signal_performance(self, signal_id: str, session: AsyncSession) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific signal."""
        query = select('signal_performance').where(
            'signal_performance.signal_id' == signal_id
        )
        
        result = await session.execute(query)
        performance = result.fetchone()
        
        if performance:
            return {
                'pnl_percentage': performance.pnl_percentage,
                'hit_stop_loss': performance.hit_stop_loss,
                'hit_take_profit': performance.hit_take_profit,
                'duration_hours': performance.duration_hours
            }
        return None
    
    async def _get_channel_performance(self, channel_id: int, timestamp: datetime) -> Dict[str, float]:
        """Calculate channel performance metrics."""
        async with get_async_session() as session:
            query = select('channel_statistics').where(
                'channel_statistics.channel_id' == channel_id
            )
            
            result = await session.execute(query)
            stats = result.fetchone()
            
            if stats:
                total = stats.total_signals
                successful = stats.successful_signals
                failed = stats.failed_signals
                
                return {
                    'success_rate': successful / total if total > 0 else 0,
                    'failure_rate': failed / total if total > 0 else 0,
                    'total_signals': total,
                    'reputation_score': stats.reputation_score or 0.5,
                    'avg_confidence': stats.average_confidence or 0
                }
            
            return await self._get_default_performance()
    
    async def _get_default_performance(self) -> Dict[str, float]:
        """Return default performance metrics."""
        return {
            'success_rate': 0.5,
            'failure_rate': 0.5,
            'total_signals': 0,
            'reputation_score': 0.5,
            'avg_confidence': 50.0
        }
    
    async def _get_pair_statistics(self, pair: str, timestamp: datetime) -> Dict[str, Any]:
        """Calculate statistics for the trading pair."""
        window_start = timestamp - timedelta(hours=self.window_hours)
        
        async with get_async_session() as session:
            # Get signal statistics
            query = select(
                func.count('signals.id').label('total_signals'),
                func.avg('signals.confidence_score').label('avg_confidence'),
                func.count(func.distinct('signals.source_channel_id')).label('unique_channels')
            ).where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start
                )
            )
            
            result = await session.execute(query)
            stats = result.fetchone()
            
            # Get direction distribution
            direction_query = select(
                'signals.direction',
                func.count('signals.id').label('count')
            ).where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start
                )
            ).group_by('signals.direction')
            
            direction_result = await session.execute(direction_query)
            direction_dist = {row.direction: row.count for row in direction_result}
            
            return {
                'total_signals': stats.total_signals or 0,
                'avg_confidence': float(stats.avg_confidence or 0),
                'unique_channels': stats.unique_channels or 0,
                'direction_distribution': direction_dist,
                'long_bias': direction_dist.get('long', 0) / (stats.total_signals or 1),
                'short_bias': direction_dist.get('short', 0) / (stats.total_signals or 1)
            }
    
    async def _analyze_time_patterns(self, pair: str, timestamp: datetime) -> Dict[str, Any]:
        """Analyze temporal patterns in signal generation."""
        window_start = timestamp - timedelta(hours=self.window_hours)
        
        async with get_async_session() as session:
            # Get hourly distribution
            query = select(
                func.extract('hour', 'signals.created_at').label('hour'),
                func.count('signals.id').label('count')
            ).where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start
                )
            ).group_by('hour')
            
            result = await session.execute(query)
            hourly_dist = {int(row.hour): row.count for row in result}
            
            # Calculate patterns
            total_signals = sum(hourly_dist.values())
            peak_hour = max(hourly_dist, key=hourly_dist.get) if hourly_dist else 0
            
            # Identify active periods
            active_hours = [hour for hour, count in hourly_dist.items() 
                          if count > (total_signals / 24) * 1.5]
            
            return {
                'hourly_distribution': hourly_dist,
                'peak_hour': peak_hour,
                'active_hours': active_hours,
                'signal_frequency': total_signals / self.window_hours,
                'distribution_variance': np.var(list(hourly_dist.values())) if hourly_dist else 0
            }
    
    async def _detect_anomalies(self, pair: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Detect anomalies in historical data."""
        window_start = timestamp - timedelta(hours=self.window_hours)
        anomalies = []
        
        async with get_async_session() as session:
            # Get recent signals for anomaly detection
            query = select('signals').where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start
                )
            ).order_by('signals.created_at.desc()').limit(self.sample_size)
            
            result = await session.execute(query)
            signals = result.fetchall()
            
            if len(signals) < 5:
                return anomalies
            
            # Extract price data
            entry_prices = [float(s.entry_price) for s in signals]
            confidences = [s.confidence_score for s in signals]
            
            # Calculate statistics
            mean_price = np.mean(entry_prices)
            std_price = np.std(entry_prices)
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            
            # Detect anomalies (2 standard deviations)
            for signal in signals:
                anomaly_flags = []
                
                # Price anomaly
                price_zscore = abs((float(signal.entry_price) - mean_price) / (std_price + 1e-10))
                if price_zscore > 2:
                    anomaly_flags.append('price_deviation')
                
                # Confidence anomaly
                conf_zscore = abs((signal.confidence_score - mean_confidence) / (std_confidence + 1e-10))
                if conf_zscore > 2:
                    anomaly_flags.append('confidence_deviation')
                
                # Rapid succession (multiple signals within 1 minute)
                for other in signals:
                    if signal.id != other.id:
                        time_diff = abs((signal.created_at - other.created_at).total_seconds())
                        if time_diff < 60:
                            anomaly_flags.append('rapid_succession')
                            break
                
                if anomaly_flags:
                    anomalies.append({
                        'signal_id': signal.id,
                        'timestamp': signal.created_at.isoformat(),
                        'anomaly_types': list(set(anomaly_flags)),
                        'price_zscore': round(price_zscore, 2),
                        'confidence_zscore': round(conf_zscore, 2)
                    })
            
            return anomalies[:10]  # Limit to top 10 anomalies
    
    async def _calculate_performance_metrics(self, pair: str, timestamp: datetime) -> Dict[str, float]:
        """Calculate historical performance metrics."""
        window_start = timestamp - timedelta(hours=self.window_hours)
        
        async with get_async_session() as session:
            # Get completed signals with performance data
            query = select(
                'signal_performance.pnl_percentage',
                'signal_performance.hit_stop_loss',
                'signal_performance.hit_take_profit',
                'signal_performance.duration_hours'
            ).join(
                'signals',
                'signal_performance.signal_id' == 'signals.id'
            ).where(
                and_(
                    'signals.pair' == pair,
                    'signals.created_at' >= window_start,
                    'signal_performance.pnl_percentage' != None
                )
            )
            
            result = await session.execute(query)
            performances = result.fetchall()
            
            if not performances:
                return {
                    'win_rate': 0.5,
                    'avg_pnl': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profit_factor': 1.0,
                    'avg_duration': 0
                }
            
            # Calculate metrics
            pnls = [p.pnl_percentage for p in performances]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_pnl = np.mean(pnls) if pnls else 0
            
            # Sharpe ratio (simplified)
            if len(pnls) > 1:
                sharpe_ratio = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else total_wins
            
            # Average duration
            durations = [p.duration_hours for p in performances if p.duration_hours]
            avg_duration = np.mean(durations) if durations else 0
            
            return {
                'win_rate': round(win_rate, 4),
                'avg_pnl': round(avg_pnl, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_duration': round(avg_duration, 1)
            }
    
    def align_time_series(self, data: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
        """
        Align time series data to regular intervals.
        
        Args:
            data: List of (timestamp, value) tuples
            
        Returns:
            Aligned time series data
        """
        if not data:
            return []
        
        # Sort by timestamp
        data.sort(key=lambda x: x[0])
        
        # Determine interval (hourly alignment)
        start_time = data[0][0].replace(minute=0, second=0, microsecond=0)
        end_time = data[-1][0].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Create regular time grid
        time_grid = []
        current = start_time
        while current <= end_time:
            time_grid.append(current)
            current += timedelta(hours=1)
        
        # Align data to grid using forward fill
        aligned = []
        data_idx = 0
        last_value = 0
        
        for grid_time in time_grid:
            # Find the last data point before grid_time
            while data_idx < len(data) and data[data_idx][0] <= grid_time:
                last_value = data[data_idx][1]
                data_idx += 1
            
            aligned.append((grid_time, last_value))
        
        return aligned
    
    def sample_large_dataset(self, data: List[Any], target_size: int = None) -> List[Any]:
        """
        Sample large datasets for efficient processing.
        
        Args:
            data: Input dataset
            target_size: Target sample size (default: self.sample_size)
            
        Returns:
            Sampled dataset
        """
        if target_size is None:
            target_size = self.sample_size
        
        if len(data) <= target_size:
            return data
        
        # Use stratified sampling to maintain distribution
        step = len(data) / target_size
        sampled = []
        
        for i in range(target_size):
            idx = int(i * step)
            sampled.append(data[idx])
        
        return sampled
