"""
Slippage Analyzer

Analyzes slippage in trade execution, comparing expected vs actual prices,
identifying patterns, and providing recommendations to minimize slippage.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
import logging
from dataclasses import dataclass, field
import json
import statistics

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import pandas as pd

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.market_data import MarketDataClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SlippageType(str, Enum):
    """Types of slippage"""
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MARKET_ORDER = "market_order"
    LIMIT_ORDER = "limit_order"


class SlippageSeverity(str, Enum):
    """Severity levels of slippage"""
    NEGLIGIBLE = "negligible"  # < 0.1%
    LOW = "low"               # 0.1% - 0.5%
    MEDIUM = "medium"         # 0.5% - 1%
    HIGH = "high"             # 1% - 2%
    SEVERE = "severe"         # > 2%


@dataclass
class SlippageMetrics:
    """Detailed slippage metrics"""
    signal_id: str
    slippage_type: SlippageType
    expected_price: Decimal
    actual_price: Decimal
    slippage_amount: Decimal
    slippage_percentage: Decimal
    severity: SlippageSeverity
    cost_impact: Decimal
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SlippageAnalysis:
    """Comprehensive slippage analysis"""
    total_slippage: Decimal
    average_slippage_pct: Decimal
    median_slippage_pct: Decimal
    max_slippage: Decimal
    min_slippage: Decimal
    entry_slippage: Decimal
    exit_slippage: Decimal
    slippage_by_hour: Dict[int, Decimal]
    slippage_by_pair: Dict[str, Decimal]
    slippage_by_volume: Dict[str, Decimal]
    patterns_identified: List[str]
    recommendations: List[str]
    estimated_savings: Decimal


class SlippageAnalyzer:
    """
    Analyzes and provides insights on trading slippage
    """
    
    def __init__(self, market_client: Optional[MarketDataClient] = None):
        self.redis = get_redis()
        self.market_client = market_client or MarketDataClient()
        
        # Slippage thresholds
        self.severity_thresholds = {
            SlippageSeverity.NEGLIGIBLE: Decimal("0.001"),  # 0.1%
            SlippageSeverity.LOW: Decimal("0.005"),         # 0.5%
            SlippageSeverity.MEDIUM: Decimal("0.01"),       # 1%
            SlippageSeverity.HIGH: Decimal("0.02"),         # 2%
            SlippageSeverity.SEVERE: Decimal("999")         # > 2%
        }
        
        # Acceptable slippage by market condition
        self.acceptable_slippage = {
            'high_volatility': Decimal("0.015"),    # 1.5%
            'normal': Decimal("0.005"),             # 0.5%
            'low_liquidity': Decimal("0.02"),       # 2%
            'news_event': Decimal("0.025")          # 2.5%
        }
    
    async def analyze_signal_slippage(self, signal_id: str) -> Optional[SlippageMetrics]:
        """
        Analyze slippage for a single signal
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            SlippageMetrics or None if analysis fails
        """
        try:
            async with get_async_session() as session:
                # Get signal and performance data
                signal = await self._get_signal(session, signal_id)
                performance = await self._get_performance(session, signal_id)
                
                if not signal or not performance:
                    logger.error(f"Signal or performance data not found for {signal_id}")
                    return None
                
                # Analyze entry slippage
                entry_metrics = await self._analyze_entry_slippage(signal, performance)
                
                # Analyze exit slippage if applicable
                exit_metrics = None
                if performance.actual_exit:
                    exit_metrics = await self._analyze_exit_slippage(signal, performance)
                
                # Get market conditions at time of trade
                market_conditions = await self._get_market_conditions(
                    signal.pair,
                    performance.entry_time or datetime.utcnow()
                )
                
                # Combine metrics
                if entry_metrics:
                    entry_metrics.market_conditions = market_conditions
                    
                    # Cache the metrics
                    await self._cache_slippage_metrics(signal_id, entry_metrics)
                    
                    return entry_metrics
                
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing slippage for signal {signal_id}: {str(e)}")
            return None
    
    async def analyze_portfolio_slippage(self, signal_ids: List[str],
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None) -> SlippageAnalysis:
        """
        Analyze slippage across multiple signals
        
        Args:
            signal_ids: List of signal identifiers
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Comprehensive SlippageAnalysis
        """
        try:
            all_metrics = []
            slippage_by_hour = {}
            slippage_by_pair = {}
            slippage_by_volume = {}
            
            # Analyze each signal
            for signal_id in signal_ids:
                metrics = await self.analyze_signal_slippage(signal_id)
                if metrics:
                    all_metrics.append(metrics)
                    
                    # Aggregate by hour
                    hour = metrics.timestamp.hour
                    if hour not in slippage_by_hour:
                        slippage_by_hour[hour] = []
                    slippage_by_hour[hour].append(metrics.slippage_percentage)
                    
                    # Get signal details for pair aggregation
                    async with get_async_session() as session:
                        signal = await self._get_signal(session, signal_id)
                        if signal:
                            if signal.pair not in slippage_by_pair:
                                slippage_by_pair[signal.pair] = []
                            slippage_by_pair[signal.pair].append(metrics.slippage_percentage)
                            
                            # Aggregate by volume bucket
                            volume = metrics.market_conditions.get('volume', 0)
                            volume_bucket = self._get_volume_bucket(volume)
                            if volume_bucket not in slippage_by_volume:
                                slippage_by_volume[volume_bucket] = []
                            slippage_by_volume[volume_bucket].append(metrics.slippage_percentage)
            
            if not all_metrics:
                return self._create_empty_analysis()
            
            # Calculate aggregate metrics
            slippage_values = [m.slippage_percentage for m in all_metrics]
            entry_slippage = [m.slippage_percentage for m in all_metrics 
                            if m.slippage_type == SlippageType.ENTRY]
            exit_slippage = [m.slippage_percentage for m in all_metrics 
                           if m.slippage_type == SlippageType.EXIT]
            
            # Average by hour
            avg_by_hour = {}
            for hour, values in slippage_by_hour.items():
                avg_by_hour[hour] = Decimal(str(statistics.mean(map(float, values))))
            
            # Average by pair
            avg_by_pair = {}
            for pair, values in slippage_by_pair.items():
                avg_by_pair[pair] = Decimal(str(statistics.mean(map(float, values))))
            
            # Average by volume
            avg_by_volume = {}
            for bucket, values in slippage_by_volume.items():
                avg_by_volume[bucket] = Decimal(str(statistics.mean(map(float, values))))
            
            # Identify patterns
            patterns = await self._identify_slippage_patterns(
                all_metrics, avg_by_hour, avg_by_pair, avg_by_volume
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                patterns, avg_by_hour, avg_by_pair, avg_by_volume
            )
            
            # Estimate potential savings
            estimated_savings = await self._estimate_savings(all_metrics, recommendations)
            
            # Create analysis
            analysis = SlippageAnalysis(
                total_slippage=Decimal(str(sum(map(float, slippage_values)))),
                average_slippage_pct=Decimal(str(statistics.mean(map(float, slippage_values)))),
                median_slippage_pct=Decimal(str(statistics.median(map(float, slippage_values)))),
                max_slippage=max(slippage_values),
                min_slippage=min(slippage_values),
                entry_slippage=Decimal(str(statistics.mean(map(float, entry_slippage)))) if entry_slippage else Decimal("0"),
                exit_slippage=Decimal(str(statistics.mean(map(float, exit_slippage)))) if exit_slippage else Decimal("0"),
                slippage_by_hour=avg_by_hour,
                slippage_by_pair=avg_by_pair,
                slippage_by_volume=avg_by_volume,
                patterns_identified=patterns,
                recommendations=recommendations,
                estimated_savings=estimated_savings
            )
            
            # Cache the analysis
            await self._cache_portfolio_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio slippage: {str(e)}")
            return self._create_empty_analysis()
    
    # Slippage Analysis Methods
    async def _analyze_entry_slippage(self, signal: Signal, 
                                      performance: SignalPerformance) -> Optional[SlippageMetrics]:
        """Analyze entry slippage"""
        if not performance.actual_entry or not signal.entry_price:
            return None
        
        expected_price = Decimal(str(signal.entry_price))
        actual_price = Decimal(str(performance.actual_entry))
        
        # Calculate slippage
        if signal.direction == 'long':
            # For long, paying more is negative slippage
            slippage_amount = actual_price - expected_price
        else:  # short
            # For short, paying less is negative slippage
            slippage_amount = expected_price - actual_price
        
        slippage_percentage = (abs(slippage_amount) / expected_price) * Decimal("100")
        
        # Determine severity
        severity = self._determine_severity(slippage_percentage)
        
        # Calculate cost impact (assuming $1000 position)
        position_size = Decimal("1000")
        cost_impact = (slippage_amount / expected_price) * position_size
        
        return SlippageMetrics(
            signal_id=str(signal.id),
            slippage_type=SlippageType.ENTRY,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            severity=severity,
            cost_impact=abs(cost_impact)
        )
    
    async def _analyze_exit_slippage(self, signal: Signal, 
                                     performance: SignalPerformance) -> Optional[SlippageMetrics]:
        """Analyze exit slippage"""
        if not performance.actual_exit:
            return None
        
        # Determine expected exit price
        expected_price = None
        slippage_type = SlippageType.EXIT
        
        if performance.hit_stop_loss and signal.stop_loss:
            expected_price = Decimal(str(signal.stop_loss))
            slippage_type = SlippageType.STOP_LOSS
        elif performance.hit_take_profit and signal.take_profits:
            tp_index = (performance.hit_take_profit - 1) if performance.hit_take_profit > 0 else 0
            if tp_index < len(signal.take_profits):
                expected_price = Decimal(str(signal.take_profits[tp_index].get('price', 0)))
                slippage_type = SlippageType.TAKE_PROFIT
        
        if not expected_price:
            return None
        
        actual_price = Decimal(str(performance.actual_exit))
        
        # Calculate slippage
        if signal.direction == 'long':
            # For long exit, getting less is negative slippage
            slippage_amount = expected_price - actual_price
        else:  # short
            # For short exit, getting more is negative slippage
            slippage_amount = actual_price - expected_price
        
        slippage_percentage = (abs(slippage_amount) / expected_price) * Decimal("100")
        
        # Determine severity
        severity = self._determine_severity(slippage_percentage)
        
        # Calculate cost impact
        position_size = Decimal("1000")
        cost_impact = (slippage_amount / expected_price) * position_size
        
        return SlippageMetrics(
            signal_id=str(signal.id),
            slippage_type=slippage_type,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            severity=severity,
            cost_impact=abs(cost_impact)
        )
    
    async def _get_market_conditions(self, pair: str, timestamp: datetime) -> Dict[str, Any]:
        """Get market conditions at time of trade"""
        try:
            # Get market data
            volume_24h = await self.market_client.get_24h_volume(pair)
            volatility = await self.market_client.get_volatility(pair)
            spread = await self.market_client.get_spread(pair)
            
            # Determine market condition
            conditions = {
                'volume': volume_24h.get('volume', 0) if volume_24h else 0,
                'volatility': volatility if volatility else 0,
                'spread': spread if spread else 0,
                'timestamp': timestamp.isoformat(),
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_weekend': timestamp.weekday() >= 5
            }
            
            # Classify market condition
            if volatility and volatility > 50:  # High volatility threshold
                conditions['market_state'] = 'high_volatility'
            elif volume_24h and volume_24h.get('volume', 0) < 100000:  # Low liquidity
                conditions['market_state'] = 'low_liquidity'
            else:
                conditions['market_state'] = 'normal'
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return {}
    
    async def _identify_slippage_patterns(self, metrics: List[SlippageMetrics],
                                          by_hour: Dict[int, Decimal],
                                          by_pair: Dict[str, Decimal],
                                          by_volume: Dict[str, Decimal]) -> List[str]:
        """Identify patterns in slippage data"""
        patterns = []
        
        # Time-based patterns
        if by_hour:
            worst_hours = sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
            if worst_hours[0][1] > Decimal("1"):  # > 1% slippage
                patterns.append(f"High slippage during hour {worst_hours[0][0]}:00")
        
        # Pair-based patterns
        if by_pair:
            worst_pairs = sorted(by_pair.items(), key=lambda x: x[1], reverse=True)[:3]
            if worst_pairs[0][1] > Decimal("1.5"):  # > 1.5% slippage
                patterns.append(f"Consistently high slippage on {worst_pairs[0][0]}")
        
        # Volume-based patterns
        if by_volume:
            if 'low' in by_volume and by_volume['low'] > Decimal("2"):
                patterns.append("Higher slippage during low volume periods")
        
        # Entry vs Exit patterns
        entry_metrics = [m for m in metrics if m.slippage_type == SlippageType.ENTRY]
        exit_metrics = [m for m in metrics if m.slippage_type in 
                       [SlippageType.EXIT, SlippageType.STOP_LOSS, SlippageType.TAKE_PROFIT]]
        
        if entry_metrics and exit_metrics:
            avg_entry = statistics.mean([float(m.slippage_percentage) for m in entry_metrics])
            avg_exit = statistics.mean([float(m.slippage_percentage) for m in exit_metrics])
            
            if avg_exit > avg_entry * 1.5:
                patterns.append("Exit slippage significantly higher than entry")
        
        # Stop loss slippage pattern
        sl_metrics = [m for m in metrics if m.slippage_type == SlippageType.STOP_LOSS]
        if sl_metrics:
            avg_sl_slippage = statistics.mean([float(m.slippage_percentage) for m in sl_metrics])
            if avg_sl_slippage > 2:  # > 2%
                patterns.append("High slippage on stop loss orders")
        
        # Severity patterns
        severe_count = sum(1 for m in metrics if m.severity == SlippageSeverity.SEVERE)
        if severe_count > len(metrics) * 0.1:  # > 10% severe
            patterns.append("Frequent severe slippage events")
        
        return patterns
    
    async def _generate_recommendations(self, patterns: List[str],
                                       by_hour: Dict[int, Decimal],
                                       by_pair: Dict[str, Decimal],
                                       by_volume: Dict[str, Decimal]) -> List[str]:
        """Generate recommendations to reduce slippage"""
        recommendations = []
        
        # Time-based recommendations
        if by_hour:
            best_hours = sorted(by_hour.items(), key=lambda x: x[1])[:3]
            if best_hours:
                recommendations.append(
                    f"Consider trading during {best_hours[0][0]}:00-{best_hours[0][0]+1}:00 for lower slippage"
                )
        
        # Pair-based recommendations
        if by_pair:
            high_slippage_pairs = [p for p, s in by_pair.items() if s > Decimal("1.5")]
            if high_slippage_pairs:
                recommendations.append(
                    f"Use limit orders for {', '.join(high_slippage_pairs)} to reduce slippage"
                )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if "High slippage during hour" in pattern:
                recommendations.append("Avoid market orders during high slippage hours")
            elif "low volume" in pattern.lower():
                recommendations.append("Increase limit order usage during low volume periods")
            elif "stop loss" in pattern.lower():
                recommendations.append("Consider using stop-limit orders instead of stop-market")
            elif "Exit slippage" in pattern:
                recommendations.append("Implement staged exits with multiple take-profit levels")
        
        # Volume-based recommendations
        if 'low' in by_volume and by_volume['low'] > Decimal("2"):
            recommendations.append("Split large orders into smaller chunks during low liquidity")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Monitor and adjust order types based on market conditions")
            recommendations.append("Consider using iceberg orders for large positions")
        
        return recommendations
    
    async def _estimate_savings(self, metrics: List[SlippageMetrics], 
                               recommendations: List[str]) -> Decimal:
        """Estimate potential savings from implementing recommendations"""
        if not metrics:
            return Decimal("0")
        
        # Current average slippage cost
        avg_cost = statistics.mean([float(m.cost_impact) for m in metrics])
        
        # Estimated improvement based on recommendations
        improvement_factor = Decimal("0")
        
        for rec in recommendations:
            if "limit order" in rec.lower():
                improvement_factor += Decimal("0.3")  # 30% improvement
            elif "stop-limit" in rec.lower():
                improvement_factor += Decimal("0.2")  # 20% improvement
            elif "staged exits" in rec.lower():
                improvement_factor += Decimal("0.15")  # 15% improvement
            elif "smaller chunks" in rec.lower():
                improvement_factor += Decimal("0.1")  # 10% improvement
        
        # Cap improvement at 50%
        improvement_factor = min(improvement_factor, Decimal("0.5"))
        
        # Calculate monthly savings (assuming 100 trades per month)
        monthly_savings = Decimal(str(avg_cost)) * improvement_factor * Decimal("100")
        
        return monthly_savings.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    # Helper Methods
    def _determine_severity(self, slippage_pct: Decimal) -> SlippageSeverity:
        """Determine slippage severity"""
        slippage_pct = abs(slippage_pct)
        
        if slippage_pct < self.severity_thresholds[SlippageSeverity.NEGLIGIBLE] * 100:
            return SlippageSeverity.NEGLIGIBLE
        elif slippage_pct < self.severity_thresholds[SlippageSeverity.LOW] * 100:
            return SlippageSeverity.LOW
        elif slippage_pct < self.severity_thresholds[SlippageSeverity.MEDIUM] * 100:
            return SlippageSeverity.MEDIUM
        elif slippage_pct < self.severity_thresholds[SlippageSeverity.HIGH] * 100:
            return SlippageSeverity.HIGH
        else:
            return SlippageSeverity.SEVERE
    
    def _get_volume_bucket(self, volume: float) -> str:
        """Categorize volume into buckets"""
        if volume < 100000:
            return 'low'
        elif volume < 1000000:
            return 'medium'
        elif volume < 10000000:
            return 'high'
        else:
            return 'very_high'
    
    def _create_empty_analysis(self) -> SlippageAnalysis:
        """Create empty analysis object"""
        return SlippageAnalysis(
            total_slippage=Decimal("0"),
            average_slippage_pct=Decimal("0"),
            median_slippage_pct=Decimal("0"),
            max_slippage=Decimal("0"),
            min_slippage=Decimal("0"),
            entry_slippage=Decimal("0"),
            exit_slippage=Decimal("0"),
            slippage_by_hour={},
            slippage_by_pair={},
            slippage_by_volume={},
            patterns_identified=[],
            recommendations=[],
            estimated_savings=Decimal("0")
        )
    
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
    
    async def _get_signals_in_range(self, session: AsyncSession, 
                                    start_date: datetime, end_date: datetime) -> List[str]:
        """Get signals within date range"""
        result = await session.execute(
            select(Signal.id)
            .where(and_(
                Signal.created_at >= start_date,
                Signal.created_at <= end_date
            ))
        )
        return [str(row[0]) for row in result]
    
    async def _cache_slippage_metrics(self, signal_id: str, metrics: SlippageMetrics):
        """Cache slippage metrics"""
        cache_data = {
            'signal_id': metrics.signal_id,
            'slippage_type': metrics.slippage_type.value,
            'expected_price': str(metrics.expected_price),
            'actual_price': str(metrics.actual_price),
            'slippage_amount': str(metrics.slippage_amount),
            'slippage_percentage': str(metrics.slippage_percentage),
            'severity': metrics.severity.value,
            'cost_impact': str(metrics.cost_impact),
            'market_conditions': metrics.market_conditions,
            'timestamp': metrics.timestamp.isoformat()
        }
        
        await self.redis.setex(
            f"slippage_metrics:{signal_id}",
            3600,  # 1 hour cache
            json.dumps(cache_data, default=str)
        )
    
    async def _cache_portfolio_analysis(self, analysis: SlippageAnalysis):
        """Cache portfolio slippage analysis"""
        cache_data = {
            'total_slippage': str(analysis.total_slippage),
            'average_slippage_pct': str(analysis.average_slippage_pct),
            'median_slippage_pct': str(analysis.median_slippage_pct),
            'max_slippage': str(analysis.max_slippage),
            'min_slippage': str(analysis.min_slippage),
            'entry_slippage': str(analysis.entry_slippage),
            'exit_slippage': str(analysis.exit_slippage),
            'slippage_by_hour': {str(k): str(v) for k, v in analysis.slippage_by_hour.items()},
            'slippage_by_pair': {k: str(v) for k, v in analysis.slippage_by_pair.items()},
            'slippage_by_volume': {k: str(v) for k, v in analysis.slippage_by_volume.items()},
            'patterns_identified': analysis.patterns_identified,
            'recommendations': analysis.recommendations,
            'estimated_savings': str(analysis.estimated_savings),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            "portfolio_slippage_analysis",
            3600,  # 1 hour cache
            json.dumps(cache_data, default=str)
        )
