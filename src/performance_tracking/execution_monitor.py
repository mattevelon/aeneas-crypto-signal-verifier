"""
Trade Execution Monitor

Monitors trade execution quality, tracks execution metrics,
and provides real-time insights into order execution performance.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum
import uuid
import logging
from dataclasses import dataclass, field
import json
from collections import deque, defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.kafka_client import KafkaClient
from src.core.market_data import MarketDataClient
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ExecutionStatus(str, Enum):
    """Trade execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"


class ExecutionQuality(str, Enum):
    """Execution quality rating"""
    EXCELLENT = "excellent"  # < 10ms, no slippage
    GOOD = "good"           # < 50ms, minimal slippage
    ACCEPTABLE = "acceptable"  # < 100ms, acceptable slippage
    POOR = "poor"           # < 500ms, high slippage
    FAILED = "failed"       # Execution failed or > 500ms


@dataclass
class ExecutionMetrics:
    """Detailed execution metrics"""
    signal_id: str
    order_id: str
    order_type: OrderType
    execution_status: ExecutionStatus
    submitted_at: datetime
    executed_at: Optional[datetime]
    latency_ms: Optional[int]
    fill_price: Optional[Decimal]
    fill_quantity: Optional[Decimal]
    partial_fills: List[Dict[str, Any]] = field(default_factory=list)
    slippage_bps: Optional[Decimal] = None
    execution_quality: ExecutionQuality = ExecutionQuality.POOR
    market_impact: Optional[Decimal] = None
    rejection_reason: Optional[str] = None


@dataclass
class ExecutionReport:
    """Comprehensive execution report"""
    total_orders: int
    successful_executions: int
    failed_executions: int
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    average_slippage_bps: Decimal
    execution_rate: float
    quality_distribution: Dict[ExecutionQuality, int]
    order_type_performance: Dict[OrderType, Dict[str, Any]]
    recommendations: List[str]


class ExecutionMonitor:
    """
    Real-time monitoring of trade execution quality and performance
    """
    
    def __init__(self, market_client: Optional[MarketDataClient] = None,
                 kafka_client: Optional[KafkaClient] = None):
        self.redis = get_redis()
        self.market_client = market_client or MarketDataClient()
        self.kafka_client = kafka_client or KafkaClient()
        
        # Execution metrics storage
        self.active_orders: Dict[str, ExecutionMetrics] = {}
        self.completed_orders = deque(maxlen=1000)
        self.execution_history = defaultdict(list)
        
        # Performance thresholds
        self.latency_thresholds = {
            ExecutionQuality.EXCELLENT: 10,
            ExecutionQuality.GOOD: 50,
            ExecutionQuality.ACCEPTABLE: 100,
            ExecutionQuality.POOR: 500
        }
        
        self.slippage_thresholds = {
            ExecutionQuality.EXCELLENT: 5,
            ExecutionQuality.GOOD: 10,
            ExecutionQuality.ACCEPTABLE: 25,
            ExecutionQuality.POOR: 50
        }
        
        self._monitoring_task = None
        self._monitoring_interval = 1
    
    async def start_monitoring(self):
        """Start execution monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Execution monitoring started")
    
    async def stop_monitoring(self):
        """Stop execution monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Execution monitoring stopped")
    
    async def submit_order(self, signal_id: str, order_type: OrderType,
                          quantity: Decimal, price: Optional[Decimal] = None) -> str:
        """Submit an order for execution monitoring"""
        try:
            order_id = str(uuid.uuid4())
            
            metrics = ExecutionMetrics(
                signal_id=signal_id,
                order_id=order_id,
                order_type=order_type,
                execution_status=ExecutionStatus.PENDING,
                submitted_at=datetime.utcnow(),
                executed_at=None,
                latency_ms=None,
                fill_price=None,
                fill_quantity=None
            )
            
            self.active_orders[order_id] = metrics
            
            # Simulate order submission
            await self._submit_to_exchange(order_id, signal_id, order_type, quantity, price)
            
            metrics.execution_status = ExecutionStatus.SUBMITTED
            
            await self._publish_execution_event(order_id, "order_submitted", metrics)
            
            logger.info(f"Order {order_id} submitted for signal {signal_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            raise
    
    async def update_execution(self, order_id: str, status: ExecutionStatus,
                             fill_price: Optional[Decimal] = None,
                             fill_quantity: Optional[Decimal] = None,
                             rejection_reason: Optional[str] = None):
        """Update execution status for an order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return
            
            metrics = self.active_orders[order_id]
            metrics.execution_status = status
            
            if status == ExecutionStatus.FILLED:
                metrics.executed_at = datetime.utcnow()
                metrics.fill_price = fill_price
                metrics.fill_quantity = fill_quantity
                
                # Calculate latency
                latency = (metrics.executed_at - metrics.submitted_at).total_seconds() * 1000
                metrics.latency_ms = int(latency)
                
                # Calculate slippage
                if fill_price:
                    metrics.slippage_bps = await self._calculate_slippage_bps(
                        metrics.signal_id, fill_price
                    )
                
                # Determine execution quality
                metrics.execution_quality = self._determine_execution_quality(
                    metrics.latency_ms, metrics.slippage_bps
                )
                
                # Calculate market impact
                metrics.market_impact = await self._calculate_market_impact(
                    metrics.signal_id, fill_price, fill_quantity
                )
                
                # Move to completed orders
                self.completed_orders.append(metrics)
                self.execution_history[metrics.signal_id].append(metrics)
                del self.active_orders[order_id]
                
                await self._store_execution_metrics(metrics)
                
            elif status == ExecutionStatus.PARTIALLY_FILLED:
                if fill_price and fill_quantity:
                    metrics.partial_fills.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'price': str(fill_price),
                        'quantity': str(fill_quantity)
                    })
                
            elif status == ExecutionStatus.REJECTED:
                metrics.rejection_reason = rejection_reason
                metrics.execution_quality = ExecutionQuality.FAILED
                
                self.completed_orders.append(metrics)
                self.execution_history[metrics.signal_id].append(metrics)
                del self.active_orders[order_id]
            
            await self._publish_execution_event(order_id, "execution_updated", metrics)
            
            logger.info(f"Order {order_id} updated to status {status}")
            
        except Exception as e:
            logger.error(f"Error updating execution: {str(e)}")
    
    async def get_execution_report(self, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> ExecutionReport:
        """Generate comprehensive execution report"""
        try:
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(days=1)
            
            orders = await self._get_orders_in_range(start_time, end_time)
            
            if not orders:
                return self._create_empty_report()
            
            total_orders = len(orders)
            successful = [o for o in orders if o.execution_status == ExecutionStatus.FILLED]
            failed = [o for o in orders if o.execution_status in 
                     [ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED]]
            
            # Latency metrics
            latencies = [o.latency_ms for o in successful if o.latency_ms is not None]
            avg_latency = np.mean(latencies) if latencies else 0
            median_latency = np.median(latencies) if latencies else 0
            p95_latency = np.percentile(latencies, 95) if latencies else 0
            
            # Slippage metrics
            slippages = [float(o.slippage_bps) for o in successful 
                        if o.slippage_bps is not None]
            avg_slippage = Decimal(str(np.mean(slippages))) if slippages else Decimal("0")
            
            # Quality distribution
            quality_dist = defaultdict(int)
            for order in orders:
                quality_dist[order.execution_quality] += 1
            
            # Order type performance
            order_type_perf = await self._analyze_order_type_performance(orders)
            
            # Generate recommendations
            recommendations = await self._generate_execution_recommendations(
                orders, avg_latency, avg_slippage, quality_dist
            )
            
            return ExecutionReport(
                total_orders=total_orders,
                successful_executions=len(successful),
                failed_executions=len(failed),
                average_latency_ms=avg_latency,
                median_latency_ms=median_latency,
                p95_latency_ms=p95_latency,
                average_slippage_bps=avg_slippage,
                execution_rate=len(successful) / total_orders if total_orders else 0,
                quality_distribution=dict(quality_dist),
                order_type_performance=order_type_perf,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating execution report: {str(e)}")
            return self._create_empty_report()
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time execution metrics"""
        try:
            active_count = len(self.active_orders)
            pending = sum(1 for o in self.active_orders.values() 
                         if o.execution_status == ExecutionStatus.PENDING)
            submitted = sum(1 for o in self.active_orders.values() 
                          if o.execution_status == ExecutionStatus.SUBMITTED)
            
            recent_orders = list(self.completed_orders)[-100:]
            if recent_orders:
                recent_latencies = [o.latency_ms for o in recent_orders 
                                  if o.latency_ms is not None]
                recent_avg_latency = np.mean(recent_latencies) if recent_latencies else 0
                
                recent_success_rate = sum(1 for o in recent_orders 
                                         if o.execution_status == ExecutionStatus.FILLED) / len(recent_orders)
            else:
                recent_avg_latency = 0
                recent_success_rate = 0
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'active_orders': active_count,
                'pending_orders': pending,
                'submitted_orders': submitted,
                'recent_avg_latency_ms': recent_avg_latency,
                'recent_success_rate': recent_success_rate,
                'health_status': self._determine_health_status(
                    recent_avg_latency, recent_success_rate
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {str(e)}")
            return {}
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while True:
                await self._check_stale_orders()
                await self._publish_monitoring_heartbeat()
                await asyncio.sleep(self._monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _check_stale_orders(self):
        """Check for and handle stale orders"""
        stale_threshold = timedelta(minutes=5)
        current_time = datetime.utcnow()
        
        stale_orders = []
        for order_id, metrics in self.active_orders.items():
            if current_time - metrics.submitted_at > stale_threshold:
                stale_orders.append(order_id)
        
        for order_id in stale_orders:
            logger.warning(f"Order {order_id} is stale, marking as expired")
            await self.update_execution(
                order_id, 
                ExecutionStatus.EXPIRED,
                rejection_reason="Order expired due to timeout"
            )
    
    async def _analyze_order_type_performance(self, 
                                             orders: List[ExecutionMetrics]) -> Dict[OrderType, Dict[str, Any]]:
        """Analyze performance by order type"""
        performance = {}
        
        for order_type in OrderType:
            type_orders = [o for o in orders if o.order_type == order_type]
            if type_orders:
                successful = [o for o in type_orders 
                            if o.execution_status == ExecutionStatus.FILLED]
                
                latencies = [o.latency_ms for o in successful if o.latency_ms]
                slippages = [float(o.slippage_bps) for o in successful if o.slippage_bps]
                
                performance[order_type] = {
                    'count': len(type_orders),
                    'success_rate': len(successful) / len(type_orders),
                    'avg_latency_ms': np.mean(latencies) if latencies else 0,
                    'avg_slippage_bps': np.mean(slippages) if slippages else 0
                }
        
        return performance
    
    async def _generate_execution_recommendations(self, 
                                                 orders: List[ExecutionMetrics],
                                                 avg_latency: float,
                                                 avg_slippage: Decimal,
                                                 quality_dist: Dict[ExecutionQuality, int]) -> List[str]:
        """Generate recommendations to improve execution"""
        recommendations = []
        
        if avg_latency > 100:
            recommendations.append("Consider co-location or VPS closer to exchange servers")
        elif avg_latency > 50:
            recommendations.append("Optimize network connectivity for lower latency")
        
        if avg_slippage > Decimal("25"):
            recommendations.append("Increase use of limit orders to reduce slippage")
            recommendations.append("Consider splitting large orders into smaller chunks")
        
        poor_quality = quality_dist.get(ExecutionQuality.POOR, 0) + \
                      quality_dist.get(ExecutionQuality.FAILED, 0)
        total = sum(quality_dist.values())
        
        if total > 0 and poor_quality / total > 0.2:
            recommendations.append("Review order routing logic for optimization")
            recommendations.append("Consider using smart order routing algorithms")
        
        market_orders = [o for o in orders if o.order_type == OrderType.MARKET]
        if market_orders and len(market_orders) / len(orders) > 0.7:
            recommendations.append("Reduce reliance on market orders during volatile periods")
        
        if not recommendations:
            recommendations.append("Execution quality is good, maintain current strategies")
        
        return recommendations
    
    # Helper Methods
    async def _submit_to_exchange(self, order_id: str, signal_id: str,
                                 order_type: OrderType, quantity: Decimal,
                                 price: Optional[Decimal] = None):
        """Submit order to exchange (mock implementation)"""
        logger.info(f"Submitting order {order_id} to exchange")
    
    async def _calculate_slippage_bps(self, signal_id: str, 
                                     fill_price: Decimal) -> Decimal:
        """Calculate slippage in basis points"""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(Signal).where(Signal.id == uuid.UUID(signal_id))
                )
                signal = result.scalar_one_or_none()
                
                if signal and signal.entry_price:
                    expected_price = Decimal(str(signal.entry_price))
                    slippage = abs(fill_price - expected_price) / expected_price
                    return (slippage * Decimal("10000")).quantize(Decimal("0.01"))
            
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {str(e)}")
            return Decimal("0")
    
    async def _calculate_market_impact(self, signal_id: str,
                                      fill_price: Decimal,
                                      fill_quantity: Decimal) -> Decimal:
        """Calculate market impact of the trade"""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(Signal).where(Signal.id == uuid.UUID(signal_id))
                )
                signal = result.scalar_one_or_none()
                
                if signal:
                    volume_data = await self.market_client.get_24h_volume(signal.pair)
                    if volume_data:
                        daily_volume = Decimal(str(volume_data.get('volume', 1000000)))
                        trade_value = fill_price * fill_quantity
                        impact = (trade_value / daily_volume) * Decimal("10000")
                        return impact.quantize(Decimal("0.01"))
            
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            return Decimal("0")
    
    def _determine_execution_quality(self, latency_ms: int, 
                                    slippage_bps: Optional[Decimal]) -> ExecutionQuality:
        """Determine overall execution quality"""
        latency_quality = ExecutionQuality.POOR
        for quality, threshold in self.latency_thresholds.items():
            if latency_ms <= threshold:
                latency_quality = quality
                break
        
        slippage_quality = ExecutionQuality.POOR
        if slippage_bps is not None:
            for quality, threshold in self.slippage_thresholds.items():
                if slippage_bps <= Decimal(str(threshold)):
                    slippage_quality = quality
                    break
        else:
            slippage_quality = latency_quality
        
        quality_order = [ExecutionQuality.EXCELLENT, ExecutionQuality.GOOD,
                        ExecutionQuality.ACCEPTABLE, ExecutionQuality.POOR,
                        ExecutionQuality.FAILED]
        
        latency_idx = quality_order.index(latency_quality)
        slippage_idx = quality_order.index(slippage_quality)
        
        return quality_order[max(latency_idx, slippage_idx)]
    
    def _determine_health_status(self, avg_latency: float, 
                                success_rate: float) -> str:
        """Determine system health status"""
        if avg_latency < 50 and success_rate > 0.95:
            return "excellent"
        elif avg_latency < 100 and success_rate > 0.90:
            return "good"
        elif avg_latency < 200 and success_rate > 0.80:
            return "degraded"
        else:
            return "poor"
    
    async def _store_execution_metrics(self, metrics: ExecutionMetrics):
        """Store execution metrics in database and cache"""
        cache_data = self._serialize_metrics(metrics)
        await self.redis.setex(
            f"execution_metrics:{metrics.order_id}",
            86400,
            json.dumps(cache_data, default=str)
        )
        logger.info(f"Stored execution metrics for order {metrics.order_id}")
    
    async def _publish_execution_event(self, order_id: str, event_type: str,
                                      metrics: ExecutionMetrics):
        """Publish execution event to Kafka"""
        try:
            event_data = {
                'event_type': event_type,
                'order_id': order_id,
                'signal_id': metrics.signal_id,
                'status': metrics.execution_status.value,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': self._serialize_metrics(metrics)
            }
            
            await self.kafka_client.send_message(
                'execution-events',
                json.dumps(event_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error publishing execution event: {str(e)}")
    
    async def _publish_monitoring_heartbeat(self):
        """Publish monitoring heartbeat"""
        try:
            heartbeat = {
                'timestamp': datetime.utcnow().isoformat(),
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders),
                'status': 'healthy'
            }
            
            await self.redis.setex(
                "execution_monitor:heartbeat",
                10,
                json.dumps(heartbeat)
            )
            
        except Exception as e:
            logger.error(f"Error publishing heartbeat: {str(e)}")
    
    async def _get_orders_in_range(self, start_time: datetime,
                                  end_time: datetime) -> List[ExecutionMetrics]:
        """Get orders within time range"""
        orders = []
        
        for metrics in self.completed_orders:
            if start_time <= metrics.submitted_at <= end_time:
                orders.append(metrics)
        
        for metrics in self.active_orders.values():
            if start_time <= metrics.submitted_at <= end_time:
                orders.append(metrics)
        
        return orders
    
    def _create_empty_report(self) -> ExecutionReport:
        """Create empty execution report"""
        return ExecutionReport(
            total_orders=0,
            successful_executions=0,
            failed_executions=0,
            average_latency_ms=0,
            median_latency_ms=0,
            p95_latency_ms=0,
            average_slippage_bps=Decimal("0"),
            execution_rate=0,
            quality_distribution={},
            order_type_performance={},
            recommendations=[]
        )
    
    def _serialize_metrics(self, metrics: ExecutionMetrics) -> Dict[str, Any]:
        """Serialize metrics for storage"""
        return {
            'signal_id': metrics.signal_id,
            'order_id': metrics.order_id,
            'order_type': metrics.order_type.value,
            'execution_status': metrics.execution_status.value,
            'submitted_at': metrics.submitted_at.isoformat(),
            'executed_at': metrics.executed_at.isoformat() if metrics.executed_at else None,
            'latency_ms': metrics.latency_ms,
            'fill_price': str(metrics.fill_price) if metrics.fill_price else None,
            'fill_quantity': str(metrics.fill_quantity) if metrics.fill_quantity else None,
            'slippage_bps': str(metrics.slippage_bps) if metrics.slippage_bps else None,
            'execution_quality': metrics.execution_quality.value,
            'market_impact': str(metrics.market_impact) if metrics.market_impact else None
        }
