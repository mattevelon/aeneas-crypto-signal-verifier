"""
Prometheus metrics collection and custom business metrics.
Provides comprehensive metrics for monitoring system health and performance.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, Info
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import structlog
import psutil
import aiohttp

from src.config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class MetricsCollector:
    """Collects and manages Prometheus metrics."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('system_memory_usage_bytes', 'Memory usage in bytes', registry=self.registry)
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
        # Application metrics
        self.active_connections = Gauge('app_active_connections', 'Number of active connections', registry=self.registry)
        self.request_duration = Histogram(
            'app_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.request_count = Counter(
            'app_request_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # Signal processing metrics
        self.signals_processed = Counter(
            'signals_processed_total',
            'Total signals processed',
            ['source', 'status'],
            registry=self.registry
        )
        self.signal_processing_time = Histogram(
            'signal_processing_duration_seconds',
            'Signal processing time',
            ['source', 'type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        self.signal_confidence = Histogram(
            'signal_confidence_score',
            'Signal confidence distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # AI/LLM metrics
        self.llm_requests = Counter(
            'llm_requests_total',
            'Total LLM API requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        self.llm_tokens_used = Counter(
            'llm_tokens_used_total',
            'Total tokens used',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        self.llm_response_time = Histogram(
            'llm_response_duration_seconds',
            'LLM response time',
            ['provider', 'model'],
            registry=self.registry
        )
        self.llm_cost = Counter(
            'llm_cost_dollars',
            'LLM API cost in dollars',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['database', 'operation'],
            registry=self.registry
        )
        self.db_errors = Counter(
            'database_errors_total',
            'Database errors',
            ['database', 'error_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_level'],
            registry=self.registry
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['cache_level'],
            registry=self.registry
        )
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['cache_level'],
            registry=self.registry
        )
        
        # Trading metrics
        self.signals_validated = Counter(
            'signals_validated_total',
            'Signals validated',
            ['result', 'reason'],
            registry=self.registry
        )
        self.position_size = Histogram(
            'position_size_dollars',
            'Position size distribution',
            buckets=[100, 500, 1000, 5000, 10000, 50000],
            registry=self.registry
        )
        self.risk_score = Histogram(
            'risk_score_distribution',
            'Risk score distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        self.profit_loss = Gauge(
            'profit_loss_dollars',
            'Current P&L',
            ['strategy', 'timeframe'],
            registry=self.registry
        )
        
        # Business metrics
        self.daily_signals = Gauge(
            'business_daily_signals',
            'Daily signal count',
            registry=self.registry
        )
        self.active_users = Gauge(
            'business_active_users',
            'Active users',
            ['timeframe'],
            registry=self.registry
        )
        self.revenue = Counter(
            'business_revenue_dollars',
            'Revenue in dollars',
            ['source'],
            registry=self.registry
        )
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, Any] = {}
        
        # Background tasks
        self._tasks = []
        
    async def start_collectors(self):
        """Start background metric collectors."""
        self._tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_application_metrics()),
        ]
        logger.info("Metrics collectors started")
        
    async def stop_collectors(self):
        """Stop background collectors."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Metrics collectors stopped")
        
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        while True:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.percent)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
                
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        while True:
            try:
                # Collect metrics from various services
                # This would integrate with your actual services
                
                # Example: Get active connections from WebSocket manager
                # connections = await get_active_connections()
                # self.active_connections.set(connections)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting application metrics: {e}")
                await asyncio.sleep(30)
                
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint, status=str(status)).observe(duration)
        
    def track_signal(
        self,
        source: str,
        status: str,
        processing_time: float,
        confidence: Optional[float] = None,
        signal_type: Optional[str] = None
    ):
        """Track signal processing metrics."""
        self.signals_processed.labels(source=source, status=status).inc()
        
        if signal_type:
            self.signal_processing_time.labels(source=source, type=signal_type).observe(processing_time)
        
        if confidence is not None:
            self.signal_confidence.observe(confidence)
            
    def track_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        response_time: float,
        tokens_used: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None
    ):
        """Track LLM API metrics."""
        self.llm_requests.labels(provider=provider, model=model, status=status).inc()
        self.llm_response_time.labels(provider=provider, model=model).observe(response_time)
        
        if tokens_used:
            for token_type, count in tokens_used.items():
                self.llm_tokens_used.labels(
                    provider=provider,
                    model=model,
                    type=token_type
                ).inc(count)
                
        if cost:
            self.llm_cost.labels(provider=provider, model=model).inc(cost)
            
    def track_database_operation(
        self,
        database: str,
        operation: str,
        duration: float,
        error: Optional[str] = None
    ):
        """Track database operation metrics."""
        self.db_query_duration.labels(database=database, operation=operation).observe(duration)
        
        if error:
            self.db_errors.labels(database=database, error_type=error).inc()
            
    def track_cache_operation(self, cache_level: str, hit: bool, size: Optional[int] = None):
        """Track cache operation metrics."""
        if hit:
            self.cache_hits.labels(cache_level=cache_level).inc()
        else:
            self.cache_misses.labels(cache_level=cache_level).inc()
            
        if size is not None:
            self.cache_size.labels(cache_level=cache_level).set(size)
            
    def track_trading_metrics(
        self,
        validation_result: Optional[str] = None,
        validation_reason: Optional[str] = None,
        position_size: Optional[float] = None,
        risk_score: Optional[float] = None,
        pnl: Optional[Dict[str, float]] = None
    ):
        """Track trading-related metrics."""
        if validation_result and validation_reason:
            self.signals_validated.labels(result=validation_result, reason=validation_reason).inc()
            
        if position_size is not None:
            self.position_size.observe(position_size)
            
        if risk_score is not None:
            self.risk_score.observe(risk_score)
            
        if pnl:
            for key, value in pnl.items():
                strategy, timeframe = key.split('_', 1) if '_' in key else (key, 'all')
                self.profit_loss.labels(strategy=strategy, timeframe=timeframe).set(value)
                
    def track_business_metrics(
        self,
        daily_signals: Optional[int] = None,
        active_users: Optional[Dict[str, int]] = None,
        revenue: Optional[Dict[str, float]] = None
    ):
        """Track business metrics."""
        if daily_signals is not None:
            self.daily_signals.set(daily_signals)
            
        if active_users:
            for timeframe, count in active_users.items():
                self.active_users.labels(timeframe=timeframe).set(count)
                
        if revenue:
            for source, amount in revenue.items():
                self.revenue.labels(source=source).inc(amount)
                
    def create_custom_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Optional[List[str]] = None
    ):
        """Create a custom metric."""
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.INFO:
            metric = Info(name, description, registry=self.registry)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
            
        self.custom_metrics[name] = metric
        return metric
        
    def get_metric(self, name: str) -> Any:
        """Get a metric by name."""
        return self.custom_metrics.get(name)
        
    def generate_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        return generate_latest(self.registry)
        
    async def push_to_gateway(self, gateway_url: str, job: str = "aeneas"):
        """Push metrics to Prometheus Pushgateway."""
        try:
            metrics_data = self.generate_metrics()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{gateway_url}/metrics/job/{job}",
                    data=metrics_data,
                    headers={'Content-Type': CONTENT_TYPE_LATEST}
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Pushed metrics to gateway: {gateway_url}")
                    else:
                        logger.error(f"Failed to push metrics: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error pushing metrics to gateway: {e}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "system": {
                "cpu_usage": self.cpu_usage._value.get(),
                "memory_usage": self.memory_usage._value.get(),
                "disk_usage": self.disk_usage._value.get()
            },
            "application": {
                "active_connections": self.active_connections._value.get(),
                "total_requests": sum(
                    sample.value for sample in self.request_count.collect()[0].samples
                ),
                "total_signals": sum(
                    sample.value for sample in self.signals_processed.collect()[0].samples
                )
            },
            "business": {
                "daily_signals": self.daily_signals._value.get(),
                "active_users": {
                    label[0]: value 
                    for label, value in self.active_users._metrics.items()
                }
            }
        }


# Global metrics collector
metrics_collector: Optional[MetricsCollector] = None


def init_metrics_collector(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Initialize global metrics collector."""
    global metrics_collector
    
    metrics_collector = MetricsCollector(registry=registry)
    return metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    if not metrics_collector:
        raise RuntimeError("Metrics collector not initialized")
    return metrics_collector
