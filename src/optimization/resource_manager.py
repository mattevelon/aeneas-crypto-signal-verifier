"""
Resource Management System for dynamic allocation and auto-scaling.

Monitors resource usage, provides auto-scaling recommendations,
and optimizes resource allocation based on demand patterns.
"""

import asyncio
import os
import psutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
from pydantic import BaseModel, Field

from src.core.redis_client import get_redis
from src.core.kafka_client import KafkaClient
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of resources to manage."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    WORKERS = "workers"


class ScalingAction(str, Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"     # Add more instances
    SCALE_IN = "scale_in"        # Remove instances
    NO_ACTION = "no_action"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ResourceMetrics:
    """Current resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_in_mb: float
    network_out_mb: float
    active_connections: int
    worker_count: int
    queue_size: int
    cache_hit_rate: float
    database_connections: int
    response_time_ms: float


class ResourceThresholds(BaseModel):
    """Thresholds for resource alerts and scaling."""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    response_time_warning_ms: float = 1000.0
    response_time_critical_ms: float = 2000.0
    queue_size_warning: int = 100
    queue_size_critical: int = 500
    
    # Auto-scaling thresholds
    scale_up_cpu: float = 80.0
    scale_down_cpu: float = 30.0
    scale_up_memory: float = 80.0
    scale_down_memory: float = 30.0
    scale_up_queue_size: int = 200
    scale_down_queue_size: int = 10
    
    # Cooldown periods (seconds)
    scale_up_cooldown: int = 300      # 5 minutes
    scale_down_cooldown: int = 900    # 15 minutes


class ResourceAlert(BaseModel):
    """Resource alert model."""
    id: str
    timestamp: datetime
    resource_type: ResourceType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    suggested_action: Optional[ScalingAction] = None


class ScalingRecommendation(BaseModel):
    """Auto-scaling recommendation."""
    timestamp: datetime
    action: ScalingAction
    resource_type: ResourceType
    current_value: float
    target_value: float
    reason: str
    estimated_cost_impact: float
    confidence: float = 0.0


class ResourceMonitor:
    """Monitors system resources."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.last_metrics: Optional[ResourceMetrics] = None
        self.process = psutil.Process(os.getpid())
        
    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_in_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            network_out_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            
            # Process-specific metrics
            process_connections = len(self.process.connections())
            
            # Application metrics (placeholders)
            metrics = ResourceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_percent=disk_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_in_mb=network_in_mb,
                network_out_mb=network_out_mb,
                active_connections=process_connections,
                worker_count=5,  # Placeholder
                queue_size=np.random.randint(0, 100),  # Placeholder
                cache_hit_rate=np.random.uniform(0.7, 0.95),  # Placeholder
                database_connections=np.random.randint(10, 50),  # Placeholder
                response_time_ms=np.random.uniform(100, 500)  # Placeholder
            )
            
            self.metrics_history.append(metrics)
            self.last_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return self.last_metrics or self._default_metrics()
    
    def _default_metrics(self) -> ResourceMetrics:
        """Return default metrics if collection fails."""
        return ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_mb=0.0,
            disk_percent=0.0,
            disk_io_read_mb=0.0,
            disk_io_write_mb=0.0,
            network_in_mb=0.0,
            network_out_mb=0.0,
            active_connections=0,
            worker_count=0,
            queue_size=0,
            cache_hit_rate=0.0,
            database_connections=0,
            response_time_ms=0.0
        )
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        if not recent:
            return {}
        
        return {
            'avg_cpu': np.mean([m.cpu_percent for m in recent]),
            'max_cpu': np.max([m.cpu_percent for m in recent]),
            'avg_memory': np.mean([m.memory_percent for m in recent]),
            'max_memory': np.max([m.memory_percent for m in recent]),
            'avg_response_time': np.mean([m.response_time_ms for m in recent]),
            'avg_queue_size': np.mean([m.queue_size for m in recent]),
            'total_requests': sum(m.active_connections for m in recent)
        }


class AutoScaler:
    """Handles auto-scaling decisions."""
    
    def __init__(self, thresholds: ResourceThresholds):
        self.thresholds = thresholds
        self.last_scale_action: Dict[ResourceType, datetime] = {}
        self.scaling_history: List[ScalingRecommendation] = []
        
    async def analyze(
        self,
        metrics: ResourceMetrics,
        history: List[ResourceMetrics]
    ) -> Optional[ScalingRecommendation]:
        """Analyze metrics and recommend scaling action."""
        # Check cooldown periods
        for resource_type in ResourceType:
            if resource_type in self.last_scale_action:
                time_since = (datetime.utcnow() - self.last_scale_action[resource_type]).seconds
                if time_since < self.thresholds.scale_up_cooldown:
                    continue
        
        # CPU-based scaling
        if metrics.cpu_percent > self.thresholds.scale_up_cpu:
            return self._create_recommendation(
                ScalingAction.SCALE_UP,
                ResourceType.CPU,
                metrics.cpu_percent,
                50.0,
                "High CPU utilization"
            )
        elif metrics.cpu_percent < self.thresholds.scale_down_cpu:
            if self._is_safe_to_scale_down(history):
                return self._create_recommendation(
                    ScalingAction.SCALE_DOWN,
                    ResourceType.CPU,
                    metrics.cpu_percent,
                    50.0,
                    "Low CPU utilization"
                )
        
        # Memory-based scaling
        if metrics.memory_percent > self.thresholds.scale_up_memory:
            return self._create_recommendation(
                ScalingAction.SCALE_UP,
                ResourceType.MEMORY,
                metrics.memory_percent,
                60.0,
                "High memory utilization"
            )
        
        # Queue-based scaling
        if metrics.queue_size > self.thresholds.scale_up_queue_size:
            return self._create_recommendation(
                ScalingAction.SCALE_OUT,
                ResourceType.WORKERS,
                metrics.worker_count,
                metrics.worker_count + 2,
                "Large queue size"
            )
        
        return None
    
    def _create_recommendation(
        self,
        action: ScalingAction,
        resource_type: ResourceType,
        current_value: float,
        target_value: float,
        reason: str
    ) -> ScalingRecommendation:
        """Create scaling recommendation."""
        # Estimate cost impact
        cost_impact = 0.05 if action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT] else -0.05
        
        recommendation = ScalingRecommendation(
            timestamp=datetime.utcnow(),
            action=action,
            resource_type=resource_type,
            current_value=current_value,
            target_value=target_value,
            reason=reason,
            estimated_cost_impact=cost_impact,
            confidence=0.85
        )
        
        self.scaling_history.append(recommendation)
        self.last_scale_action[resource_type] = datetime.utcnow()
        
        return recommendation
    
    def _is_safe_to_scale_down(self, history: List[ResourceMetrics]) -> bool:
        """Check if it's safe to scale down."""
        if len(history) < 10:
            return False
        
        recent = history[-10:]
        cpu_values = [m.cpu_percent for m in recent]
        memory_values = [m.memory_percent for m in recent]
        
        # All recent values should be below threshold
        if any(cpu > self.thresholds.scale_down_cpu * 1.5 for cpu in cpu_values):
            return False
        
        if any(mem > self.thresholds.scale_down_memory * 1.5 for mem in memory_values):
            return False
        
        return True


class AlertManager:
    """Manages resource alerts."""
    
    def __init__(self, thresholds: ResourceThresholds):
        self.thresholds = thresholds
        self.active_alerts: Dict[str, ResourceAlert] = {}
        self.alert_history: List[ResourceAlert] = []
        self.kafka = None
        
    async def initialize(self):
        """Initialize alert manager."""
        self.kafka = KafkaClient()
        await self.kafka.initialize()
    
    async def check_alerts(self, metrics: ResourceMetrics) -> List[ResourceAlert]:
        """Check for alert conditions."""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_percent > self.thresholds.cpu_critical:
            alerts.append(self._create_alert(
                ResourceType.CPU,
                AlertSeverity.CRITICAL,
                f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                metrics.cpu_percent,
                self.thresholds.cpu_critical
            ))
        elif metrics.cpu_percent > self.thresholds.cpu_warning:
            alerts.append(self._create_alert(
                ResourceType.CPU,
                AlertSeverity.WARNING,
                f"CPU usage high: {metrics.cpu_percent:.1f}%",
                metrics.cpu_percent,
                self.thresholds.cpu_warning
            ))
        
        # Memory alerts
        if metrics.memory_percent > self.thresholds.memory_critical:
            alerts.append(self._create_alert(
                ResourceType.MEMORY,
                AlertSeverity.CRITICAL,
                f"Memory usage critical: {metrics.memory_percent:.1f}%",
                metrics.memory_percent,
                self.thresholds.memory_critical
            ))
        
        # Response time alerts
        if metrics.response_time_ms > self.thresholds.response_time_critical_ms:
            alerts.append(self._create_alert(
                ResourceType.NETWORK,
                AlertSeverity.CRITICAL,
                f"Response time critical: {metrics.response_time_ms:.0f}ms",
                metrics.response_time_ms,
                self.thresholds.response_time_critical_ms
            ))
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
        
        return alerts
    
    def _create_alert(
        self,
        resource_type: ResourceType,
        severity: AlertSeverity,
        message: str,
        current_value: float,
        threshold: float
    ) -> ResourceAlert:
        """Create resource alert."""
        alert = ResourceAlert(
            id=f"{resource_type}_{severity}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            resource_type=resource_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold=threshold
        )
        
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        return alert
    
    async def _send_alert(self, alert: ResourceAlert):
        """Send alert to notification systems."""
        try:
            if self.kafka:
                await self.kafka.publish(
                    'resource-alerts',
                    {
                        'alert_id': alert.id,
                        'timestamp': alert.timestamp.isoformat(),
                        'resource_type': alert.resource_type.value,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold
                    }
                )
            
            # Log alert
            if alert.severity == AlertSeverity.CRITICAL:
                logger.error(f"CRITICAL ALERT: {alert.message}")
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(f"WARNING ALERT: {alert.message}")
            else:
                logger.info(f"INFO ALERT: {alert.message}")
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")


class ResourceManager:
    """Main resource management orchestrator."""
    
    def __init__(self):
        self.thresholds = ResourceThresholds()
        self.monitor = ResourceMonitor()
        self.auto_scaler = AutoScaler(self.thresholds)
        self.alert_manager = AlertManager(self.thresholds)
        self.redis = None
        self.monitoring_task = None
        
    async def initialize(self):
        """Initialize resource manager."""
        self.redis = await get_redis()
        await self.alert_manager.initialize()
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Resource manager initialized")
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        metrics = await self.monitor.collect_metrics()
        summary = self.monitor.get_metrics_summary()
        
        return {
            'timestamp': metrics.timestamp.isoformat(),
            'resources': {
                'cpu': {
                    'current': metrics.cpu_percent,
                    'threshold_warning': self.thresholds.cpu_warning,
                    'threshold_critical': self.thresholds.cpu_critical
                },
                'memory': {
                    'current': metrics.memory_percent,
                    'used_mb': metrics.memory_mb,
                    'threshold_warning': self.thresholds.memory_warning,
                    'threshold_critical': self.thresholds.memory_critical
                },
                'disk': {
                    'current': metrics.disk_percent,
                    'threshold_warning': self.thresholds.disk_warning,
                    'threshold_critical': self.thresholds.disk_critical
                },
                'workers': {
                    'count': metrics.worker_count,
                    'queue_size': metrics.queue_size,
                    'connections': metrics.active_connections
                },
                'performance': {
                    'response_time_ms': metrics.response_time_ms,
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'database_connections': metrics.database_connections
                }
            },
            'summary': summary,
            'active_alerts': len(self.alert_manager.active_alerts),
            'last_scaling_action': self._get_last_scaling_action()
        }
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation."""
        try:
            metrics = await self.monitor.collect_metrics()
            summary = self.monitor.get_metrics_summary()
            
            recommendations = []
            
            # Check for over-provisioned resources
            if summary.get('avg_cpu', 0) < 20:
                recommendations.append({
                    'resource': 'CPU',
                    'action': 'Consider reducing CPU allocation',
                    'potential_savings': '$10-20/month'
                })
            
            if summary.get('avg_memory', 0) < 30:
                recommendations.append({
                    'resource': 'Memory',
                    'action': 'Consider reducing memory allocation',
                    'potential_savings': '$5-10/month'
                })
            
            # Check cache efficiency
            if metrics.cache_hit_rate < 0.7:
                recommendations.append({
                    'resource': 'Cache',
                    'action': 'Increase cache size for better hit rate',
                    'potential_benefit': 'Reduce API costs by 20%'
                })
            
            return {
                'current_efficiency': {
                    'cpu_utilization': summary.get('avg_cpu', 0),
                    'memory_utilization': summary.get('avg_memory', 0),
                    'cache_efficiency': metrics.cache_hit_rate
                },
                'recommendations': recommendations,
                'estimated_monthly_savings': self._calculate_potential_savings(summary)
            }
            
        except Exception as e:
            logger.error(f"Resource optimization error: {e}")
            return {'error': str(e)}
    
    def _get_last_scaling_action(self) -> Optional[Dict[str, Any]]:
        """Get last scaling action."""
        if not self.auto_scaler.scaling_history:
            return None
        
        last = self.auto_scaler.scaling_history[-1]
        return {
            'timestamp': last.timestamp.isoformat(),
            'action': last.action.value,
            'resource': last.resource_type.value,
            'reason': last.reason
        }
    
    def _calculate_potential_savings(self, summary: Dict[str, Any]) -> float:
        """Calculate potential monthly cost savings."""
        savings = 0.0
        
        # CPU savings
        if summary.get('avg_cpu', 0) < 30:
            savings += 15.0
        
        # Memory savings
        if summary.get('avg_memory', 0) < 30:
            savings += 10.0
        
        # Worker optimization
        if summary.get('avg_queue_size', 0) < 10:
            savings += 20.0
        
        return savings
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Collect metrics
                metrics = await self.monitor.collect_metrics()
                
                # Check for alerts
                alerts = await self.alert_manager.check_alerts(metrics)
                
                # Check for scaling needs
                history = list(self.monitor.metrics_history)
                recommendation = await self.auto_scaler.analyze(metrics, history)
                
                if recommendation:
                    logger.info(f"Scaling recommendation: {recommendation.action} for {recommendation.resource_type}")
                
                # Store metrics in Redis
                if self.redis:
                    metrics_data = {
                        'timestamp': metrics.timestamp.isoformat(),
                        'cpu': metrics.cpu_percent,
                        'memory': metrics.memory_percent,
                        'response_time': metrics.response_time_ms
                    }
                    await self.redis.setex(
                        f"metrics:{int(time.time())}",
                        3600,
                        json.dumps(metrics_data)
                    )
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
