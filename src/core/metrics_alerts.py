"""
Prometheus alerting rules and alert manager.
Defines alert conditions and handles alert routing.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import yaml
import aiohttp
import structlog
from pathlib import Path

from src.config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


class AlertRule:
    """Represents a Prometheus alert rule."""
    
    def __init__(
        self,
        name: str,
        expr: str,
        duration: str,
        severity: AlertSeverity,
        annotations: Dict[str, str],
        labels: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.expr = expr
        self.duration = duration
        self.severity = severity
        self.annotations = annotations
        self.labels = labels or {}
        
    def to_prometheus_rule(self) -> Dict[str, Any]:
        """Convert to Prometheus rule format."""
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.duration,
            "labels": {
                "severity": self.severity.value,
                **self.labels
            },
            "annotations": self.annotations
        }


class MetricsAlertManager:
    """Manages metric alerts and notifications."""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        
        # System alerts
        self.add_rule(AlertRule(
            name="HighCPUUsage",
            expr="system_cpu_usage_percent > 80",
            duration="5m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "High CPU usage detected",
                "description": "CPU usage is above 80% for more than 5 minutes"
            }
        ))
        
        self.add_rule(AlertRule(
            name="HighMemoryUsage",
            expr="(system_memory_usage_bytes / system_memory_total_bytes) * 100 > 90",
            duration="5m",
            severity=AlertSeverity.CRITICAL,
            annotations={
                "summary": "High memory usage detected",
                "description": "Memory usage is above 90% for more than 5 minutes"
            }
        ))
        
        self.add_rule(AlertRule(
            name="DiskSpaceLow",
            expr="system_disk_usage_percent > 85",
            duration="10m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Low disk space",
                "description": "Disk usage is above 85%"
            }
        ))
        
        # Application alerts
        self.add_rule(AlertRule(
            name="HighErrorRate",
            expr="rate(app_request_total{status=~'5..'}[5m]) > 0.05",
            duration="5m",
            severity=AlertSeverity.CRITICAL,
            annotations={
                "summary": "High error rate",
                "description": "5xx error rate is above 5% for 5 minutes"
            }
        ))
        
        self.add_rule(AlertRule(
            name="SlowResponseTime",
            expr="histogram_quantile(0.95, rate(app_request_duration_seconds_bucket[5m])) > 2",
            duration="10m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Slow API response time",
                "description": "95th percentile response time is above 2 seconds"
            }
        ))
        
        # Signal processing alerts
        self.add_rule(AlertRule(
            name="SignalProcessingBacklog",
            expr="rate(signals_processed_total[5m]) < rate(signals_received_total[5m]) * 0.9",
            duration="10m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Signal processing backlog",
                "description": "Signal processing rate is below 90% of receiving rate"
            }
        ))
        
        self.add_rule(AlertRule(
            name="LowSignalConfidence",
            expr="histogram_quantile(0.5, signal_confidence_score_bucket) < 0.5",
            duration="30m",
            severity=AlertSeverity.INFO,
            annotations={
                "summary": "Low signal confidence",
                "description": "Median signal confidence is below 50%"
            }
        ))
        
        # LLM alerts
        self.add_rule(AlertRule(
            name="HighLLMCost",
            expr="rate(llm_cost_dollars[1h]) > 10",
            duration="5m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "High LLM API costs",
                "description": "LLM costs exceeding $10/hour"
            },
            labels={"team": "finance"}
        ))
        
        self.add_rule(AlertRule(
            name="LLMHighErrorRate",
            expr="rate(llm_requests_total{status='error'}[5m]) / rate(llm_requests_total[5m]) > 0.1",
            duration="5m",
            severity=AlertSeverity.CRITICAL,
            annotations={
                "summary": "High LLM error rate",
                "description": "LLM error rate is above 10%"
            }
        ))
        
        # Database alerts
        self.add_rule(AlertRule(
            name="DatabaseConnectionPoolExhausted",
            expr="database_connections_active / database_connections_max > 0.9",
            duration="5m",
            severity=AlertSeverity.CRITICAL,
            annotations={
                "summary": "Database connection pool near exhaustion",
                "description": "Database connection pool is above 90% capacity"
            }
        ))
        
        self.add_rule(AlertRule(
            name="SlowDatabaseQueries",
            expr="histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m])) > 1",
            duration="10m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Slow database queries",
                "description": "95th percentile query time is above 1 second"
            }
        ))
        
        # Cache alerts
        self.add_rule(AlertRule(
            name="LowCacheHitRate",
            expr="rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.7",
            duration="15m",
            severity=AlertSeverity.INFO,
            annotations={
                "summary": "Low cache hit rate",
                "description": "Cache hit rate is below 70%"
            }
        ))
        
        # Business alerts
        self.add_rule(AlertRule(
            name="LowDailySignals",
            expr="business_daily_signals < 100",
            duration="1h",
            severity=AlertSeverity.INFO,
            annotations={
                "summary": "Low daily signal volume",
                "description": "Daily signal count is below 100"
            },
            labels={"team": "business"}
        ))
        
        self.add_rule(AlertRule(
            name="HighRiskSignals",
            expr="histogram_quantile(0.9, risk_score_distribution_bucket) > 0.8",
            duration="30m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "High risk signals",
                "description": "90th percentile risk score is above 0.8"
            }
        ))
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
        
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """Register an alert handler for a channel."""
        self.alert_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")
        
    async def evaluate_alerts(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against current metrics."""
        # This would typically query Prometheus
        # For now, we'll simulate alert evaluation
        
        for rule in self.alert_rules:
            # Check if alert condition is met
            # In production, this would use PromQL evaluation
            alert_key = f"{rule.name}"
            
            # Simulate alert triggering logic
            if self._should_trigger_alert(rule, metrics):
                if alert_key not in self.active_alerts:
                    # New alert
                    await self._trigger_alert(rule, metrics)
                    self.active_alerts[alert_key] = {
                        "rule": rule,
                        "triggered_at": datetime.utcnow(),
                        "metrics": metrics
                    }
            else:
                if alert_key in self.active_alerts:
                    # Alert resolved
                    await self._resolve_alert(rule, self.active_alerts[alert_key])
                    del self.active_alerts[alert_key]
                    
    def _should_trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Check if alert should be triggered."""
        # Simplified logic for demonstration
        # In production, this would evaluate the PromQL expression
        
        if rule.name == "HighCPUUsage":
            return metrics.get("system", {}).get("cpu_usage", 0) > 80
        elif rule.name == "HighErrorRate":
            total_requests = metrics.get("application", {}).get("total_requests", 1)
            error_requests = metrics.get("application", {}).get("error_requests", 0)
            return (error_requests / total_requests) > 0.05 if total_requests > 0 else False
            
        return False
        
    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_data = {
            "name": rule.name,
            "severity": rule.severity.value,
            "annotations": rule.annotations,
            "labels": rule.labels,
            "triggered_at": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        
        logger.warning(f"Alert triggered: {rule.name}", alert=alert_data)
        
        # Send to appropriate channels based on severity
        channels = self._get_channels_for_severity(rule.severity)
        
        for channel in channels:
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](alert_data)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel.value}: {e}")
                    
    async def _resolve_alert(self, rule: AlertRule, alert_info: Dict[str, Any]):
        """Resolve an alert."""
        duration = datetime.utcnow() - alert_info["triggered_at"]
        
        logger.info(
            f"Alert resolved: {rule.name}",
            duration=duration.total_seconds(),
            triggered_at=alert_info["triggered_at"]
        )
        
        # Notify resolution
        resolution_data = {
            "name": rule.name,
            "severity": rule.severity.value,
            "resolved_at": datetime.utcnow().isoformat(),
            "duration_seconds": duration.total_seconds()
        }
        
        # Send resolution notification
        for channel in self._get_channels_for_severity(AlertSeverity.INFO):
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](resolution_data)
                except Exception as e:
                    logger.error(f"Failed to send resolution to {channel.value}: {e}")
                    
    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get notification channels for a severity level."""
        if severity == AlertSeverity.EMERGENCY:
            return [AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.EMAIL]
        elif severity == AlertSeverity.CRITICAL:
            return [AlertChannel.SLACK, AlertChannel.EMAIL]
        elif severity == AlertSeverity.WARNING:
            return [AlertChannel.SLACK, AlertChannel.LOG]
        else:  # INFO
            return [AlertChannel.LOG]
            
    def generate_prometheus_rules(self) -> str:
        """Generate Prometheus rules configuration."""
        groups = [{
            "name": "aeneas_alerts",
            "interval": "30s",
            "rules": [rule.to_prometheus_rule() for rule in self.alert_rules]
        }]
        
        return yaml.dump({"groups": groups}, default_flow_style=False)
        
    def save_rules_to_file(self, filepath: Path):
        """Save rules to a YAML file."""
        rules_yaml = self.generate_prometheus_rules()
        filepath.write_text(rules_yaml)
        logger.info(f"Saved {len(self.alert_rules)} rules to {filepath}")
        
    async def send_test_alert(self, channel: AlertChannel) -> bool:
        """Send a test alert to verify channel configuration."""
        test_alert = {
            "name": "TestAlert",
            "severity": "info",
            "annotations": {
                "summary": "Test alert",
                "description": "This is a test alert to verify channel configuration"
            },
            "test": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if channel in self.alert_handlers:
            try:
                await self.alert_handlers[channel](test_alert)
                logger.info(f"Test alert sent successfully to {channel.value}")
                return True
            except Exception as e:
                logger.error(f"Failed to send test alert to {channel.value}: {e}")
                return False
        else:
            logger.warning(f"No handler registered for channel {channel.value}")
            return False
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        return [
            {
                "name": alert_key,
                "severity": info["rule"].severity.value,
                "triggered_at": info["triggered_at"].isoformat(),
                "duration": (datetime.utcnow() - info["triggered_at"]).total_seconds(),
                "annotations": info["rule"].annotations
            }
            for alert_key, info in self.active_alerts.items()
        ]


# Default alert handlers
async def log_alert_handler(alert_data: Dict[str, Any]):
    """Log alert handler."""
    logger.warning("Alert notification", **alert_data)
    

async def slack_alert_handler(alert_data: Dict[str, Any]):
    """Slack alert handler."""
    webhook_url = getattr(settings, 'slack_webhook_url', None)
    if not webhook_url:
        logger.warning("Slack webhook URL not configured")
        return
        
    payload = {
        "text": f"🚨 Alert: {alert_data['name']}",
        "attachments": [{
            "color": "danger" if alert_data['severity'] in ['critical', 'emergency'] else "warning",
            "fields": [
                {"title": "Severity", "value": alert_data['severity'], "short": True},
                {"title": "Summary", "value": alert_data['annotations']['summary'], "short": False},
                {"title": "Description", "value": alert_data['annotations']['description'], "short": False},
            ],
            "footer": "AENEAS Alert System",
            "ts": int(datetime.utcnow().timestamp())
        }]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=payload) as response:
            if response.status != 200:
                logger.error(f"Failed to send Slack alert: {response.status}")


# Global alert manager
alert_manager: Optional[MetricsAlertManager] = None


def init_alert_manager() -> MetricsAlertManager:
    """Initialize global alert manager."""
    global alert_manager
    
    alert_manager = MetricsAlertManager()
    
    # Register default handlers
    alert_manager.register_handler(AlertChannel.LOG, log_alert_handler)
    alert_manager.register_handler(AlertChannel.SLACK, slack_alert_handler)
    
    return alert_manager


def get_alert_manager() -> MetricsAlertManager:
    """Get global alert manager."""
    if not alert_manager:
        raise RuntimeError("Alert manager not initialized")
    return alert_manager
