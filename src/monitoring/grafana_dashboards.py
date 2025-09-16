"""
Grafana dashboard configurations for AENEAS monitoring.
Provides dashboard definitions and management utilities.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class GrafanaDashboard:
    """Represents a Grafana dashboard configuration."""
    
    def __init__(self, title: str, uid: str, description: str = ""):
        self.title = title
        self.uid = uid
        self.description = description
        self.panels = []
        self.templating = {"list": []}
        self.time = {"from": "now-6h", "to": "now"}
        self.refresh = "30s"
        
    def add_panel(self, panel: Dict[str, Any]):
        """Add a panel to the dashboard."""
        panel["id"] = len(self.panels) + 1
        self.panels.append(panel)
        
    def add_variable(self, variable: Dict[str, Any]):
        """Add a template variable."""
        self.templating["list"].append(variable)
        
    def to_json(self) -> Dict[str, Any]:
        """Convert dashboard to JSON format."""
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["aeneas", "monitoring"],
            "style": "dark",
            "timezone": "utc",
            "editable": True,
            "hideControls": False,
            "graphTooltip": 1,
            "panels": self.panels,
            "time": self.time,
            "timepicker": {
                "refresh_intervals": ["10s", "30s", "1m", "5m", "15m", "30m", "1h"],
                "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
            },
            "templating": self.templating,
            "annotations": {
                "list": [
                    {
                        "datasource": "Prometheus",
                        "enable": True,
                        "expr": "ALERTS{alertstate=\"firing\"}",
                        "iconColor": "rgba(255, 96, 96, 1)",
                        "name": "Active Alerts",
                        "step": "30s",
                        "tagKeys": "severity,alertname",
                        "titleFormat": "{{ alertname }}",
                        "type": "tags"
                    }
                ]
            },
            "refresh": self.refresh,
            "schemaVersion": 16,
            "version": 1
        }


def create_system_overview_dashboard() -> GrafanaDashboard:
    """Create system overview dashboard."""
    dashboard = GrafanaDashboard(
        title="AENEAS System Overview",
        uid="aeneas-overview",
        description="Main system health and performance metrics"
    )
    
    # Add template variables
    dashboard.add_variable({
        "name": "datasource",
        "label": "Data Source",
        "type": "datasource",
        "query": "prometheus",
        "current": {"text": "Prometheus", "value": "Prometheus"},
        "hide": 0
    })
    
    # Row 1: System Metrics
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 0, "y": 0},
        "type": "graph",
        "title": "CPU Usage",
        "datasource": "$datasource",
        "targets": [
            {
                "expr": "system_cpu_usage_percent",
                "legendFormat": "CPU %",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "percent", "max": 100, "min": 0},
        "lines": True,
        "fill": 1,
        "linewidth": 2,
        "nullPointMode": "null",
        "thresholds": [
            {"value": 80, "colorMode": "critical", "fill": True, "line": True, "op": "gt"}
        ]
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 8, "y": 0},
        "type": "graph",
        "title": "Memory Usage",
        "datasource": "$datasource",
        "targets": [
            {
                "expr": "system_memory_usage_bytes / 1024 / 1024 / 1024",
                "legendFormat": "Used GB",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "gbytes"},
        "lines": True,
        "fill": 1,
        "linewidth": 2
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 16, "y": 0},
        "type": "stat",
        "title": "System Uptime",
        "datasource": "$datasource",
        "targets": [
            {
                "expr": "(time() - process_start_time_seconds)",
                "refId": "A"
            }
        ],
        "format": "s",
        "decimals": 0,
        "colorMode": "background",
        "graphMode": "area",
        "orientation": "horizontal"
    })
    
    # Row 2: Application Metrics
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 0, "y": 7},
        "type": "graph",
        "title": "Request Rate",
        "datasource": "$datasource",
        "targets": [
            {
                "expr": "rate(app_request_total[1m])",
                "legendFormat": "{{method}} {{endpoint}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "reqps"},
        "lines": True,
        "fill": 1,
        "stack": True
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 12, "y": 7},
        "type": "graph",
        "title": "Response Time (p95)",
        "datasource": "$datasource",
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(app_request_duration_seconds_bucket[5m]))",
                "legendFormat": "p95",
                "refId": "A"
            },
            {
                "expr": "histogram_quantile(0.99, rate(app_request_duration_seconds_bucket[5m]))",
                "legendFormat": "p99",
                "refId": "B"
            }
        ],
        "yaxis": {"format": "s"},
        "lines": True,
        "fill": 0,
        "linewidth": 2
    })
    
    # Row 3: Active Alerts
    dashboard.add_panel({
        "gridPos": {"h": 4, "w": 24, "x": 0, "y": 14},
        "type": "alertlist",
        "title": "Active Alerts",
        "datasource": "$datasource",
        "options": {
            "showOptions": "current",
            "maxItems": 10,
            "sortOrder": 1,
            "dashboardAlerts": False,
            "alertName": "",
            "dashboardTitle": "",
            "tags": []
        }
    })
    
    return dashboard


def create_signal_processing_dashboard() -> GrafanaDashboard:
    """Create signal processing dashboard."""
    dashboard = GrafanaDashboard(
        title="AENEAS Signal Processing",
        uid="aeneas-signals",
        description="Signal detection and processing metrics"
    )
    
    # Signal Processing Metrics
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "type": "graph",
        "title": "Signal Processing Rate",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(signals_processed_total[5m])",
                "legendFormat": "{{source}} - {{status}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "ops"},
        "lines": True,
        "fill": 1,
        "stack": True
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "type": "heatmap",
        "title": "Signal Confidence Distribution",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "signal_confidence_score_bucket",
                "format": "heatmap",
                "refId": "A"
            }
        ],
        "dataFormat": "tsbuckets",
        "yAxis": {"format": "percentunit"},
        "cards": {"cardPadding": 2, "cardRound": 2},
        "color": {
            "mode": "spectrum",
            "scheme": "interpolateViridis",
            "exponent": 0.5
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
        "type": "stat",
        "title": "Total Signals Today",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "business_daily_signals",
                "refId": "A"
            }
        ],
        "format": "short",
        "colorMode": "value",
        "graphMode": "area",
        "orientation": "horizontal",
        "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 100},
                {"color": "green", "value": 500}
            ]
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
        "type": "gauge",
        "title": "Average Processing Time",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(signal_processing_duration_seconds_sum[5m]) / rate(signal_processing_duration_seconds_count[5m])",
                "refId": "A"
            }
        ],
        "format": "s",
        "min": 0,
        "max": 10,
        "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 2},
                {"color": "red", "value": 5}
            ]
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
        "type": "piechart",
        "title": "Signal Validation Results",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "sum by (result) (signals_validated_total)",
                "legendFormat": "{{result}}",
                "refId": "A"
            }
        ],
        "pieType": "donut",
        "legendDisplayMode": "table",
        "legendPlacement": "right",
        "legendValues": ["value", "percent"]
    })
    
    return dashboard


def create_llm_metrics_dashboard() -> GrafanaDashboard:
    """Create LLM metrics dashboard."""
    dashboard = GrafanaDashboard(
        title="AENEAS LLM Metrics",
        uid="aeneas-llm",
        description="LLM API usage and performance metrics"
    )
    
    # LLM Request Metrics
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 0, "y": 0},
        "type": "graph",
        "title": "LLM Request Rate",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(llm_requests_total[5m])",
                "legendFormat": "{{provider}} - {{model}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "reqps"},
        "lines": True,
        "fill": 1
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 12, "y": 0},
        "type": "graph",
        "title": "LLM Response Time",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(llm_response_duration_seconds_bucket[5m]))",
                "legendFormat": "{{provider}} p95",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "s"},
        "lines": True,
        "fill": 0,
        "linewidth": 2
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 0, "y": 7},
        "type": "graph",
        "title": "Token Usage",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(llm_tokens_used_total[1h])",
                "legendFormat": "{{provider}} - {{type}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "short"},
        "lines": True,
        "fill": 1,
        "stack": True
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 12, "y": 7},
        "type": "graph",
        "title": "LLM API Cost ($/hour)",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(llm_cost_dollars[1h]) * 3600",
                "legendFormat": "{{provider}} - {{model}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "currencyUSD"},
        "lines": True,
        "fill": 1,
        "stack": True,
        "alert": {
            "conditions": [
                {
                    "evaluator": {"params": [10], "type": "gt"},
                    "operator": {"type": "and"},
                    "query": {"params": ["A", "5m", "now"]},
                    "reducer": {"params": [], "type": "avg"},
                    "type": "query"
                }
            ],
            "executionErrorState": "alerting",
            "frequency": "60s",
            "handler": 1,
            "name": "High LLM Cost Alert",
            "noDataState": "no_data",
            "notifications": []
        }
    })
    
    return dashboard


def create_database_performance_dashboard() -> GrafanaDashboard:
    """Create database performance dashboard."""
    dashboard = GrafanaDashboard(
        title="AENEAS Database Performance",
        uid="aeneas-database",
        description="Database connections and query performance"
    )
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 0, "y": 0},
        "type": "graph",
        "title": "Active Database Connections",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "database_connections_active",
                "legendFormat": "{{database}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "short"},
        "lines": True,
        "fill": 1
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 12, "x": 12, "y": 0},
        "type": "heatmap",
        "title": "Query Duration Heatmap",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "database_query_duration_seconds_bucket",
                "format": "heatmap",
                "refId": "A"
            }
        ],
        "dataFormat": "tsbuckets",
        "yAxis": {"format": "s"},
        "cards": {"cardPadding": 2, "cardRound": 2}
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 0, "y": 7},
        "type": "graph",
        "title": "Cache Hit Rate",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) * 100",
                "legendFormat": "{{cache_level}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "percent", "max": 100, "min": 0},
        "lines": True,
        "fill": 1
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 8, "y": 7},
        "type": "stat",
        "title": "Cache Size",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "sum by (cache_level) (cache_size_bytes) / 1024 / 1024",
                "legendFormat": "{{cache_level}}",
                "refId": "A"
            }
        ],
        "format": "mbytes",
        "colorMode": "value",
        "graphMode": "area"
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 7, "w": 8, "x": 16, "y": 7},
        "type": "graph",
        "title": "Database Errors",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(database_errors_total[5m])",
                "legendFormat": "{{database}} - {{error_type}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "short"},
        "lines": True,
        "fill": 1,
        "bars": False,
        "nullPointMode": "null as zero"
    })
    
    return dashboard


def create_business_metrics_dashboard() -> GrafanaDashboard:
    """Create business metrics dashboard."""
    dashboard = GrafanaDashboard(
        title="AENEAS Business Metrics",
        uid="aeneas-business",
        description="Business KPIs and trading metrics"
    )
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
        "type": "stat",
        "title": "Daily Active Users",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "business_active_users{timeframe=\"daily\"}",
                "refId": "A"
            }
        ],
        "format": "short",
        "colorMode": "background",
        "graphMode": "area",
        "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 10},
                {"color": "green", "value": 50}
            ]
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
        "type": "stat",
        "title": "Total P&L Today",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "sum(profit_loss_dollars{timeframe=\"daily\"})",
                "refId": "A"
            }
        ],
        "format": "currencyUSD",
        "colorMode": "value",
        "graphMode": "area",
        "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "red", "value": -1000},
                {"color": "yellow", "value": 0},
                {"color": "green", "value": 1000}
            ]
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
        "type": "bargauge",
        "title": "Risk Score Distribution",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "histogram_quantile(0.5, risk_score_distribution_bucket)",
                "legendFormat": "Median",
                "refId": "A"
            },
            {
                "expr": "histogram_quantile(0.9, risk_score_distribution_bucket)",
                "legendFormat": "p90",
                "refId": "B"
            },
            {
                "expr": "histogram_quantile(0.95, risk_score_distribution_bucket)",
                "legendFormat": "p95",
                "refId": "C"
            }
        ],
        "orientation": "horizontal",
        "displayMode": "gradient",
        "thresholds": {
            "mode": "absolute",
            "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 0.8}
            ]
        }
    })
    
    dashboard.add_panel({
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
        "type": "graph",
        "title": "Revenue Trend",
        "datasource": "Prometheus",
        "targets": [
            {
                "expr": "rate(business_revenue_dollars[1d])",
                "legendFormat": "{{source}}",
                "refId": "A"
            }
        ],
        "yaxis": {"format": "currencyUSD"},
        "lines": True,
        "fill": 1,
        "stack": True,
        "aliasColors": {
            "subscriptions": "#7EB26D",
            "api_calls": "#EAB839",
            "premium_signals": "#6ED0E0"
        }
    })
    
    return dashboard


class DashboardManager:
    """Manages Grafana dashboards."""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", api_key: Optional[str] = None):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.dashboards = {}
        
        # Initialize default dashboards
        self._init_default_dashboards()
        
    def _init_default_dashboards(self):
        """Initialize default dashboards."""
        self.dashboards["overview"] = create_system_overview_dashboard()
        self.dashboards["signals"] = create_signal_processing_dashboard()
        self.dashboards["llm"] = create_llm_metrics_dashboard()
        self.dashboards["database"] = create_database_performance_dashboard()
        self.dashboards["business"] = create_business_metrics_dashboard()
        
    def export_dashboards(self, output_dir: Path):
        """Export all dashboards to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, dashboard in self.dashboards.items():
            filepath = output_dir / f"{dashboard.uid}.json"
            with open(filepath, 'w') as f:
                json.dump(dashboard.to_json(), f, indent=2)
            logger.info(f"Exported dashboard to {filepath}")
            
    def get_dashboard_json(self, dashboard_name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard JSON by name."""
        if dashboard_name in self.dashboards:
            return self.dashboards[dashboard_name].to_json()
        return None


# Initialize dashboard manager
def init_dashboard_manager(grafana_url: Optional[str] = None, api_key: Optional[str] = None) -> DashboardManager:
    """Initialize dashboard manager."""
    from src.config.settings import get_settings
    settings = get_settings()
    
    if not grafana_url:
        grafana_url = getattr(settings, 'grafana_url', 'http://localhost:3000')
    if not api_key:
        api_key = getattr(settings, 'grafana_api_key', None)
        
    return DashboardManager(grafana_url, api_key)
