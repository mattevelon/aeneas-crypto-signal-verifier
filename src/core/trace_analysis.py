"""
Distributed tracing analysis and performance profiling.
Provides trace analysis, sampling strategies, and performance insights.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import defaultdict
import numpy as np
import structlog
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from src.config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()


class SamplingStrategy(Enum):
    """Trace sampling strategies."""
    ALWAYS = "always"
    NEVER = "never"
    PROBABILITY = "probability"
    RATE_LIMITED = "rate_limited"
    ADAPTIVE = "adaptive"
    ERROR_BASED = "error_based"


class TraceAnalyzer:
    """Analyzes distributed traces for performance insights."""
    
    def __init__(self):
        self.traces = {}
        self.span_statistics = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0,
            "errors": 0,
            "p50": [],
            "p95": [],
            "p99": []
        })
        self.critical_paths = {}
        self.bottlenecks = []
        self.anomalies = []
        
    def analyze_span(self, span: ReadableSpan):
        """Analyze a single span."""
        # Extract span data
        span_data = {
            "name": span.name,
            "trace_id": format(span.context.trace_id, '032x'),
            "span_id": format(span.context.span_id, '016x'),
            "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration": (span.end_time - span.start_time) / 1e9 if span.end_time else 0,  # Convert to seconds
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes) if span.attributes else {},
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {}
                }
                for event in span.events
            ] if span.events else [],
            "links": [
                {
                    "trace_id": format(link.context.trace_id, '032x'),
                    "span_id": format(link.context.span_id, '016x'),
                    "attributes": dict(link.attributes) if link.attributes else {}
                }
                for link in span.links
            ] if span.links else []
        }
        
        # Store span in trace
        trace_id = span_data["trace_id"]
        if trace_id not in self.traces:
            self.traces[trace_id] = {}
        self.traces[trace_id][span_data["span_id"]] = span_data
        
        # Update statistics
        self._update_statistics(span_data)
        
        # Check for anomalies
        self._detect_anomalies(span_data)
        
        return span_data
        
    def _update_statistics(self, span_data: Dict[str, Any]):
        """Update span statistics."""
        name = span_data["name"]
        duration = span_data["duration"]
        
        stats = self.span_statistics[name]
        stats["count"] += 1
        stats["total_duration"] += duration
        
        if span_data["status"] == "ERROR":
            stats["errors"] += 1
            
        # Store duration for percentile calculation
        stats["p50"].append(duration)
        stats["p95"].append(duration)
        stats["p99"].append(duration)
        
        # Keep only last 1000 samples for memory efficiency
        if len(stats["p50"]) > 1000:
            stats["p50"] = stats["p50"][-1000:]
            stats["p95"] = stats["p95"][-1000:]
            stats["p99"] = stats["p99"][-1000:]
            
    def _detect_anomalies(self, span_data: Dict[str, Any]):
        """Detect anomalies in span data."""
        # Detect slow spans
        if span_data["duration"] > 5.0:  # 5 seconds threshold
            self.anomalies.append({
                "type": "slow_span",
                "span_id": span_data["span_id"],
                "trace_id": span_data["trace_id"],
                "name": span_data["name"],
                "duration": span_data["duration"],
                "timestamp": datetime.fromtimestamp(span_data["start_time"] / 1e9)
            })
            
        # Detect error spans
        if span_data["status"] == "ERROR":
            self.anomalies.append({
                "type": "error_span",
                "span_id": span_data["span_id"],
                "trace_id": span_data["trace_id"],
                "name": span_data["name"],
                "timestamp": datetime.fromtimestamp(span_data["start_time"] / 1e9)
            })
            
        # Detect spans with many events (potential issues)
        if len(span_data.get("events", [])) > 10:
            self.anomalies.append({
                "type": "excessive_events",
                "span_id": span_data["span_id"],
                "trace_id": span_data["trace_id"],
                "name": span_data["name"],
                "event_count": len(span_data["events"]),
                "timestamp": datetime.fromtimestamp(span_data["start_time"] / 1e9)
            })
            
    def calculate_critical_path(self, trace_id: str) -> List[str]:
        """Calculate the critical path for a trace."""
        if trace_id not in self.traces:
            return []
            
        spans = self.traces[trace_id]
        
        # Build dependency graph
        graph = {}
        for span_id, span_data in spans.items():
            graph[span_id] = {
                "duration": span_data["duration"],
                "parent": span_data["parent_id"],
                "children": [],
                "name": span_data["name"]
            }
            
        # Find children for each span
        for span_id, node in graph.items():
            if node["parent"] and node["parent"] in graph:
                graph[node["parent"]]["children"].append(span_id)
                
        # Find root spans
        roots = [span_id for span_id, node in graph.items() if not node["parent"]]
        
        # Calculate critical path from each root
        critical_path = []
        max_duration = 0
        
        for root in roots:
            path, duration = self._find_critical_path_from_node(root, graph)
            if duration > max_duration:
                max_duration = duration
                critical_path = path
                
        self.critical_paths[trace_id] = {
            "path": critical_path,
            "total_duration": max_duration,
            "spans": [graph[span_id]["name"] for span_id in critical_path]
        }
        
        return critical_path
        
    def _find_critical_path_from_node(self, node_id: str, graph: Dict) -> Tuple[List[str], float]:
        """Find critical path from a specific node."""
        if node_id not in graph:
            return [], 0
            
        node = graph[node_id]
        
        if not node["children"]:
            return [node_id], node["duration"]
            
        # Find critical path among children
        max_child_path = []
        max_child_duration = 0
        
        for child_id in node["children"]:
            child_path, child_duration = self._find_critical_path_from_node(child_id, graph)
            if child_duration > max_child_duration:
                max_child_duration = child_duration
                max_child_path = child_path
                
        return [node_id] + max_child_path, node["duration"] + max_child_duration
        
    def identify_bottlenecks(self, threshold_percentile: float = 0.95) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for span_name, stats in self.span_statistics.items():
            if stats["count"] < 10:  # Need sufficient samples
                continue
                
            durations = stats["p95"]
            if not durations:
                continue
                
            p95_duration = np.percentile(durations, 95)
            avg_duration = stats["total_duration"] / stats["count"]
            
            # Identify bottleneck if p95 is significantly higher than average
            if p95_duration > avg_duration * 2:
                bottlenecks.append({
                    "span_name": span_name,
                    "avg_duration": avg_duration,
                    "p95_duration": p95_duration,
                    "count": stats["count"],
                    "error_rate": stats["errors"] / stats["count"] if stats["count"] > 0 else 0,
                    "severity": "high" if p95_duration > 5.0 else "medium"
                })
                
        # Sort by severity and duration
        bottlenecks.sort(key=lambda x: (x["severity"], -x["p95_duration"]))
        self.bottlenecks = bottlenecks
        
        return bottlenecks
        
    def get_span_statistics(self) -> Dict[str, Any]:
        """Get aggregated span statistics."""
        statistics = {}
        
        for span_name, stats in self.span_statistics.items():
            if stats["count"] == 0:
                continue
                
            durations = stats["p50"]
            statistics[span_name] = {
                "count": stats["count"],
                "avg_duration": stats["total_duration"] / stats["count"],
                "p50": np.percentile(durations, 50) if durations else 0,
                "p95": np.percentile(durations, 95) if durations else 0,
                "p99": np.percentile(durations, 99) if durations else 0,
                "error_rate": stats["errors"] / stats["count"],
                "total_time": stats["total_duration"]
            }
            
        return statistics
        
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific trace."""
        if trace_id not in self.traces:
            return None
            
        spans = self.traces[trace_id]
        
        # Calculate trace metrics
        total_spans = len(spans)
        total_duration = max(
            (span["end_time"] - span["start_time"]) / 1e9
            for span in spans.values()
            if span["end_time"]
        ) if spans else 0
        
        error_count = sum(1 for span in spans.values() if span["status"] == "ERROR")
        
        # Find root span
        root_spans = [
            span for span in spans.values()
            if not span["parent_id"]
        ]
        
        return {
            "trace_id": trace_id,
            "total_spans": total_spans,
            "total_duration": total_duration,
            "error_count": error_count,
            "error_rate": error_count / total_spans if total_spans > 0 else 0,
            "root_spans": len(root_spans),
            "critical_path": self.critical_paths.get(trace_id, {}),
            "span_breakdown": self._get_span_breakdown(spans)
        }
        
    def _get_span_breakdown(self, spans: Dict[str, Dict]) -> Dict[str, Any]:
        """Get breakdown of spans by type."""
        breakdown = defaultdict(lambda: {"count": 0, "total_duration": 0})
        
        for span in spans.values():
            span_type = span["attributes"].get("span.kind", "internal")
            breakdown[span_type]["count"] += 1
            breakdown[span_type]["total_duration"] += span["duration"]
            
        return dict(breakdown)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "summary": {
                "total_traces": len(self.traces),
                "total_spans": sum(len(spans) for spans in self.traces.values()),
                "total_anomalies": len(self.anomalies),
                "total_bottlenecks": len(self.bottlenecks)
            },
            "statistics": self.get_span_statistics(),
            "bottlenecks": self.bottlenecks[:10],  # Top 10 bottlenecks
            "recent_anomalies": self.anomalies[-20:],  # Last 20 anomalies
            "critical_paths": {
                trace_id: info
                for trace_id, info in self.critical_paths.items()
                if info["total_duration"] > 1.0  # Only show slow traces
            }
        }


class AdaptiveSampler:
    """Adaptive trace sampling based on system load and error rates."""
    
    def __init__(
        self,
        base_rate: float = 0.1,
        error_rate: float = 1.0,
        max_rate: float = 1.0,
        min_rate: float = 0.01
    ):
        self.base_rate = base_rate
        self.error_rate = error_rate
        self.max_rate = max_rate
        self.min_rate = min_rate
        
        self.current_rate = base_rate
        self.request_count = 0
        self.error_count = 0
        self.last_adjustment = datetime.utcnow()
        
    def should_sample(self, span_context: Any, has_error: bool = False) -> bool:
        """Decide whether to sample a trace."""
        # Always sample errors
        if has_error:
            return np.random.random() < self.error_rate
            
        # Adaptive sampling based on current rate
        return np.random.random() < self.current_rate
        
    def adjust_rate(self):
        """Adjust sampling rate based on system metrics."""
        now = datetime.utcnow()
        
        # Adjust every minute
        if (now - self.last_adjustment).total_seconds() < 60:
            return
            
        # Calculate error rate
        error_percentage = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        # Increase sampling if error rate is high
        if error_percentage > 5:
            self.current_rate = min(self.current_rate * 1.5, self.max_rate)
        # Decrease sampling if system is stable
        elif error_percentage < 1 and self.request_count > 1000:
            self.current_rate = max(self.current_rate * 0.8, self.min_rate)
        else:
            # Return to base rate
            self.current_rate = self.base_rate
            
        # Reset counters
        self.request_count = 0
        self.error_count = 0
        self.last_adjustment = now
        
        logger.info(f"Adjusted sampling rate to {self.current_rate:.2%}")
        
    def record_span(self, has_error: bool = False):
        """Record span for rate adjustment."""
        self.request_count += 1
        if has_error:
            self.error_count += 1
            
        # Periodically adjust rate
        if self.request_count % 100 == 0:
            self.adjust_rate()


class TraceAggregator(SpanExporter):
    """Custom span exporter that aggregates traces for analysis."""
    
    def __init__(self, analyzer: TraceAnalyzer):
        self.analyzer = analyzer
        self.exported_spans = 0
        
    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """Export spans to the analyzer."""
        try:
            for span in spans:
                self.analyzer.analyze_span(span)
                self.exported_spans += 1
                
            # Periodically calculate critical paths and bottlenecks
            if self.exported_spans % 100 == 0:
                # Analyze recent traces
                recent_traces = list(self.analyzer.traces.keys())[-10:]
                for trace_id in recent_traces:
                    self.analyzer.calculate_critical_path(trace_id)
                    
                # Update bottlenecks
                self.analyzer.identify_bottlenecks()
                
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
            
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class PerformanceProfiler:
    """Performance profiler for tracing hotspots."""
    
    def __init__(self):
        self.profiles = {}
        self.flame_graph_data = []
        
    def profile_trace(self, trace_id: str, spans: Dict[str, Dict]) -> Dict[str, Any]:
        """Profile a trace for performance analysis."""
        profile = {
            "trace_id": trace_id,
            "total_time": 0,
            "cpu_time": 0,
            "io_time": 0,
            "db_time": 0,
            "network_time": 0,
            "function_breakdown": defaultdict(float)
        }
        
        for span in spans.values():
            duration = span["duration"]
            profile["total_time"] += duration
            
            # Categorize span time
            span_type = span["attributes"].get("span.kind", "internal")
            
            if span_type == "server" or span_type == "client":
                profile["network_time"] += duration
            elif "database" in span["name"].lower() or "sql" in span["name"].lower():
                profile["db_time"] += duration
            elif "file" in span["name"].lower() or "disk" in span["name"].lower():
                profile["io_time"] += duration
            else:
                profile["cpu_time"] += duration
                
            # Function breakdown
            profile["function_breakdown"][span["name"]] += duration
            
        # Generate flame graph data
        self._generate_flame_graph_data(trace_id, spans)
        
        self.profiles[trace_id] = profile
        return profile
        
    def _generate_flame_graph_data(self, trace_id: str, spans: Dict[str, Dict]):
        """Generate data for flame graph visualization."""
        # Build call stack for each span
        for span_id, span in spans.items():
            stack = []
            current = span
            
            # Build stack by following parent chain
            while current:
                stack.append(current["name"])
                parent_id = current.get("parent_id")
                current = spans.get(parent_id) if parent_id else None
                
            # Reverse to get top-down stack
            stack.reverse()
            
            self.flame_graph_data.append({
                "stack": ";".join(stack),
                "value": span["duration"] * 1000  # Convert to milliseconds
            })
            
    def get_hotspots(self, trace_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N hotspots in a trace."""
        if trace_id not in self.profiles:
            return []
            
        profile = self.profiles[trace_id]
        
        # Sort functions by time spent
        hotspots = [
            {
                "function": func,
                "duration": duration,
                "percentage": (duration / profile["total_time"] * 100) if profile["total_time"] > 0 else 0
            }
            for func, duration in profile["function_breakdown"].items()
        ]
        
        hotspots.sort(key=lambda x: -x["duration"])
        return hotspots[:top_n]
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.profiles:
            return {}
            
        total_traces = len(self.profiles)
        avg_total_time = sum(p["total_time"] for p in self.profiles.values()) / total_traces
        avg_cpu_time = sum(p["cpu_time"] for p in self.profiles.values()) / total_traces
        avg_io_time = sum(p["io_time"] for p in self.profiles.values()) / total_traces
        avg_db_time = sum(p["db_time"] for p in self.profiles.values()) / total_traces
        avg_network_time = sum(p["network_time"] for p in self.profiles.values()) / total_traces
        
        return {
            "total_traces_profiled": total_traces,
            "average_times": {
                "total": avg_total_time,
                "cpu": avg_cpu_time,
                "io": avg_io_time,
                "database": avg_db_time,
                "network": avg_network_time
            },
            "time_breakdown": {
                "cpu_percentage": (avg_cpu_time / avg_total_time * 100) if avg_total_time > 0 else 0,
                "io_percentage": (avg_io_time / avg_total_time * 100) if avg_total_time > 0 else 0,
                "db_percentage": (avg_db_time / avg_total_time * 100) if avg_total_time > 0 else 0,
                "network_percentage": (avg_network_time / avg_total_time * 100) if avg_total_time > 0 else 0
            },
            "flame_graph_samples": len(self.flame_graph_data)
        }


# Global instances
trace_analyzer: Optional[TraceAnalyzer] = None
adaptive_sampler: Optional[AdaptiveSampler] = None
performance_profiler: Optional[PerformanceProfiler] = None


def init_trace_analysis():
    """Initialize trace analysis components."""
    global trace_analyzer, adaptive_sampler, performance_profiler
    
    trace_analyzer = TraceAnalyzer()
    adaptive_sampler = AdaptiveSampler()
    performance_profiler = PerformanceProfiler()
    
    logger.info("Trace analysis components initialized")
    
    return trace_analyzer, adaptive_sampler, performance_profiler


def get_trace_analyzer() -> TraceAnalyzer:
    """Get trace analyzer instance."""
    if not trace_analyzer:
        raise RuntimeError("Trace analyzer not initialized")
    return trace_analyzer


def get_adaptive_sampler() -> AdaptiveSampler:
    """Get adaptive sampler instance."""
    if not adaptive_sampler:
        raise RuntimeError("Adaptive sampler not initialized")
    return adaptive_sampler


def get_performance_profiler() -> PerformanceProfiler:
    """Get performance profiler instance."""
    if not performance_profiler:
        raise RuntimeError("Performance profiler not initialized")
    return performance_profiler
