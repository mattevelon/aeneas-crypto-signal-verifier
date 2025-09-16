"""
Distributed tracing context management for log correlation.
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from contextvars import ContextVar
from uuid import uuid4
import json

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

logger = structlog.get_logger()

# Context variables for trace correlation
trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('trace_context', default=None)


class TraceContextManager:
    """Manages distributed tracing context across the application."""
    
    def __init__(self, service_name: str = "aeneas", environment: str = "development"):
        self.service_name = service_name
        self.environment = environment
        self.tracer = None
        self._initialized = False
    
    def initialize(self, otlp_endpoint: Optional[str] = None):
        """Initialize OpenTelemetry tracing."""
        if self._initialized:
            return
        
        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.environment": self.environment,
            "service.version": "1.0.0"
        })
        
        # Set up tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter if endpoint provided
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        
        self._initialized = True
        logger.info("Trace context manager initialized", 
                   service=self.service_name,
                   environment=self.environment,
                   otlp_endpoint=otlp_endpoint)
    
    def instrument_app(self, app):
        """Instrument FastAPI application."""
        if not self._initialized:
            self.initialize()
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument database clients
        SQLAlchemyInstrumentor().instrument()
        RedisInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        
        logger.info("Application instrumented for tracing")
    
    def create_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL) -> trace.Span:
        """Create a new span."""
        if not self.tracer:
            self.initialize()
        
        return self.tracer.start_span(name, kind=kind)
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        span = trace.get_current_span()
        if span and span.get_span_context().trace_id:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID."""
        span = trace.get_current_span()
        if span and span.get_span_context().span_id:
            return format(span.get_span_context().span_id, '016x')
        return None
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers."""
        from opentelemetry.propagate import inject
        inject(headers)
        return headers
    
    def extract_context(self, headers: Dict[str, str]):
        """Extract trace context from headers."""
        from opentelemetry.propagate import extract
        return extract(headers)


class TraceCorrelator:
    """Correlates logs with traces."""
    
    def __init__(self, trace_manager: TraceContextManager):
        self.trace_manager = trace_manager
    
    def add_trace_context(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add trace context to log event."""
        # Add OpenTelemetry trace context
        if trace_id := self.trace_manager.get_current_trace_id():
            event_dict['trace_id'] = trace_id
        
        if span_id := self.trace_manager.get_current_span_id():
            event_dict['span_id'] = span_id
        
        # Add custom trace context
        if ctx := trace_context.get():
            event_dict.update(ctx)
        
        return event_dict
    
    def correlate_logs(self):
        """Set up log correlation with traces."""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                self.add_trace_context,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


class TracedOperation:
    """Context manager for traced operations."""
    
    def __init__(
        self,
        name: str,
        trace_manager: TraceContextManager,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.trace_manager = trace_manager
        self.kind = kind
        self.attributes = attributes or {}
        self.span = None
    
    def __enter__(self):
        self.span = self.trace_manager.create_span(self.name, self.kind)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        # Set trace context
        context = {
            "operation": self.name,
            "trace_id": self.trace_manager.get_current_trace_id(),
            "span_id": self.trace_manager.get_current_span_id()
        }
        trace_context.set(context)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # Set status based on exception
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()
        
        # Clear trace context
        trace_context.set(None)
        
        return False
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self.span:
            self.span.add_event(name, attributes=attributes)
    
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        if self.span:
            self.span.set_attribute(key, value)


class AsyncTracedOperation:
    """Async context manager for traced operations."""
    
    def __init__(
        self,
        name: str,
        trace_manager: TraceContextManager,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.trace_manager = trace_manager
        self.kind = kind
        self.attributes = attributes or {}
        self.span = None
    
    async def __aenter__(self):
        self.span = self.trace_manager.create_span(self.name, self.kind)
        
        # Add attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        # Set trace context
        context = {
            "operation": self.name,
            "trace_id": self.trace_manager.get_current_trace_id(),
            "span_id": self.trace_manager.get_current_span_id()
        }
        trace_context.set(context)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # Set status based on exception
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()
        
        # Clear trace context
        trace_context.set(None)
        
        return False


def trace_function(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """Decorator for tracing functions."""
    def decorator(func: Callable):
        operation_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                manager = TraceContextManager()
                async with AsyncTracedOperation(operation_name, manager, kind, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                manager = TraceContextManager()
                with TracedOperation(operation_name, manager, kind, attributes):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global trace manager instance
trace_manager = TraceContextManager()


def init_tracing(
    service_name: str = "aeneas",
    environment: str = "development",
    otlp_endpoint: Optional[str] = None
):
    """Initialize global tracing."""
    global trace_manager
    
    trace_manager = TraceContextManager(service_name, environment)
    trace_manager.initialize(otlp_endpoint)
    
    # Set up log correlation
    correlator = TraceCorrelator(trace_manager)
    correlator.correlate_logs()
    
    return trace_manager
