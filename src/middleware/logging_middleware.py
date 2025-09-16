"""
Logging middleware for FastAPI requests.
"""

import time
import json
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

from src.core.logging_config import (
    LogContext,
    generate_request_id,
    generate_trace_id,
    performance_logger,
    security_logger
)

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate IDs
        request_id = generate_request_id()
        trace_id = request.headers.get("X-Trace-ID", generate_trace_id())
        
        # Store in request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Get user info if available
        user_id = getattr(request.state, "user_id", None)
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.url.query),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            user_id=user_id
        )
        
        # Process request with context
        with LogContext(request_id=request_id, trace_id=trace_id, user_id=user_id):
            try:
                response = await call_next(request)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log response
                logger.info(
                    "request_completed",
                    request_id=request_id,
                    trace_id=trace_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration * 1000,
                    user_id=user_id
                )
                
                # Log performance metrics
                performance_logger.log_api_call(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                    request_id=request_id
                )
                
                # Add trace headers to response
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Trace-ID"] = trace_id
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    "request_failed",
                    request_id=request_id,
                    trace_id=trace_id,
                    method=request.method,
                    path=request.url.path,
                    duration_ms=duration * 1000,
                    error=str(e),
                    error_type=type(e).__name__,
                    user_id=user_id,
                    exc_info=e
                )
                
                # Re-raise exception
                raise


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging of sensitive operations."""
    
    # Define sensitive endpoints
    SENSITIVE_ENDPOINTS = {
        "/api/v1/auth/login": "user_login",
        "/api/v1/auth/register": "user_registration",
        "/api/v1/auth/logout": "user_logout",
        "/api/v1/signals": "signal_creation",
        "/api/v1/signals/execute": "signal_execution",
    }
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = structlog.get_logger("audit")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if endpoint needs audit logging
        path = request.url.path
        
        if any(path.startswith(endpoint) for endpoint in self.SENSITIVE_ENDPOINTS):
            # Get request details
            request_id = getattr(request.state, "request_id", None)
            user_id = getattr(request.state, "user_id", None)
            
            # Capture request body for POST/PUT/PATCH
            request_body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    request_body = body.decode("utf-8")
                    # Reset body for downstream processing
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
                except:
                    pass
            
            # Log audit event
            self.audit_logger.info(
                "audit_event",
                event_type=self.SENSITIVE_ENDPOINTS.get(path, "unknown"),
                request_id=request_id,
                user_id=user_id,
                method=request.method,
                path=path,
                client_ip=request.client.host if request.client else None,
                request_body=request_body if request_body and len(request_body) < 1000 else None
            )
        
        # Process request
        response = await call_next(request)
        
        # Log response for sensitive endpoints
        if any(path.startswith(endpoint) for endpoint in self.SENSITIVE_ENDPOINTS):
            self.audit_logger.info(
                "audit_response",
                event_type=self.SENSITIVE_ENDPOINTS.get(path, "unknown"),
                request_id=getattr(request.state, "request_id", None),
                user_id=getattr(request.state, "user_id", None),
                status_code=response.status_code
            )
        
        return response


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed error logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_logger = structlog.get_logger("errors")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            
            # Log client errors (4xx)
            if 400 <= response.status_code < 500:
                self.error_logger.warning(
                    "client_error",
                    request_id=getattr(request.state, "request_id", None),
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    client_ip=request.client.host if request.client else None
                )
            
            # Log server errors (5xx)
            elif response.status_code >= 500:
                self.error_logger.error(
                    "server_error",
                    request_id=getattr(request.state, "request_id", None),
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code
                )
            
            return response
            
        except Exception as e:
            # Log unhandled exceptions
            self.error_logger.error(
                "unhandled_exception",
                request_id=getattr(request.state, "request_id", None),
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e
            )
            raise


class MetricsLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging metrics and statistics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics_logger = structlog.get_logger("metrics")
        self.request_count = 0
        self.error_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Increment request counter
        self.request_count += 1
        
        # Get request size
        request_size = int(request.headers.get("Content-Length", 0))
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Get response size
        response_size = int(response.headers.get("Content-Length", 0))
        
        # Track errors
        if response.status_code >= 400:
            self.error_count += 1
        
        # Log metrics
        self.metrics_logger.info(
            "request_metrics",
            request_id=getattr(request.state, "request_id", None),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration * 1000,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            total_requests=self.request_count,
            total_errors=self.error_count,
            error_rate=self.error_count / self.request_count if self.request_count > 0 else 0
        )
        
        return response
