"""
Structured logging configuration for the AENEAS system.
"""

import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar
from uuid import uuid4

import structlog
from pythonjsonlogger import jsonlogger

from src.config.settings import get_settings

settings = get_settings()

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional metadata."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add context variables
        if request_id := request_id_var.get():
            log_record['request_id'] = request_id
        if user_id := user_id_var.get():
            log_record['user_id'] = user_id
        if trace_id := trace_id_var.get():
            log_record['trace_id'] = trace_id
        
        # Add service metadata
        log_record['service'] = 'aeneas'
        log_record['environment'] = settings.environment
        log_record['version'] = settings.app_version
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add source location
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def add_context_processor(logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add context information to log events."""
    # Add request context
    if request_id := request_id_var.get():
        event_dict['request_id'] = request_id
    if user_id := user_id_var.get():
        event_dict['user_id'] = user_id
    if trace_id := trace_id_var.get():
        event_dict['trace_id'] = trace_id
    
    # Add timestamp
    event_dict['timestamp'] = datetime.utcnow().isoformat()
    
    # Add service metadata
    event_dict['service'] = 'aeneas'
    event_dict['environment'] = settings.app_env
    
    return event_dict


def setup_logging(
    log_level: str = None,
    log_file: Optional[str] = None,
    json_logs: bool = True,
    enable_console: bool = True
):
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        json_logs: Whether to output logs in JSON format
        enable_console: Whether to enable console logging
    """
    log_level = log_level or settings.log_level
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            add_context_processor,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure Python logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_logs:
            formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        if json_logs:
            formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    
    return structlog.get_logger()


class LogContext:
    """Context manager for adding contextual information to logs."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.tokens = {}
    
    def __enter__(self):
        for key, value in self.context.items():
            if key == 'request_id':
                self.tokens[key] = request_id_var.set(value)
            elif key == 'user_id':
                self.tokens[key] = user_id_var.set(value)
            elif key == 'trace_id':
                self.tokens[key] = trace_id_var.set(value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, token in self.tokens.items():
            if key == 'request_id':
                request_id_var.reset(token)
            elif key == 'user_id':
                user_id_var.reset(token)
            elif key == 'trace_id':
                trace_id_var.reset(token)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid4().hex[:12]}"


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid4().hex[:16]}"


# Performance logging utilities
class PerformanceLogger:
    """Utility for logging performance metrics."""
    
    def __init__(self, logger=None):
        self.logger = logger or structlog.get_logger()
    
    def log_operation(self, operation: str, duration: float, **kwargs):
        """Log operation performance."""
        self.logger.info(
            "operation_performance",
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_api_call(self, method: str, endpoint: str, status_code: int, duration: float, **kwargs):
        """Log API call performance."""
        self.logger.info(
            "api_call",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_db_query(self, query_type: str, table: str, duration: float, rows_affected: int = 0, **kwargs):
        """Log database query performance."""
        self.logger.info(
            "db_query",
            query_type=query_type,
            table=table,
            duration_ms=duration * 1000,
            rows_affected=rows_affected,
            **kwargs
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, duration: float, **kwargs):
        """Log cache operation."""
        self.logger.info(
            "cache_operation",
            operation=operation,
            key=key,
            cache_hit=hit,
            duration_ms=duration * 1000,
            **kwargs
        )


# Security logging utilities
class SecurityLogger:
    """Utility for logging security events."""
    
    def __init__(self, logger=None):
        self.logger = logger or structlog.get_logger()
    
    def log_auth_attempt(self, username: str, success: bool, ip_address: str, **kwargs):
        """Log authentication attempt."""
        self.logger.info(
            "auth_attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            event_type="authentication",
            **kwargs
        )
    
    def log_access_denied(self, user_id: str, resource: str, reason: str, **kwargs):
        """Log access denied event."""
        self.logger.warning(
            "access_denied",
            user_id=user_id,
            resource=resource,
            reason=reason,
            event_type="authorization",
            **kwargs
        )
    
    def log_rate_limit(self, identifier: str, endpoint: str, limit: int, **kwargs):
        """Log rate limit event."""
        self.logger.warning(
            "rate_limit_exceeded",
            identifier=identifier,
            endpoint=endpoint,
            limit=limit,
            event_type="rate_limiting",
            **kwargs
        )
    
    def log_suspicious_activity(self, user_id: Optional[str], activity: str, details: Dict[str, Any], **kwargs):
        """Log suspicious activity."""
        self.logger.warning(
            "suspicious_activity",
            user_id=user_id,
            activity=activity,
            details=details,
            event_type="security",
            **kwargs
        )


# Business event logging
class BusinessEventLogger:
    """Utility for logging business events."""
    
    def __init__(self, logger=None):
        self.logger = logger or structlog.get_logger()
    
    def log_signal_created(self, signal_id: str, pair: str, direction: str, confidence: float, **kwargs):
        """Log signal creation."""
        self.logger.info(
            "signal_created",
            signal_id=signal_id,
            pair=pair,
            direction=direction,
            confidence=confidence,
            event_type="signal",
            **kwargs
        )
    
    def log_signal_executed(self, signal_id: str, execution_price: float, slippage: float, **kwargs):
        """Log signal execution."""
        self.logger.info(
            "signal_executed",
            signal_id=signal_id,
            execution_price=execution_price,
            slippage=slippage,
            event_type="execution",
            **kwargs
        )
    
    def log_risk_alert(self, signal_id: str, risk_level: str, reason: str, **kwargs):
        """Log risk alert."""
        self.logger.warning(
            "risk_alert",
            signal_id=signal_id,
            risk_level=risk_level,
            reason=reason,
            event_type="risk",
            **kwargs
        )


# Initialize loggers
logger = setup_logging()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
business_logger = BusinessEventLogger()
