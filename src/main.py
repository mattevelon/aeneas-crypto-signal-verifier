"""
Main FastAPI application for the Crypto Signals Verification System.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import structlog
import uvicorn

from src.config.settings import settings
from src.api import health, signals, websocket, channels, performance, collector
from src.core.database import init_db, close_db
from src.core.redis_client import init_redis, close_redis
from src.core.kafka_client import init_kafka, close_kafka
from src.core.qdrant_client import init_qdrant
from src.core.cache_warmer import start_cache_warming, stop_cache_warming
from src.data_ingestion.telegram_collector_enhanced import (
    start_enhanced_telegram_collector,
    stop_enhanced_telegram_collector,
    get_collector_statistics
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Crypto Signals Verification System")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await init_redis()
    logger.info("Redis initialized")
    
    # Initialize Kafka (non-critical, allow failure)
    try:
        await init_kafka()
        logger.info("Kafka initialized")
    except Exception as e:
        logger.warning(f"Kafka initialization failed (non-critical): {e}")
    
    # Initialize Qdrant (non-critical for basic operation)
    try:
        await init_qdrant()
        logger.info("Qdrant initialized")
    except Exception as e:
        logger.warning(f"Qdrant initialization failed (non-critical): {e}")
    
    # Start cache warming (non-critical)
    try:
        await start_cache_warming()
        logger.info("Cache warming started")
    except Exception as e:
        logger.warning(f"Cache warming failed to start (non-critical): {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Crypto Signals Verification System")
    
    try:
        await stop_cache_warming()
    except Exception as e:
        logger.warning(f"Error stopping cache warming: {e}")
    
    await close_db()
    await close_redis()
    
    try:
        await close_kafka()
    except Exception as e:
        logger.warning(f"Error closing Kafka: {e}")
    
    logger.info("Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Crypto Signals Verification API",
    description="AI-powered cryptocurrency trading signal verification system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on environment
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API Routes
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(signals.router, prefix=settings.api_prefix)
app.include_router(channels.router, prefix=settings.api_prefix)
app.include_router(performance.router, prefix=settings.api_prefix)
app.include_router(collector.router, prefix=settings.api_prefix)
app.include_router(websocket.router, prefix=settings.api_prefix, tags=["websocket"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "Crypto Signals Verification API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        workers=1 if settings.debug else 4
    )
