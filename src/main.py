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
from src.api import health, signals, websocket, channels, performance
from src.core.database import init_db, close_db
from src.core.redis_client import init_redis, close_redis
from src.core.kafka_client import init_kafka, close_kafka
from src.core.qdrant_client import init_qdrant

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
    
    # Initialize Kafka
    await init_kafka()
    logger.info("Kafka initialized")
    
    # Initialize Qdrant
    await init_qdrant()
    logger.info("Qdrant initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Crypto Signals Verification System")
    await close_db()
    await close_redis()
    await close_kafka()
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

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


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
