"""
Health check endpoints.
"""

from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from src.core.database import get_db
from src.core.redis_client import get_redis
from src.core.kafka_client import get_producer
from src.core.qdrant_client import get_qdrant

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "crypto-signals-api"
    }


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Detailed health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check database
    try:
        result = await db.execute(text("SELECT 1"))
        health_status["components"]["database"] = {
            "status": "healthy",
            "response_time_ms": 0
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis_client = get_redis()
        await redis_client.ping()
        health_status["components"]["redis"] = {
            "status": "healthy"
        }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Kafka
    try:
        producer = get_producer()
        health_status["components"]["kafka"] = {
            "status": "healthy" if producer else "unhealthy"
        }
    except Exception as e:
        health_status["components"]["kafka"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Qdrant
    try:
        qdrant = get_qdrant()
        collections = qdrant.get_collections()
        health_status["components"]["qdrant"] = {
            "status": "healthy",
            "collections": len(collections.collections)
        }
    except Exception as e:
        health_status["components"]["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/ready")
async def readiness_check() -> Dict[str, bool]:
    """Readiness probe for Kubernetes."""
    try:
        # Perform basic checks
        redis_client = get_redis()
        await redis_client.ping()
        
        return {"ready": True}
    except Exception:
        return {"ready": False}


@router.get("/live")
async def liveness_check() -> Dict[str, bool]:
    """Liveness probe for Kubernetes."""
    return {"alive": True}
