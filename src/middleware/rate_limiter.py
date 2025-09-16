"""
Rate limiting middleware for API endpoints.
"""

import time
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from src.core.redis_client import get_redis
from src.config.settings import get_settings

logger = structlog.get_logger()


class RateLimiter:
    """Token bucket rate limiter using Redis."""
    
    def __init__(self):
        self.redis = get_redis()
        self.settings = get_settings()
    
    async def check_rate_limit(
        self,
        identifier: str,
        requests_per_minute: int = 60,
        burst_size: int = 100
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, metadata) where metadata contains:
            - remaining: number of requests remaining
            - reset: timestamp when the limit resets
        """
        key = f"rate_limit:{identifier}"
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        await self.redis.zremrangebyscore(key, 0, minute_ago)
        
        # Count recent requests
        current_count = await self.redis.zcard(key)
        
        if current_count < requests_per_minute:
            # Add current request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, 60)
            
            return True, {
                "remaining": requests_per_minute - current_count - 1,
                "reset": int(now + 60)
            }
        else:
            # Check burst allowance
            burst_key = f"burst:{identifier}"
            burst_count = await self.redis.get(burst_key)
            
            if burst_count is None:
                # Initialize burst counter
                await self.redis.setex(burst_key, 60, 1)
                await self.redis.zadd(key, {str(now): now})
                return True, {
                    "remaining": burst_size - 1,
                    "reset": int(now + 60)
                }
            elif int(burst_count) < burst_size:
                # Use burst allowance
                await self.redis.incr(burst_key)
                await self.redis.zadd(key, {str(now): now})
                return True, {
                    "remaining": burst_size - int(burst_count) - 1,
                    "reset": int(now + 60)
                }
            else:
                # Rate limit exceeded
                return False, {
                    "remaining": 0,
                    "reset": int(now + 60)
                }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits on API endpoints."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 100):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
    
    def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting."""
        # Try to get authenticated user
        user = getattr(request.state, "user", None)
        if user:
            return f"user:{user.get('id', 'unknown')}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _should_rate_limit(self, path: str) -> bool:
        """Check if path should be rate limited."""
        # Don't rate limit health checks or docs
        exempt_paths = [
            "/api/v1/health",
            "/api/v1/health/detailed",
            "/api/v1/ready",
            "/api/v1/live",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        return not any(path.startswith(p) for p in exempt_paths)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Check if this path should be rate limited
        if not self._should_rate_limit(request.url.path):
            return await call_next(request)
        
        # Get identifier
        identifier = self._get_identifier(request)
        
        # Check rate limit
        allowed, metadata = await self.rate_limiter.check_rate_limit(
            identifier,
            self.requests_per_minute,
            self.burst_size
        )
        
        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                path=request.url.path
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": metadata["reset"]
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(metadata["remaining"]),
                    "X-RateLimit-Reset": str(metadata["reset"]),
                    "Retry-After": str(metadata["reset"] - int(time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        response.headers["X-RateLimit-Reset"] = str(metadata["reset"])
        
        return response


class EndpointRateLimiter:
    """Decorator for endpoint-specific rate limiting."""
    
    def __init__(self, requests_per_minute: int = 10, burst_size: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.rate_limiter = RateLimiter()
    
    async def __call__(self, request: Request):
        """Check rate limit for specific endpoint."""
        # Create unique key for this endpoint
        endpoint_key = hashlib.md5(
            f"{request.method}:{request.url.path}".encode()
        ).hexdigest()
        
        # Get identifier
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        identifier = f"{ip}:{endpoint_key}"
        
        # Check rate limit
        allowed, metadata = await self.rate_limiter.check_rate_limit(
            identifier,
            self.requests_per_minute,
            self.burst_size
        )
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for this endpoint",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(metadata["remaining"]),
                    "X-RateLimit-Reset": str(metadata["reset"]),
                    "Retry-After": str(metadata["reset"] - int(time.time()))
                }
            )
        
        return True
