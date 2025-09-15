"""
Redis client for caching and session management.
"""

from typing import Optional, Any, Dict
import json
import pickle

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
import structlog

from src.config.settings import settings

logger = structlog.get_logger()

# Global Redis client
redis_client: Optional[redis.Redis] = None
connection_pool: Optional[ConnectionPool] = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client, connection_pool
    
    try:
        connection_pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=False
        )
        
        redis_client = redis.Redis(connection_pool=connection_pool)
        
        # Test connection
        await redis_client.ping()
        logger.info("Redis connection established")
        
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise


async def close_redis():
    """Close Redis connection."""
    global redis_client, connection_pool
    
    if redis_client:
        await redis_client.close()
        redis_client = None
    
    if connection_pool:
        await connection_pool.disconnect()
        connection_pool = None
    
    logger.info("Redis connection closed")


def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    return redis_client


class RedisCache:
    """Redis cache operations."""
    
    def __init__(self, prefix: str = "cache"):
        self.prefix = prefix
        self.ttl = settings.redis_ttl
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if not redis_client:
                return None  # Gracefully handle when Redis is not available
            client = get_redis()
            full_key = self._make_key(key)
            value = await client.get(full_key)
            
            if value:
                # Try JSON first, then pickle
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return pickle.loads(value)
            
            return None
            
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            if not redis_client:
                return False  # Gracefully handle when Redis is not available
            client = get_redis()
            full_key = self._make_key(key)
            ttl = ttl or self.ttl
            
            # Try JSON first, then pickle
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)
            
            await client.setex(full_key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if not redis_client:
                return False  # Gracefully handle when Redis is not available
            client = get_redis()
            full_key = self._make_key(key)
            result = await client.delete(full_key)
            return bool(result)
            
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if not redis_client:
                return False  # Gracefully handle when Redis is not available
            client = get_redis()
            full_key = self._make_key(key)
            return bool(await client.exists(full_key))
            
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False
    
    async def get_many(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            if not redis_client:
                return {}  # Gracefully handle when Redis is not available
            client = get_redis()
            full_keys = [self._make_key(k) for k in keys]
            values = await client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = pickle.loads(value)
            
            return result
            
        except Exception as e:
            logger.error("Cache get_many error", error=str(e))
            return {}
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache."""
        try:
            if not redis_client:
                return None  # Gracefully handle when Redis is not available
            client = get_redis()
            full_key = self._make_key(key)
            return await client.incrby(full_key, amount)
            
        except Exception as e:
            logger.error("Cache increment error", key=key, error=str(e))
            return None


# Create cache instances for different purposes
signal_cache = RedisCache("signals")
market_cache = RedisCache("market")
session_cache = RedisCache("sessions")
