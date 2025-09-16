"""Cache helper module for backward compatibility and convenience."""

from src.core.redis_client import get_redis, RedisCache

def get_redis_client():
    """Get Redis client instance for backward compatibility."""
    return get_redis()

# Re-export cache instances
from src.core.redis_client import signal_cache, market_cache, session_cache

__all__ = [
    'get_redis_client',
    'get_redis',
    'RedisCache',
    'signal_cache',
    'market_cache',
    'session_cache'
]
