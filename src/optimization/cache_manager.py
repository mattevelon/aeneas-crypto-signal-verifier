"""
Intelligent Multi-tier Caching Layer for cost optimization.

Implements a sophisticated caching strategy with multiple tiers,
predictive warming, and intelligent invalidation policies.
"""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import OrderedDict
import heapq

import numpy as np
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.redis_client import get_redis
from src.core.database import get_async_session
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_REDIS = "l2_redis"         # Redis cache (fast)
    L3_DATABASE = "l3_database"   # Database cache (persistent)


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    FIFO = "fifo"                 # First In First Out
    TTL = "ttl"                   # Time To Live based
    ADAPTIVE = "adaptive"         # Adaptive based on usage patterns


class CacheEntry(BaseModel):
    """Cache entry metadata."""
    key: str
    value: Any
    tier: CacheTier
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl: Optional[int] = None
    tags: List[str] = []
    cost_score: float = 0.0  # Cost of cache miss
    priority: int = 0         # Priority level


class CacheMetrics(BaseModel):
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_latency_ms: float = 0.0
    cost_savings: float = 0.0


class InMemoryCache:
    """L1 in-memory cache implementation."""
    
    def __init__(self, max_size_mb: int = 100, policy: CachePolicy = CachePolicy.LRU):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.cache: OrderedDict = OrderedDict()
        self.metadata: Dict[str, Dict] = {}
        self.current_size = 0
        self.access_counts: Dict[str, int] = {}
        self.metrics = CacheMetrics()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.metrics.hits += 1
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # Update position for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            # Update last accessed
            if key in self.metadata:
                self.metadata[key]['last_accessed'] = datetime.utcnow()
            
            return self.cache[key]
        
        self.metrics.misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: List[str] = None
    ) -> bool:
        """Set value in cache."""
        try:
            # Calculate size
            serialized = pickle.dumps(value)
            size = len(serialized)
            
            # Check if we need to evict
            while self.current_size + size > self.max_size_bytes:
                if not await self._evict():
                    return False
            
            # Store value
            self.cache[key] = value
            self.current_size += size
            
            # Store metadata
            self.metadata[key] = {
                'size': size,
                'created_at': datetime.utcnow(),
                'last_accessed': datetime.utcnow(),
                'ttl': ttl,
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
                'tags': tags or []
            }
            
            self.access_counts[key] = 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            size = self.metadata[key]['size']
            del self.cache[key]
            del self.metadata[key]
            self.current_size -= size
            if key in self.access_counts:
                del self.access_counts[key]
            return True
        return False
    
    async def _evict(self) -> bool:
        """Evict entry based on policy."""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            key = next(iter(self.cache))
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.access_counts, key=self.access_counts.get)
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest
            key = next(iter(self.cache))
        else:
            # Default to LRU
            key = next(iter(self.cache))
        
        await self.delete(key)
        self.metrics.evictions += 1
        return True
    
    async def clear_expired(self):
        """Clear expired entries."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, meta in self.metadata.items():
            if meta.get('expires_at') and meta['expires_at'] < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.metrics.hits / (self.metrics.hits + self.metrics.misses) if (self.metrics.hits + self.metrics.misses) > 0 else 0
        
        return {
            'entries': len(self.cache),
            'size_mb': self.current_size / (1024 * 1024),
            'hit_rate': hit_rate,
            'total_hits': self.metrics.hits,
            'total_misses': self.metrics.misses,
            'evictions': self.metrics.evictions
        }


class MultiTierCache:
    """Multi-tier intelligent caching system."""
    
    def __init__(self):
        self.l1_cache = InMemoryCache(max_size_mb=100, policy=CachePolicy.LRU)
        self.redis_client = None
        self.cache_config = {
            CacheTier.L1_MEMORY: {'ttl': 300, 'max_size_mb': 100},
            CacheTier.L2_REDIS: {'ttl': 3600, 'max_size_mb': 512},
            CacheTier.L3_DATABASE: {'ttl': 86400, 'max_size_mb': 10240}
        }
        self.warming_queue: List[str] = []
        self.invalidation_rules: Dict[str, Callable] = {}
        self.cost_calculator = CostCalculator()
        self.predictive_warmer = PredictiveWarmer()
        
    async def initialize(self):
        """Initialize cache system."""
        self.redis_client = await get_redis()
        
        # Start background tasks
        asyncio.create_task(self._ttl_cleanup_task())
        asyncio.create_task(self._warming_task())
        asyncio.create_task(self._metrics_collection_task())
        
        logger.info("Multi-tier cache initialized")
    
    async def get(
        self,
        key: str,
        tier: Optional[CacheTier] = None
    ) -> Tuple[Optional[Any], CacheTier]:
        """
        Get value from cache, checking tiers in order.
        
        Args:
            key: Cache key
            tier: Specific tier to check (optional)
            
        Returns:
            Tuple of (value, tier_found)
        """
        start_time = datetime.utcnow()
        
        # Check L1 (memory)
        if not tier or tier == CacheTier.L1_MEMORY:
            value = await self.l1_cache.get(key)
            if value is not None:
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self._record_access(key, CacheTier.L1_MEMORY, True, latency)
                return value, CacheTier.L1_MEMORY
        
        # Check L2 (Redis)
        if not tier or tier == CacheTier.L2_REDIS:
            value = await self._get_from_redis(key)
            if value is not None:
                # Promote to L1
                await self.l1_cache.set(key, value, ttl=300)
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self._record_access(key, CacheTier.L2_REDIS, True, latency)
                return value, CacheTier.L2_REDIS
        
        # Check L3 (Database)
        if not tier or tier == CacheTier.L3_DATABASE:
            value = await self._get_from_database(key)
            if value is not None:
                # Promote to L2 and L1
                await self._set_in_redis(key, value, ttl=3600)
                await self.l1_cache.set(key, value, ttl=300)
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                await self._record_access(key, CacheTier.L3_DATABASE, True, latency)
                return value, CacheTier.L3_DATABASE
        
        # Cache miss
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        await self._record_access(key, None, False, latency)
        return None, None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tiers: Optional[List[CacheTier]] = None,
        tags: Optional[List[str]] = None,
        cost_score: float = 0.0
    ) -> bool:
        """
        Set value in cache tiers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tiers: Specific tiers to set (default: all)
            tags: Tags for grouping/invalidation
            cost_score: Cost of cache miss (for prioritization)
            
        Returns:
            Success status
        """
        try:
            if not tiers:
                # Determine tiers based on cost score
                tiers = self._determine_tiers(cost_score)
            
            success = True
            
            # Set in L1
            if CacheTier.L1_MEMORY in tiers:
                l1_ttl = ttl or self.cache_config[CacheTier.L1_MEMORY]['ttl']
                success &= await self.l1_cache.set(key, value, l1_ttl, tags)
            
            # Set in L2
            if CacheTier.L2_REDIS in tiers:
                l2_ttl = ttl or self.cache_config[CacheTier.L2_REDIS]['ttl']
                success &= await self._set_in_redis(key, value, l2_ttl, tags)
            
            # Set in L3
            if CacheTier.L3_DATABASE in tiers:
                l3_ttl = ttl or self.cache_config[CacheTier.L3_DATABASE]['ttl']
                success &= await self._set_in_database(key, value, l3_ttl, tags)
            
            # Track for predictive warming
            await self.predictive_warmer.track_set(key, cost_score)
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting multi-tier cache: {e}")
            return False
    
    async def invalidate(
        self,
        key: Optional[str] = None,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            pattern: Pattern to match keys
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Invalidate specific key
        if key:
            await self.l1_cache.delete(key)
            await self._delete_from_redis(key)
            await self._delete_from_database(key)
            count = 1
        
        # Invalidate by pattern
        if pattern:
            # L1 invalidation
            for cache_key in list(self.l1_cache.cache.keys()):
                if self._match_pattern(cache_key, pattern):
                    await self.l1_cache.delete(cache_key)
                    count += 1
            
            # L2 invalidation
            count += await self._invalidate_redis_pattern(pattern)
        
        # Invalidate by tags
        if tags:
            count += await self._invalidate_by_tags(tags)
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    async def warm(self, keys: List[str], priority: int = 0):
        """
        Warm cache with specific keys.
        
        Args:
            keys: Keys to warm
            priority: Priority for warming queue
        """
        for key in keys:
            heapq.heappush(self.warming_queue, (priority, key))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = await self._get_redis_stats()
        
        return {
            'l1': l1_stats,
            'l2': l2_stats,
            'total_cost_savings': self.cost_calculator.total_savings,
            'warming_queue_size': len(self.warming_queue),
            'invalidation_rules': len(self.invalidation_rules)
        }
    
    def _determine_tiers(self, cost_score: float) -> List[CacheTier]:
        """Determine which tiers to use based on cost score."""
        if cost_score > 10.0:
            # High cost - use all tiers
            return [CacheTier.L1_MEMORY, CacheTier.L2_REDIS, CacheTier.L3_DATABASE]
        elif cost_score > 5.0:
            # Medium cost - use L1 and L2
            return [CacheTier.L1_MEMORY, CacheTier.L2_REDIS]
        else:
            # Low cost - use L1 only
            return [CacheTier.L1_MEMORY]
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            if not self.redis_client:
                return None
            
            data = await self.redis_client.get(f"cache:{key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: int,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in Redis."""
        try:
            if not self.redis_client:
                return False
            
            serialized = pickle.dumps(value)
            await self.redis_client.setex(f"cache:{key}", ttl, serialized)
            
            # Store tags
            if tags:
                for tag in tags:
                    await self.redis_client.sadd(f"tag:{tag}", key)
                    await self.redis_client.expire(f"tag:{tag}", ttl)
            
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def _delete_from_redis(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            if not self.redis_client:
                return False
            
            await self.redis_client.delete(f"cache:{key}")
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def _get_from_database(self, key: str) -> Optional[Any]:
        """Get value from database cache table."""
        try:
            async with get_async_session() as session:
                # Assuming a cache table exists
                query = "SELECT value FROM cache WHERE key = :key AND expires_at > NOW()"
                result = await session.execute(query, {"key": key})
                row = result.fetchone()
                
                if row:
                    return pickle.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Database get error: {e}")
            return None
    
    async def _set_in_database(
        self,
        key: str,
        value: Any,
        ttl: int,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in database cache table."""
        try:
            async with get_async_session() as session:
                serialized = pickle.dumps(value)
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                # Upsert cache entry
                query = """
                    INSERT INTO cache (key, value, expires_at, tags)
                    VALUES (:key, :value, :expires_at, :tags)
                    ON CONFLICT (key) DO UPDATE
                    SET value = :value, expires_at = :expires_at, tags = :tags
                """
                
                await session.execute(query, {
                    "key": key,
                    "value": serialized,
                    "expires_at": expires_at,
                    "tags": json.dumps(tags) if tags else None
                })
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Database set error: {e}")
            return False
    
    async def _delete_from_database(self, key: str) -> bool:
        """Delete value from database cache table."""
        try:
            async with get_async_session() as session:
                query = "DELETE FROM cache WHERE key = :key"
                await session.execute(query, {"key": key})
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Database delete error: {e}")
            return False
    
    async def _invalidate_redis_pattern(self, pattern: str) -> int:
        """Invalidate Redis keys by pattern."""
        try:
            if not self.redis_client:
                return 0
            
            count = 0
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=f"cache:{pattern}",
                    count=100
                )
                
                if keys:
                    await self.redis_client.delete(*keys)
                    count += len(keys)
                
                if cursor == 0:
                    break
            
            return count
        except Exception as e:
            logger.error(f"Redis pattern invalidation error: {e}")
            return 0
    
    async def _invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        try:
            count = 0
            keys_to_invalidate = set()
            
            # Get keys from Redis tags
            if self.redis_client:
                for tag in tags:
                    members = await self.redis_client.smembers(f"tag:{tag}")
                    keys_to_invalidate.update(members)
            
            # Invalidate all collected keys
            for key in keys_to_invalidate:
                await self.invalidate(key=key)
                count += 1
            
            return count
        except Exception as e:
            logger.error(f"Tag invalidation error: {e}")
            return 0
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def _record_access(
        self,
        key: str,
        tier: Optional[CacheTier],
        hit: bool,
        latency_ms: float
    ):
        """Record cache access for metrics."""
        # Update metrics
        if hit and tier:
            cost_saved = self.cost_calculator.calculate_savings(tier, latency_ms)
            self.cost_calculator.total_savings += cost_saved
        
        # Track for predictive warming
        await self.predictive_warmer.track_access(key, hit)
    
    async def _get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            if not self.redis_client:
                return {}
            
            info = await self.redis_client.info()
            return {
                'memory_used_mb': info.get('used_memory', 0) / (1024 * 1024),
                'keys': info.get('db0', {}).get('keys', 0),
                'hit_rate': info.get('keyspace_hit_rate', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}
    
    async def _ttl_cleanup_task(self):
        """Background task to clean expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.l1_cache.clear_expired()
                
                # Clean database cache
                async with get_async_session() as session:
                    query = "DELETE FROM cache WHERE expires_at < NOW()"
                    await session.execute(query)
                    await session.commit()
                    
            except Exception as e:
                logger.error(f"TTL cleanup error: {e}")
    
    async def _warming_task(self):
        """Background task for cache warming."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if self.warming_queue:
                    # Process highest priority items
                    batch_size = min(10, len(self.warming_queue))
                    keys_to_warm = []
                    
                    for _ in range(batch_size):
                        if self.warming_queue:
                            priority, key = heapq.heappop(self.warming_queue)
                            keys_to_warm.append(key)
                    
                    # Warm keys (fetch from source and cache)
                    for key in keys_to_warm:
                        # This would fetch from the actual data source
                        # For now, just log
                        logger.info(f"Warming cache key: {key}")
                        
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    async def _metrics_collection_task(self):
        """Background task to collect and report metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                stats = await self.get_stats()
                logger.info(f"Cache stats: {stats}")
                
                # Could send to monitoring system here
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")


class CostCalculator:
    """Calculate cost savings from cache hits."""
    
    def __init__(self):
        self.total_savings = 0.0
        self.tier_costs = {
            CacheTier.L1_MEMORY: 0.001,    # Cost per ms
            CacheTier.L2_REDIS: 0.01,      # Cost per ms
            CacheTier.L3_DATABASE: 0.1,    # Cost per ms
            None: 1.0                       # Cost of miss (API call)
        }
    
    def calculate_savings(
        self,
        tier: CacheTier,
        latency_ms: float
    ) -> float:
        """
        Calculate cost savings from cache hit.
        
        Args:
            tier: Cache tier that served the request
            latency_ms: Latency in milliseconds
            
        Returns:
            Cost savings
        """
        miss_cost = self.tier_costs[None] * 100  # Assume 100ms for miss
        hit_cost = self.tier_costs[tier] * latency_ms
        savings = miss_cost - hit_cost
        return max(0, savings)


class PredictiveWarmer:
    """Predictive cache warming based on access patterns."""
    
    def __init__(self):
        self.access_history: Dict[str, List[datetime]] = {}
        self.predictions: Dict[str, float] = {}
        
    async def track_access(self, key: str, hit: bool):
        """Track cache access for prediction."""
        if key not in self.access_history:
            self.access_history[key] = []
        
        self.access_history[key].append(datetime.utcnow())
        
        # Keep last 100 accesses
        if len(self.access_history[key]) > 100:
            self.access_history[key] = self.access_history[key][-100:]
        
        # Update predictions
        await self._update_predictions(key)
    
    async def track_set(self, key: str, cost_score: float):
        """Track when a key is set."""
        self.predictions[key] = cost_score
    
    async def _update_predictions(self, key: str):
        """Update access predictions for a key."""
        if len(self.access_history[key]) < 3:
            return
        
        # Calculate access frequency
        accesses = self.access_history[key]
        time_diffs = [
            (accesses[i] - accesses[i-1]).total_seconds()
            for i in range(1, len(accesses))
        ]
        
        if time_diffs:
            avg_interval = np.mean(time_diffs)
            
            # Predict next access time
            last_access = accesses[-1]
            predicted_next = last_access + timedelta(seconds=avg_interval)
            
            # If predicted soon, increase priority
            time_until = (predicted_next - datetime.utcnow()).total_seconds()
            if 0 < time_until < 300:  # Within 5 minutes
                self.predictions[key] = 10.0  # High priority
    
    def get_warming_candidates(self, limit: int = 10) -> List[str]:
        """Get top candidates for cache warming."""
        sorted_predictions = sorted(
            self.predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [key for key, _ in sorted_predictions[:limit]]
