"""
Request Optimization System for cost reduction.

Implements request deduplication, batching, prioritization,
and intelligent routing to minimize API costs and latency.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from uuid import UUID, uuid4
import heapq

import numpy as np
from pydantic import BaseModel, Field

from src.core.redis_client import get_redis
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RequestPriority(int, Enum):
    """Request priority levels."""
    CRITICAL = 0    # Highest priority
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BATCH = 4       # Lowest priority, can be batched


class RequestStatus(str, Enum):
    """Request processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEDUPLICATED = "deduplicated"
    BATCHED = "batched"


@dataclass
class Request:
    """Request wrapper with metadata."""
    id: str
    key: str                      # Deduplication key
    type: str                     # Request type (e.g., 'llm', 'market_data')
    payload: Dict[str, Any]
    priority: RequestPriority
    created_at: datetime
    callback: Optional[Callable] = None
    timeout: int = 30
    max_retries: int = 3
    batch_eligible: bool = True
    cost_estimate: float = 0.0
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


class RequestMetrics(BaseModel):
    """Request optimization metrics."""
    total_requests: int = 0
    deduplicated: int = 0
    batched: int = 0
    prioritized: int = 0
    total_cost: float = 0.0
    cost_saved: float = 0.0
    avg_latency_ms: float = 0.0


class RequestDeduplicator:
    """Handles request deduplication."""
    
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self.pending_requests: Dict[str, List[Callable]] = {}
        self.completed_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.redis = None
        
    async def initialize(self):
        """Initialize deduplicator."""
        self.redis = await get_redis()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
    
    def generate_key(self, request_type: str, payload: Dict[str, Any]) -> str:
        """Generate deduplication key from request."""
        # Sort payload for consistent hashing
        sorted_payload = json.dumps(payload, sort_keys=True)
        content = f"{request_type}:{sorted_payload}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def check_duplicate(
        self,
        key: str,
        callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Check if request is duplicate.
        
        Returns:
            Tuple of (is_duplicate, cached_result)
        """
        # Check completed cache
        if key in self.completed_cache:
            result, timestamp = self.completed_cache[key]
            if (datetime.utcnow() - timestamp).seconds < self.ttl:
                logger.debug(f"Request {key} deduplicated from cache")
                return True, result
        
        # Check Redis cache
        if self.redis:
            cached = await self.redis.get(f"dedup:{key}")
            if cached:
                result = json.loads(cached)
                logger.debug(f"Request {key} deduplicated from Redis")
                return True, result
        
        # Check pending requests
        if key in self.pending_requests:
            # Add callback to pending list
            if callback:
                self.pending_requests[key].append(callback)
            logger.debug(f"Request {key} deduplicated as pending")
            return True, None
        
        # Not a duplicate
        if callback:
            self.pending_requests[key] = [callback]
        
        return False, None
    
    async def store_result(self, key: str, result: Any):
        """Store result for deduplication."""
        # Store in memory cache
        self.completed_cache[key] = (result, datetime.utcnow())
        
        # Store in Redis
        if self.redis:
            await self.redis.setex(
                f"dedup:{key}",
                self.ttl,
                json.dumps(result)
            )
        
        # Trigger callbacks for pending requests
        if key in self.pending_requests:
            callbacks = self.pending_requests.pop(key)
            for callback in callbacks:
                if callback:
                    await callback(result)
    
    async def _cleanup_task(self):
        """Clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                now = datetime.utcnow()
                expired = []
                
                for key, (_, timestamp) in self.completed_cache.items():
                    if (now - timestamp).seconds > self.ttl:
                        expired.append(key)
                
                for key in expired:
                    del self.completed_cache[key]
                
                if expired:
                    logger.debug(f"Cleaned {len(expired)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Deduplication cleanup error: {e}")


class RequestBatcher:
    """Handles request batching for efficiency."""
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_window_ms: int = 100
    ):
        self.batch_size = batch_size
        self.batch_window_ms = batch_window_ms
        self.batches: Dict[str, List[Request]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self.batch_processors: Dict[str, Callable] = {}
        
    def register_processor(
        self,
        request_type: str,
        processor: Callable
    ):
        """Register batch processor for request type."""
        self.batch_processors[request_type] = processor
    
    async def add_to_batch(self, request: Request) -> bool:
        """
        Add request to batch queue.
        
        Returns:
            True if added to batch, False if should process immediately
        """
        if not request.batch_eligible:
            return False
        
        request_type = request.type
        
        # Add to batch
        self.batches[request_type].append(request)
        
        # Start timer if not already running
        if request_type not in self.batch_timers:
            self.batch_timers[request_type] = asyncio.create_task(
                self._batch_timer(request_type)
            )
        
        # Check if batch is full
        if len(self.batches[request_type]) >= self.batch_size:
            await self._process_batch(request_type)
        
        return True
    
    async def _batch_timer(self, request_type: str):
        """Timer for batch window."""
        try:
            await asyncio.sleep(self.batch_window_ms / 1000)
            await self._process_batch(request_type)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Batch timer error: {e}")
        finally:
            if request_type in self.batch_timers:
                del self.batch_timers[request_type]
    
    async def _process_batch(self, request_type: str):
        """Process accumulated batch."""
        if request_type not in self.batches:
            return
        
        batch = self.batches.pop(request_type, [])
        if not batch:
            return
        
        # Cancel timer
        if request_type in self.batch_timers:
            self.batch_timers[request_type].cancel()
            del self.batch_timers[request_type]
        
        logger.info(f"Processing batch of {len(batch)} {request_type} requests")
        
        # Process batch
        if request_type in self.batch_processors:
            processor = self.batch_processors[request_type]
            try:
                results = await processor(batch)
                
                # Distribute results to callbacks
                for request, result in zip(batch, results):
                    if request.callback:
                        await request.callback(result)
                        
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Process individually as fallback
                for request in batch:
                    if request.callback:
                        await request.callback(None)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            'pending_batches': len(self.batches),
            'batch_sizes': {k: len(v) for k, v in self.batches.items()},
            'active_timers': len(self.batch_timers)
        }


class PriorityQueue:
    """Priority queue for request scheduling."""
    
    def __init__(self):
        self.queue: List[Tuple[int, float, Request]] = []
        self.counter = 0
        
    async def push(self, request: Request):
        """Add request to priority queue."""
        # Use counter to break ties
        heapq.heappush(
            self.queue,
            (request.priority.value, self.counter, request)
        )
        self.counter += 1
    
    async def pop(self) -> Optional[Request]:
        """Get highest priority request."""
        if self.queue:
            _, _, request = heapq.heappop(self.queue)
            return request
        return None
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    def peek(self) -> Optional[Request]:
        """Peek at highest priority request without removing."""
        if self.queue:
            return self.queue[0][2]
        return None


class RequestThrottler:
    """Handles request throttling and rate limiting."""
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, Any]] = {
            'llm': {'rpm': 60, 'tokens_per_min': 90000},
            'market_data': {'rpm': 300, 'requests_per_sec': 10},
            'database': {'rpm': 1000, 'concurrent': 50}
        }
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.token_usage: Dict[str, int] = defaultdict(int)
        
    async def check_rate_limit(self, request_type: str) -> bool:
        """
        Check if request can proceed within rate limits.
        
        Returns:
            True if can proceed, False if should wait
        """
        if request_type not in self.rate_limits:
            return True
        
        limits = self.rate_limits[request_type]
        now = datetime.utcnow()
        
        # Clean old entries
        minute_ago = now - timedelta(minutes=1)
        self.request_counts[request_type] = [
            ts for ts in self.request_counts[request_type]
            if ts > minute_ago
        ]
        
        # Check requests per minute
        if 'rpm' in limits:
            current_rpm = len(self.request_counts[request_type])
            if current_rpm >= limits['rpm']:
                return False
        
        # Check requests per second
        if 'requests_per_sec' in limits:
            second_ago = now - timedelta(seconds=1)
            recent = sum(1 for ts in self.request_counts[request_type] if ts > second_ago)
            if recent >= limits['requests_per_sec']:
                return False
        
        # Record request
        self.request_counts[request_type].append(now)
        return True
    
    async def wait_if_needed(self, request_type: str) -> float:
        """
        Wait if rate limited.
        
        Returns:
            Wait time in seconds
        """
        wait_time = 0
        while not await self.check_rate_limit(request_type):
            wait_time = 0.1
            await asyncio.sleep(wait_time)
        
        return wait_time
    
    def get_current_rates(self) -> Dict[str, Dict[str, Any]]:
        """Get current rate limit usage."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        usage = {}
        for request_type, limits in self.rate_limits.items():
            recent_count = sum(
                1 for ts in self.request_counts[request_type]
                if ts > minute_ago
            )
            
            usage[request_type] = {
                'current_rpm': recent_count,
                'limit_rpm': limits.get('rpm', 'unlimited'),
                'utilization': recent_count / limits['rpm'] if 'rpm' in limits else 0
            }
        
        return usage


class CostAwareRouter:
    """Routes requests based on cost optimization."""
    
    def __init__(self):
        self.provider_costs = {
            'openai': {'cost_per_1k_tokens': 0.03, 'latency_ms': 500},
            'anthropic': {'cost_per_1k_tokens': 0.025, 'latency_ms': 600},
            'openrouter': {'cost_per_1k_tokens': 0.02, 'latency_ms': 700},
            'local': {'cost_per_1k_tokens': 0.001, 'latency_ms': 100}
        }
        self.provider_health: Dict[str, bool] = {p: True for p in self.provider_costs}
        self.provider_usage: Dict[str, float] = defaultdict(float)
        
    async def route_request(
        self,
        request: Request,
        prefer_cost: bool = True
    ) -> str:
        """
        Route request to optimal provider.
        
        Args:
            request: Request to route
            prefer_cost: Prefer cost over latency
            
        Returns:
            Selected provider
        """
        available_providers = [
            p for p, healthy in self.provider_health.items()
            if healthy
        ]
        
        if not available_providers:
            raise RuntimeError("No available providers")
        
        # Calculate scores for each provider
        scores = {}
        for provider in available_providers:
            cost = self.provider_costs[provider]['cost_per_1k_tokens']
            latency = self.provider_costs[provider]['latency_ms']
            
            if prefer_cost:
                # Lower cost is better
                score = 1.0 / (cost + 0.001)
            else:
                # Lower latency is better
                score = 1.0 / (latency + 1)
            
            # Adjust for load balancing
            usage_penalty = self.provider_usage[provider] * 0.1
            scores[provider] = score - usage_penalty
        
        # Select best provider
        best_provider = max(scores, key=scores.get)
        
        # Update usage
        self.provider_usage[best_provider] += 1
        
        # Decay usage over time (simple load balancing)
        for provider in self.provider_usage:
            self.provider_usage[provider] *= 0.99
        
        logger.debug(f"Routed request to {best_provider} (score: {scores[best_provider]:.2f})")
        return best_provider
    
    def update_provider_health(self, provider: str, healthy: bool):
        """Update provider health status."""
        self.provider_health[provider] = healthy
        if not healthy:
            logger.warning(f"Provider {provider} marked as unhealthy")
    
    def estimate_cost(
        self,
        provider: str,
        token_count: int
    ) -> float:
        """Estimate request cost."""
        if provider not in self.provider_costs:
            return 0.0
        
        cost_per_1k = self.provider_costs[provider]['cost_per_1k_tokens']
        return (token_count / 1000) * cost_per_1k


class RequestOptimizer:
    """Main request optimization orchestrator."""
    
    def __init__(self):
        self.deduplicator = RequestDeduplicator()
        self.batcher = RequestBatcher()
        self.priority_queue = PriorityQueue()
        self.throttler = RequestThrottler()
        self.router = CostAwareRouter()
        self.metrics = RequestMetrics()
        self.processing_semaphore = asyncio.Semaphore(10)
        
    async def initialize(self):
        """Initialize request optimizer."""
        await self.deduplicator.initialize()
        
        # Register batch processors
        self.batcher.register_processor('llm', self._process_llm_batch)
        self.batcher.register_processor('market_data', self._process_market_batch)
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
        logger.info("Request optimizer initialized")
    
    async def submit_request(
        self,
        request_type: str,
        payload: Dict[str, Any],
        priority: RequestPriority = RequestPriority.MEDIUM,
        batch_eligible: bool = True,
        callback: Optional[Callable] = None
    ) -> Any:
        """
        Submit request for optimized processing.
        
        Args:
            request_type: Type of request
            payload: Request payload
            priority: Request priority
            batch_eligible: Can be batched
            callback: Async callback for result
            
        Returns:
            Request result or None if async
        """
        # Create request
        request = Request(
            id=str(uuid4()),
            key=self.deduplicator.generate_key(request_type, payload),
            type=request_type,
            payload=payload,
            priority=priority,
            created_at=datetime.utcnow(),
            callback=callback,
            batch_eligible=batch_eligible
        )
        
        self.metrics.total_requests += 1
        
        # Check for duplicate
        is_duplicate, cached_result = await self.deduplicator.check_duplicate(
            request.key,
            callback
        )
        
        if is_duplicate:
            self.metrics.deduplicated += 1
            self.metrics.cost_saved += self.router.estimate_cost('openai', 1000)
            if cached_result is not None:
                return cached_result
            return None  # Will be handled by callback
        
        # Check if can batch
        if batch_eligible and await self.batcher.add_to_batch(request):
            self.metrics.batched += 1
            return None  # Will be handled by batch processor
        
        # Add to priority queue
        await self.priority_queue.push(request)
        
        if priority == RequestPriority.CRITICAL:
            self.metrics.prioritized += 1
        
        # If synchronous (no callback), wait for result
        if callback is None:
            future = asyncio.Future()
            request.callback = lambda result: future.set_result(result)
            return await future
        
        return None
    
    async def _processing_loop(self):
        """Main processing loop for queued requests."""
        while True:
            try:
                # Get next request
                request = await self.priority_queue.pop()
                if request is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process with semaphore for concurrency control
                asyncio.create_task(self._process_request(request))
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    async def _process_request(self, request: Request):
        """Process individual request."""
        async with self.processing_semaphore:
            try:
                start_time = time.time()
                
                # Apply rate limiting
                wait_time = await self.throttler.wait_if_needed(request.type)
                if wait_time > 0:
                    logger.debug(f"Rate limited {request.type} for {wait_time}s")
                
                # Route request
                provider = await self.router.route_request(
                    request,
                    prefer_cost=(request.priority != RequestPriority.CRITICAL)
                )
                
                # Process based on type
                if request.type == 'llm':
                    result = await self._process_llm_request(request, provider)
                elif request.type == 'market_data':
                    result = await self._process_market_request(request, provider)
                else:
                    result = await self._process_generic_request(request)
                
                # Store result for deduplication
                await self.deduplicator.store_result(request.key, result)
                
                # Update metrics
                latency = (time.time() - start_time) * 1000
                self.metrics.avg_latency_ms = (
                    self.metrics.avg_latency_ms * 0.9 + latency * 0.1
                )
                
                # Trigger callback
                if request.callback:
                    await request.callback(result)
                
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                if request.callback:
                    await request.callback(None)
    
    async def _process_llm_batch(
        self,
        batch: List[Request]
    ) -> List[Any]:
        """Process batch of LLM requests."""
        try:
            # Combine prompts
            combined_prompts = []
            for request in batch:
                combined_prompts.append(request.payload.get('prompt', ''))
            
            # Make batched API call
            # This is a placeholder - actual implementation would call LLM API
            logger.info(f"Processing LLM batch of {len(batch)} requests")
            
            # Simulate processing
            await asyncio.sleep(0.5)
            
            # Return mock results
            results = [f"Response for: {p[:50]}" for p in combined_prompts]
            
            # Update metrics
            self.metrics.cost_saved += self.router.estimate_cost('openai', 1000) * (len(batch) - 1)
            
            return results
            
        except Exception as e:
            logger.error(f"LLM batch processing error: {e}")
            return [None] * len(batch)
    
    async def _process_market_batch(
        self,
        batch: List[Request]
    ) -> List[Any]:
        """Process batch of market data requests."""
        try:
            # Combine symbols
            symbols = set()
            for request in batch:
                symbols.add(request.payload.get('symbol', ''))
            
            # Make batched API call
            logger.info(f"Processing market data batch for {len(symbols)} symbols")
            
            # Simulate processing
            await asyncio.sleep(0.2)
            
            # Return mock results
            results = []
            for request in batch:
                symbol = request.payload.get('symbol', '')
                results.append({'symbol': symbol, 'price': np.random.uniform(100, 50000)})
            
            return results
            
        except Exception as e:
            logger.error(f"Market batch processing error: {e}")
            return [None] * len(batch)
    
    async def _process_llm_request(
        self,
        request: Request,
        provider: str
    ) -> Any:
        """Process individual LLM request."""
        # Placeholder for actual LLM processing
        logger.debug(f"Processing LLM request via {provider}")
        await asyncio.sleep(0.5)
        return f"LLM response via {provider}"
    
    async def _process_market_request(
        self,
        request: Request,
        provider: str
    ) -> Any:
        """Process individual market data request."""
        # Placeholder for actual market data processing
        logger.debug(f"Processing market request via {provider}")
        await asyncio.sleep(0.2)
        return {'price': np.random.uniform(100, 50000)}
    
    async def _process_generic_request(
        self,
        request: Request
    ) -> Any:
        """Process generic request."""
        logger.debug(f"Processing generic request")
        await asyncio.sleep(0.1)
        return {'status': 'processed'}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics."""
        return {
            'requests': {
                'total': self.metrics.total_requests,
                'deduplicated': self.metrics.deduplicated,
                'batched': self.metrics.batched,
                'prioritized': self.metrics.prioritized
            },
            'performance': {
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'queue_size': self.priority_queue.size()
            },
            'cost': {
                'total_cost': self.metrics.total_cost,
                'cost_saved': self.metrics.cost_saved,
                'savings_percentage': (
                    self.metrics.cost_saved / (self.metrics.total_cost + self.metrics.cost_saved) * 100
                    if self.metrics.total_cost > 0 else 0
                )
            },
            'batching': self.batcher.get_batch_stats(),
            'rate_limits': self.throttler.get_current_rates()
        }
