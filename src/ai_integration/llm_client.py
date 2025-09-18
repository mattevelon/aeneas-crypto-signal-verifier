"""LLM Client for interacting with GPT-4/Claude APIs with optimization."""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from enum import Enum
import hashlib
import time

from src.config.settings import get_settings
from src.core.redis_client import get_redis

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int]
    latency_ms: float
    cached: bool
    metadata: Dict[str, Any]


class LLMClient:
    """
    Async LLM client with connection pooling, intelligent retry, and batching.
    Supports GPT-4-turbo and Claude-3-opus with fallback switching.
    """
    
    def __init__(self):
        """Initialize LLM client with configuration."""
        self.settings = get_settings()
        self.redis_client = None  # Will use get_redis() when needed
        
        # API configuration
        self.providers = {
            LLMProvider.OPENAI: {
                'api_key': self.settings.OPENAI_API_KEY if hasattr(self.settings, 'OPENAI_API_KEY') else None,
                'base_url': 'https://api.openai.com/v1',
                'models': ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo'],
                'headers': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                }
            },
            LLMProvider.ANTHROPIC: {
                'api_key': self.settings.ANTHROPIC_API_KEY if hasattr(self.settings, 'ANTHROPIC_API_KEY') else None,
                'base_url': 'https://api.anthropic.com/v1',
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
                'headers': lambda key: {
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'Content-Type': 'application/json'
                }
            },
            LLMProvider.OPENROUTER: {
                'api_key': self.settings.openrouter_api_key if hasattr(self.settings, 'openrouter_api_key') else None,
                'base_url': 'https://openrouter.ai/api/v1',
                'models': ['deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3.1:free', 'openai/gpt-4-turbo-preview', 'anthropic/claude-3-opus'],
                'headers': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://aeneas-crypto.com',
                    'X-Title': 'AENEAS Crypto Signal Verifier'
                }
            }
        }
        
        # Request configuration
        self.max_retries = 3
        self.timeout = 30  # seconds
        self.batch_size = 5  # max signals per batch
        self.cache_ttl = 3600  # 1 hour
        
        # Connection pool
        self.connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = None
        
        # Performance tracking
        self.request_history = []
        self.token_usage = {'prompt': 0, 'completion': 0, 'total': 0}
        self.cost_tracking = {'total': 0.0, 'by_model': {}}
        
        logger.info("Initialized LLMClient with multiple providers")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def analyze_signal(self,
                            prompt: Dict[str, Any],
                            provider: Optional[LLMProvider] = None,
                            model: Optional[str] = None) -> LLMResponse:
        """
        Analyze a signal using LLM.
        
        Args:
            prompt: Prompt dictionary with system and user messages
            provider: Optional provider override
            model: Optional model override
            
        Returns:
            LLM response
        """
        # Check cache first
        cache_key = self._generate_cache_key(prompt)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Default to OpenRouter for DeepSeek V3.1
        if provider is None:
            provider = LLMProvider.OPENROUTER
        
        # Default to DeepSeek V3.1 model (free version)
        if model is None:
            model = 'deepseek/deepseek-chat-v3.1:free'  # Using free version of DeepSeek V3.1
        
        # Select provider and model
        if not provider:
            provider = await self._select_best_provider()
        
        if not model:
            model = self.providers[provider]['models'][0]
        
        # Make request with retry logic
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(provider, model, prompt)
                
                # Cache successful response
                await self._cache_response(cache_key, response)
                
                # Track performance
                self._track_performance(provider, model, response, time.time() - start_time)
                
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    
                    # Try fallback model/provider on last attempt
                    if attempt == self.max_retries - 2:
                        provider, model = await self._get_fallback(provider)
                else:
                    raise
        
        raise Exception("All retry attempts failed")
    
    async def batch_analyze(self,
                           prompts: List[Dict[str, Any]],
                           provider: Optional[LLMProvider] = None) -> List[LLMResponse]:
        """
        Analyze multiple signals in batch for efficiency.
        
        Args:
            prompts: List of prompts (up to batch_size)
            provider: Optional provider override
            
        Returns:
            List of LLM responses
        """
        if len(prompts) > self.batch_size:
            prompts = prompts[:self.batch_size]
            logger.warning(f"Batch size limited to {self.batch_size}")
        
        # Process in parallel
        tasks = []
        for prompt in prompts:
            tasks.append(self.analyze_signal(prompt, provider))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch item {i} failed: {response}")
                # Create error response
                processed.append(LLMResponse(
                    content=json.dumps({"error": str(response)}),
                    model="error",
                    provider=provider or LLMProvider.OPENAI,
                    usage={},
                    latency_ms=0,
                    cached=False,
                    metadata={"error": True}
                ))
            else:
                processed.append(response)
        
        return processed
    
    async def _make_request(self,
                           provider: LLMProvider,
                           model: str,
                           prompt: Dict[str, Any]) -> LLMResponse:
        """Make API request to LLM provider."""
        config = self.providers[provider]
        
        if not config['api_key']:
            raise ValueError(f"No API key configured for {provider.value}")
        
        # Build request based on provider
        if provider == LLMProvider.OPENAI or provider == LLMProvider.OPENROUTER:
            request_data = self._build_openai_request(model, prompt)
            endpoint = f"{config['base_url']}/chat/completions"
        elif provider == LLMProvider.ANTHROPIC:
            request_data = self._build_anthropic_request(model, prompt)
            endpoint = f"{config['base_url']}/messages"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Make request
        headers = config['headers'](config['api_key'])
        
        if not self.session:
            self.session = aiohttp.ClientSession(connector=self.connector)
        
        async with self.session.post(
            endpoint,
            headers=headers,
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as response:
            response_data = await response.json()
            
            if response.status != 200:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API error ({response.status}): {error_msg}")
            
            # Parse response based on provider
            if provider == LLMProvider.OPENAI or provider == LLMProvider.OPENROUTER:
                return self._parse_openai_response(response_data, model, provider)
            elif provider == LLMProvider.ANTHROPIC:
                return self._parse_anthropic_response(response_data, model, provider)
    
    def _build_openai_request(self, model: str, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Build OpenAI-compatible request."""
        messages = [
            {"role": "system", "content": prompt.get('system', '')},
            {"role": "user", "content": prompt.get('user', '')}
        ]
        
        return {
            "model": model,
            "messages": messages,
            "max_tokens": prompt.get('max_tokens', 4000),
            "temperature": prompt.get('temperature', 0.3),
            "response_format": {"type": "json_object"} if 'json' in prompt.get('user', '').lower() else None
        }
    
    def _build_anthropic_request(self, model: str, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Build Anthropic request."""
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt.get('user', '')}
            ],
            "system": prompt.get('system', ''),
            "max_tokens": prompt.get('max_tokens', 4000),
            "temperature": prompt.get('temperature', 0.3)
        }
    
    def _parse_openai_response(self, data: Dict[str, Any], model: str, provider: LLMProvider) -> LLMResponse:
        """Parse OpenAI-compatible response."""
        choice = data['choices'][0]
        content = choice['message']['content']
        
        usage = data.get('usage', {})
        
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            usage={
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            },
            latency_ms=0,  # Will be set by caller
            cached=False,
            metadata={
                'finish_reason': choice.get('finish_reason'),
                'id': data.get('id')
            }
        )
    
    def _parse_anthropic_response(self, data: Dict[str, Any], model: str, provider: LLMProvider) -> LLMResponse:
        """Parse Anthropic response."""
        content = data['content'][0]['text']
        
        usage = data.get('usage', {})
        
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            usage={
                'prompt_tokens': usage.get('input_tokens', 0),
                'completion_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            },
            latency_ms=0,
            cached=False,
            metadata={
                'stop_reason': data.get('stop_reason'),
                'id': data.get('id')
            }
        )
    
    async def _select_best_provider(self) -> LLMProvider:
        """Select best provider based on availability and performance."""
        # Check which providers have API keys
        available = []
        for provider, config in self.providers.items():
            if config['api_key']:
                available.append(provider)
        
        if not available:
            raise ValueError("No LLM providers configured with API keys")
        
        # For now, prefer OpenRouter if available (supports multiple models)
        if LLMProvider.OPENROUTER in available:
            return LLMProvider.OPENROUTER
        elif LLMProvider.OPENAI in available:
            return LLMProvider.OPENAI
        else:
            return available[0]
    
    async def _get_fallback(self, current_provider: LLMProvider) -> Tuple[LLMProvider, str]:
        """Get fallback provider and model."""
        # Try next provider
        providers = list(self.providers.keys())
        current_idx = providers.index(current_provider)
        
        for i in range(1, len(providers)):
            next_idx = (current_idx + i) % len(providers)
            next_provider = providers[next_idx]
            
            if self.providers[next_provider]['api_key']:
                return next_provider, self.providers[next_provider]['models'][0]
        
        # If no other provider, try different model from same provider
        models = self.providers[current_provider]['models']
        if len(models) > 1:
            return current_provider, models[1]
        
        return current_provider, models[0]
    
    def _generate_cache_key(self, prompt: Dict[str, Any]) -> str:
        """Generate cache key for prompt."""
        # Create hash of prompt content
        prompt_str = json.dumps(prompt, sort_keys=True)
        hash_obj = hashlib.md5(prompt_str.encode())
        return f"llm_response:{hash_obj.hexdigest()}"
    
    async def _get_cached_response(self, key: str) -> Optional[LLMResponse]:
        """Get cached LLM response."""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return LLMResponse(
                    content=data['content'],
                    model=data['model'],
                    provider=LLMProvider(data['provider']),
                    usage=data['usage'],
                    latency_ms=0,
                    cached=True,
                    metadata=data.get('metadata', {})
                )
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def _cache_response(self, key: str, response: LLMResponse):
        """Cache LLM response."""
        if not self.redis_client:
            return
        
        try:
            cache_data = {
                'content': response.content,
                'model': response.model,
                'provider': response.provider.value,
                'usage': response.usage,
                'metadata': response.metadata
            }
            
            await self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def _track_performance(self, provider: LLMProvider, model: str, response: LLMResponse, elapsed: float):
        """Track performance metrics."""
        response.latency_ms = elapsed * 1000
        
        # Track request
        self.request_history.append({
            'timestamp': datetime.now().isoformat(),
            'provider': provider.value,
            'model': model,
            'latency_ms': response.latency_ms,
            'tokens': response.usage.get('total_tokens', 0),
            'cached': response.cached
        })
        
        # Update token usage
        self.token_usage['prompt'] += response.usage.get('prompt_tokens', 0)
        self.token_usage['completion'] += response.usage.get('completion_tokens', 0)
        self.token_usage['total'] += response.usage.get('total_tokens', 0)
        
        # Estimate cost (rough estimates)
        cost = self._estimate_cost(model, response.usage)
        self.cost_tracking['total'] += cost
        
        if model not in self.cost_tracking['by_model']:
            self.cost_tracking['by_model'][model] = 0
        self.cost_tracking['by_model'][model] += cost
        
        # Keep history limited
        if len(self.request_history) > 100:
            self.request_history.pop(0)
    
    def _estimate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Estimate API cost based on token usage."""
        # Rough cost estimates per 1K tokens
        cost_per_1k = {
            'gpt-4-turbo-preview': {'prompt': 0.01, 'completion': 0.03},
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},
            'claude-3-opus-20240229': {'prompt': 0.015, 'completion': 0.075},
            'claude-3-sonnet-20240229': {'prompt': 0.003, 'completion': 0.015}
        }
        
        if model not in cost_per_1k:
            return 0
        
        rates = cost_per_1k[model]
        prompt_cost = (usage.get('prompt_tokens', 0) / 1000) * rates['prompt']
        completion_cost = (usage.get('completion_tokens', 0) / 1000) * rates['completion']
        
        return prompt_cost + completion_cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_latency = 0
        if self.request_history:
            latencies = [r['latency_ms'] for r in self.request_history if not r['cached']]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
        
        return {
            'total_requests': len(self.request_history),
            'token_usage': self.token_usage,
            'cost_tracking': self.cost_tracking,
            'avg_latency_ms': round(avg_latency, 2),
            'cache_hit_rate': sum(1 for r in self.request_history if r['cached']) / len(self.request_history) if self.request_history else 0,
            'providers_available': [p.value for p, c in self.providers.items() if c['api_key']]
        }
    
    async def stream_response(self,
                             prompt: Dict[str, Any],
                             callback: Any,
                             provider: Optional[LLMProvider] = None) -> None:
        """
        Stream response from LLM for real-time updates.
        
        Args:
            prompt: Prompt dictionary
            callback: Async callback function for chunks
            provider: Optional provider override
        """
        # This would implement streaming for providers that support it
        # For now, just get full response and send
        response = await self.analyze_signal(prompt, provider)
        await callback(response.content)
