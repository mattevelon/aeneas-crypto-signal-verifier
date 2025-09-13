"""
Market data integration for real-time price feeds.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio

import httpx
import structlog

from src.config.settings import settings
from src.core.redis_client import market_cache

logger = structlog.get_logger()


class MarketDataProvider:
    """Unified market data provider."""
    
    def __init__(self):
        self.binance_client = BinanceClient()
        self.cache_ttl = 60  # 1 minute cache
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # Check cache first
        cache_key = f"price:{symbol}"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached
        
        # Get from Binance
        price = await self.binance_client.get_price(symbol)
        
        if price:
            await market_cache.set(cache_key, price, ttl=self.cache_ttl)
        
        return price
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for a symbol."""
        cache_key = f"orderbook:{symbol}:{depth}"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached
        
        orderbook = await self.binance_client.get_orderbook(symbol, depth)
        
        if orderbook:
            await market_cache.set(cache_key, orderbook, ttl=30)
        
        return orderbook
    
    async def get_24h_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24-hour statistics for a symbol."""
        cache_key = f"stats24h:{symbol}"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached
        
        stats = await self.binance_client.get_24h_stats(symbol)
        
        if stats:
            await market_cache.set(cache_key, stats, ttl=300)  # 5 minutes
        
        return stats
    
    async def validate_price(
        self, 
        symbol: str, 
        price: float, 
        tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """Validate if a price is within market range."""
        current_price = await self.get_price(symbol)
        
        if not current_price:
            return {
                "valid": False,
                "reason": "Unable to fetch market price"
            }
        
        deviation = abs((price - current_price) / current_price)
        
        return {
            "valid": deviation <= tolerance,
            "current_price": current_price,
            "deviation_percentage": deviation * 100,
            "reason": f"Price deviation: {deviation*100:.2f}%"
        }


class BinanceClient:
    """Binance API client."""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Binance API."""
        # Convert BTC/USDT to BTCUSDT
        return symbol.replace("/", "").replace("-", "").upper()
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance."""
        try:
            formatted_symbol = self._format_symbol(symbol)
            response = await self.client.get(
                f"{self.base_url}/api/v3/ticker/price",
                params={"symbol": formatted_symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data["price"])
            
        except Exception as e:
            logger.error(f"Failed to get Binance price: {e}")
        
        return None
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book from Binance."""
        try:
            formatted_symbol = self._format_symbol(symbol)
            response = await self.client.get(
                f"{self.base_url}/api/v3/depth",
                params={"symbol": formatted_symbol, "limit": depth}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "bids": [[float(p), float(q)] for p, q in data["bids"]],
                    "asks": [[float(p), float(q)] for p, q in data["asks"]],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to get Binance orderbook: {e}")
        
        return None
    
    async def get_24h_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24-hour statistics from Binance."""
        try:
            formatted_symbol = self._format_symbol(symbol)
            response = await self.client.get(
                f"{self.base_url}/api/v3/ticker/24hr",
                params={"symbol": formatted_symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "high": float(data["highPrice"]),
                    "low": float(data["lowPrice"]),
                    "volume": float(data["volume"]),
                    "quote_volume": float(data["quoteVolume"]),
                    "price_change": float(data["priceChange"]),
                    "price_change_percent": float(data["priceChangePercent"]),
                    "weighted_avg_price": float(data["weightedAvgPrice"]),
                    "last_price": float(data["lastPrice"]),
                    "open_price": float(data["openPrice"]),
                    "close_price": float(data["prevClosePrice"])
                }
            
        except Exception as e:
            logger.error(f"Failed to get Binance 24h stats: {e}")
        
        return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global market data provider
market_data = MarketDataProvider()
