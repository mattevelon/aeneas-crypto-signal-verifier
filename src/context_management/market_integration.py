"""Market Data Integration for real-time price feeds and market analysis."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import aiohttp
import numpy as np
from decimal import Decimal

from src.config.settings import get_settings
from src.core.redis_client import get_redis

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Real-time market data for a trading pair."""
    pair: str
    current_price: float
    bid: float
    ask: float
    spread: float
    spread_percentage: float
    volume_24h: float
    volume_quote_24h: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    order_book_depth: Dict[str, Any]
    liquidity_score: float
    volatility: float
    correlation_data: Dict[str, float]
    timestamp: datetime


class MarketDataIntegrator:
    """
    Integrates real-time market data from multiple sources.
    Implements order book analysis, volume profiling, and correlation analysis.
    """
    
    def __init__(self):
        """Initialize market data integrator with exchange connections."""
        self.settings = get_settings()
        self.redis_client = None  # Will use get_redis() when needed
        self.cache_ttl = 30  # 30 seconds for market data
        
        # Exchange API endpoints
        self.exchanges = {
            'binance': {
                'base_url': 'https://api.binance.com/api/v3',
                'ticker': '/ticker/24hr',
                'order_book': '/depth',
                'klines': '/klines'
            },
            'kucoin': {
                'base_url': 'https://api.kucoin.com/api/v1',
                'ticker': '/market/stats',
                'order_book': '/market/orderbook/level2_100',
                'klines': '/market/candles'
            }
        }
        
        # Major pairs for correlation analysis
        self.major_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        logger.info("Initialized MarketDataIntegrator")
    
    async def get_market_data(self, pair: str) -> MarketData:
        """
        Get comprehensive market data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Complete market data snapshot
        """
        # Check cache first
        cache_key = f"market_data:{pair}"
        cached = await self._get_cached_data(cache_key)
        if cached:
            return cached
        
        # Fetch fresh data
        tasks = [
            self._fetch_ticker_data(pair),
            self._fetch_order_book(pair),
            self._calculate_volume_profile(pair),
            self._calculate_volatility(pair),
            self._get_correlation_data(pair)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        ticker_data = results[0] if not isinstance(results[0], Exception) else {}
        order_book = results[1] if not isinstance(results[1], Exception) else {}
        volume_profile = results[2] if not isinstance(results[2], Exception) else {}
        volatility = results[3] if not isinstance(results[3], Exception) else 0
        correlations = results[4] if not isinstance(results[4], Exception) else {}
        
        # Build market data
        market_data = self._build_market_data(
            pair, ticker_data, order_book, volume_profile, volatility, correlations
        )
        
        # Cache the data
        await self._cache_data(cache_key, market_data, self.cache_ttl)
        
        return market_data
    
    async def _fetch_ticker_data(self, pair: str) -> Dict[str, Any]:
        """Fetch 24hr ticker data from exchange."""
        # Convert pair format for Binance
        symbol = pair.replace('/', '')
        
        url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['ticker']}"
        params = {'symbol': symbol}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'current_price': float(data.get('lastPrice', 0)),
                            'bid': float(data.get('bidPrice', 0)),
                            'ask': float(data.get('askPrice', 0)),
                            'volume_24h': float(data.get('volume', 0)),
                            'volume_quote_24h': float(data.get('quoteVolume', 0)),
                            'high_24h': float(data.get('highPrice', 0)),
                            'low_24h': float(data.get('lowPrice', 0)),
                            'price_change_24h': float(data.get('priceChange', 0)),
                            'price_change_percentage_24h': float(data.get('priceChangePercent', 0))
                        }
        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
        
        return {}
    
    async def _fetch_order_book(self, pair: str) -> Dict[str, Any]:
        """Fetch order book depth data."""
        symbol = pair.replace('/', '')
        
        url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['order_book']}"
        params = {'symbol': symbol, 'limit': 100}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        bids = [(float(price), float(qty)) for price, qty in data.get('bids', [])]
                        asks = [(float(price), float(qty)) for price, qty in data.get('asks', [])]
                        
                        return self._analyze_order_book(bids, asks)
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
        
        return {}
    
    def _analyze_order_book(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze order book for depth and liquidity metrics."""
        if not bids or not asks:
            return {}
        
        # Calculate bid/ask volumes
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        
        # Calculate weighted average prices
        bid_weighted = sum(price * qty for price, qty in bids) / bid_volume if bid_volume > 0 else 0
        ask_weighted = sum(price * qty for price, qty in asks) / ask_volume if ask_volume > 0 else 0
        
        # Calculate depth at different price levels
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        
        depth_levels = {}
        for level in [0.1, 0.5, 1.0, 2.0]:  # Percentage levels
            bid_depth = sum(qty for price, qty in bids if price >= best_bid * (1 - level/100))
            ask_depth = sum(qty for price, qty in asks if price <= best_ask * (1 + level/100))
            depth_levels[f"{level}%"] = {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth
            }
        
        # Calculate order book imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_weighted_avg': bid_weighted,
            'ask_weighted_avg': ask_weighted,
            'spread': best_ask - best_bid if best_ask and best_bid else 0,
            'spread_percentage': ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0,
            'depth_levels': depth_levels,
            'order_book_imbalance': imbalance,
            'liquidity_score': self._calculate_liquidity_score(depth_levels, imbalance)
        }
    
    def _calculate_liquidity_score(self, depth_levels: Dict[str, Any], imbalance: float) -> float:
        """Calculate liquidity score based on order book metrics."""
        score = 50.0  # Base score
        
        # Adjust for depth (higher depth = higher liquidity)
        if '0.5%' in depth_levels:
            total_depth = depth_levels['0.5%']['total_depth']
            if total_depth > 1000000:  # High liquidity threshold (in quote currency)
                score += 30
            elif total_depth > 100000:
                score += 20
            elif total_depth > 10000:
                score += 10
        
        # Adjust for balance (balanced book = higher liquidity)
        balance_score = (1 - abs(imbalance)) * 20
        score += balance_score
        
        return min(100, max(0, score))
    
    async def _calculate_volume_profile(self, pair: str) -> Dict[str, Any]:
        """Calculate volume profile and distribution."""
        symbol = pair.replace('/', '')
        
        # Fetch recent klines for volume analysis
        url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['klines']}"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': 24  # Last 24 hours
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        volumes = [float(k[5]) for k in klines]  # Volume is at index 5
                        prices = [float(k[4]) for k in klines]   # Close price is at index 4
                        
                        # Calculate volume-weighted average price (VWAP)
                        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if volumes else 0
                        
                        # Identify high volume nodes
                        avg_volume = np.mean(volumes)
                        high_volume_periods = [i for i, v in enumerate(volumes) if v > avg_volume * 1.5]
                        
                        return {
                            'vwap': vwap,
                            'total_volume': sum(volumes),
                            'avg_volume': avg_volume,
                            'volume_std': np.std(volumes),
                            'high_volume_periods': high_volume_periods,
                            'volume_trend': 'increasing' if volumes[-1] > avg_volume else 'decreasing'
                        }
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
        
        return {}
    
    async def _calculate_volatility(self, pair: str) -> float:
        """Calculate market volatility using price data."""
        symbol = pair.replace('/', '')
        
        # Fetch hourly klines for volatility calculation
        url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['klines']}"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': 24
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        # Calculate returns
                        closes = [float(k[4]) for k in klines]
                        returns = []
                        for i in range(1, len(closes)):
                            ret = (closes[i] - closes[i-1]) / closes[i-1]
                            returns.append(ret)
                        
                        # Calculate volatility (annualized)
                        if returns:
                            volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized hourly volatility
                            return volatility * 100  # Convert to percentage
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
        
        return 0
    
    async def _get_correlation_data(self, pair: str) -> Dict[str, float]:
        """Calculate correlation with major pairs."""
        correlations = {}
        
        # Skip if pair is one of the major pairs
        if pair in self.major_pairs:
            return correlations
        
        try:
            # Fetch price data for target pair
            target_prices = await self._fetch_price_series(pair)
            
            if not target_prices:
                return correlations
            
            # Calculate correlation with each major pair
            for major_pair in self.major_pairs:
                major_prices = await self._fetch_price_series(major_pair)
                
                if major_prices and len(major_prices) == len(target_prices):
                    correlation = np.corrcoef(target_prices, major_prices)[0, 1]
                    correlations[major_pair] = round(correlation, 4)
        
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
        
        return correlations
    
    async def _fetch_price_series(self, pair: str, hours: int = 24) -> List[float]:
        """Fetch historical price series for correlation calculation."""
        symbol = pair.replace('/', '')
        
        url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['klines']}"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': hours
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        return [float(k[4]) for k in klines]  # Close prices
        except Exception as e:
            logger.error(f"Error fetching price series for {pair}: {e}")
        
        return []
    
    def _build_market_data(self,
                          pair: str,
                          ticker: Dict[str, Any],
                          order_book: Dict[str, Any],
                          volume_profile: Dict[str, Any],
                          volatility: float,
                          correlations: Dict[str, float]) -> MarketData:
        """Build MarketData object from components."""
        return MarketData(
            pair=pair,
            current_price=ticker.get('current_price', 0),
            bid=ticker.get('bid', 0),
            ask=ticker.get('ask', 0),
            spread=order_book.get('spread', 0),
            spread_percentage=order_book.get('spread_percentage', 0),
            volume_24h=ticker.get('volume_24h', 0),
            volume_quote_24h=ticker.get('volume_quote_24h', 0),
            high_24h=ticker.get('high_24h', 0),
            low_24h=ticker.get('low_24h', 0),
            price_change_24h=ticker.get('price_change_24h', 0),
            price_change_percentage_24h=ticker.get('price_change_percentage_24h', 0),
            order_book_depth=order_book.get('depth_levels', {}),
            liquidity_score=order_book.get('liquidity_score', 50),
            volatility=volatility,
            correlation_data=correlations,
            timestamp=datetime.now()
        )
    
    async def _get_cached_data(self, key: str) -> Optional[MarketData]:
        """Get cached market data if available."""
        if self.redis_client:
            try:
                cached = await self.redis_client.get(key)
                if cached:
                    # Deserialize and return
                    # Implementation depends on serialization format
                    pass
            except Exception as e:
                logger.error(f"Cache get error: {e}")
        return None
    
    async def _cache_data(self, key: str, data: MarketData, ttl: int):
        """Cache market data."""
        if self.redis_client:
            try:
                # Serialize and cache
                # Implementation depends on serialization format
                await self.redis_client.setex(key, ttl, str(data))
            except Exception as e:
                logger.error(f"Cache set error: {e}")
    
    async def validate_liquidity(self, pair: str, min_volume: float = 100000) -> Tuple[bool, str]:
        """
        Validate if pair has sufficient liquidity for trading.
        
        Args:
            pair: Trading pair
            min_volume: Minimum 24h volume in quote currency
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            market_data = await self.get_market_data(pair)
            
            # Check volume
            if market_data.volume_quote_24h < min_volume:
                return False, f"Insufficient 24h volume: ${market_data.volume_quote_24h:,.0f}"
            
            # Check spread
            if market_data.spread_percentage > 0.5:
                return False, f"Spread too wide: {market_data.spread_percentage:.2f}%"
            
            # Check liquidity score
            if market_data.liquidity_score < 30:
                return False, f"Low liquidity score: {market_data.liquidity_score:.0f}"
            
            return True, "Liquidity validated"
            
        except Exception as e:
            logger.error(f"Error validating liquidity: {e}")
            return False, "Failed to validate liquidity"
    
    async def get_slippage_estimate(self, pair: str, order_size: float, side: str = 'buy') -> float:
        """
        Estimate slippage for a given order size.
        
        Args:
            pair: Trading pair
            order_size: Order size in quote currency
            side: 'buy' or 'sell'
            
        Returns:
            Estimated slippage percentage
        """
        try:
            # Fetch order book
            symbol = pair.replace('/', '')
            url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['order_book']}"
            params = {'symbol': symbol, 'limit': 1000}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if side == 'buy':
                            orders = [(float(price), float(qty)) for price, qty in data.get('asks', [])]
                        else:
                            orders = [(float(price), float(qty)) for price, qty in data.get('bids', [])]
                        
                        if not orders:
                            return 0
                        
                        # Calculate average execution price
                        remaining = order_size
                        total_cost = 0
                        base_price = orders[0][0]
                        
                        for price, qty in orders:
                            qty_value = qty * price
                            
                            if remaining <= qty_value:
                                total_cost += remaining
                                break
                            else:
                                total_cost += qty_value
                                remaining -= qty_value
                        
                        if remaining > 0:
                            # Order too large for order book
                            return 999.99
                        
                        avg_price = total_cost / order_size
                        slippage = abs(avg_price - base_price) / base_price * 100
                        
                        return round(slippage, 4)
                        
        except Exception as e:
            logger.error(f"Error estimating slippage: {e}")
        
        return 0
