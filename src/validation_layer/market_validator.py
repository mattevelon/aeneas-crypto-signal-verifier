"""
Market Data Validation Module

Validates trading signals against real-time market data to ensure accuracy and feasibility.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, time
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum

import aiohttp
import numpy as np

from src.config.settings import get_settings
from src.core.redis_client import market_cache

logger = logging.getLogger(__name__)
settings = get_settings()


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class MarketValidation:
    """Market validation result."""
    price_deviation_check: ValidationResult
    spread_analysis: ValidationResult
    liquidity_check: ValidationResult
    slippage_estimate: float
    market_hours_check: ValidationResult
    overall_result: ValidationResult
    details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


class MarketDataValidator:
    """Validates signals against real-time market data."""
    
    # Validation thresholds
    MAX_PRICE_DEVIATION = 0.02  # 2% maximum deviation from current price
    MAX_SPREAD = 0.005  # 0.5% maximum spread
    MIN_DAILY_VOLUME = 100000  # $100k minimum daily volume
    MAX_SLIPPAGE = 0.01  # 1% maximum expected slippage
    
    # Market hours (UTC)
    FOREX_OPEN = time(21, 0)  # Sunday 21:00 UTC
    FOREX_CLOSE = time(21, 0)  # Friday 21:00 UTC
    CRYPTO_24_7 = True  # Crypto markets are 24/7
    
    def __init__(self):
        """Initialize market validator."""
        self.session = None
        self._cache_ttl = 60  # 1 minute cache for market data
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def validate_signal(
        self,
        signal: Dict[str, Any]
    ) -> MarketValidation:
        """
        Perform comprehensive market validation for a signal.
        
        Args:
            signal: Signal data to validate
            
        Returns:
            MarketValidation result
        """
        errors = []
        warnings = []
        details = {}
        
        # Extract signal parameters
        pair = signal.get('pair', '')
        entry_price = float(signal.get('entry_price', 0))
        stop_loss = float(signal.get('stop_loss', 0))
        take_profits = signal.get('take_profits', [])
        position_size = float(signal.get('position_size', 0))
        
        # Get current market data
        market_data = await self._get_market_data(pair)
        if not market_data:
            errors.append(f"Failed to fetch market data for {pair}")
            return self._create_validation_result(
                ValidationResult.FAIL,
                errors=errors
            )
        
        # 1. Price deviation check
        price_check, price_details = await self._check_price_deviation(
            entry_price, market_data['current_price']
        )
        details['price_deviation'] = price_details
        
        # 2. Spread analysis
        spread_check, spread_details = await self._analyze_spread(
            market_data['bid'], market_data['ask']
        )
        details['spread'] = spread_details
        
        # 3. Liquidity check
        liquidity_check, liquidity_details = await self._check_liquidity(
            market_data['volume_24h']
        )
        details['liquidity'] = liquidity_details
        
        # 4. Slippage estimation
        slippage = await self._estimate_slippage(
            position_size,
            market_data['order_book_depth'],
            market_data['current_price']
        )
        details['slippage_estimate'] = slippage
        
        # 5. Market hours validation
        hours_check = await self._check_market_hours(pair)
        details['market_hours'] = {
            'is_open': hours_check == ValidationResult.PASS,
            'market_type': 'crypto' if 'USDT' in pair or 'BTC' in pair else 'forex'
        }
        
        # Compile warnings
        if price_check == ValidationResult.WARNING:
            warnings.append(f"Price deviation: {price_details['deviation_percent']:.2f}%")
        if spread_check == ValidationResult.WARNING:
            warnings.append(f"High spread: {spread_details['spread_percent']:.3f}%")
        if liquidity_check == ValidationResult.WARNING:
            warnings.append(f"Low liquidity: ${liquidity_details['volume_24h']:,.0f}")
        if slippage > self.MAX_SLIPPAGE:
            warnings.append(f"High expected slippage: {slippage:.2f}%")
            
        # Determine overall result
        if price_check == ValidationResult.FAIL or liquidity_check == ValidationResult.FAIL:
            overall = ValidationResult.FAIL
        elif warnings:
            overall = ValidationResult.WARNING
        else:
            overall = ValidationResult.PASS
            
        return MarketValidation(
            price_deviation_check=price_check,
            spread_analysis=spread_check,
            liquidity_check=liquidity_check,
            slippage_estimate=slippage,
            market_hours_check=hours_check,
            overall_result=overall,
            details=details,
            warnings=warnings,
            errors=errors
        )
    
    async def _get_market_data(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current market data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., BTC/USDT)
            
        Returns:
            Market data dictionary or None
        """
        # Check cache first
        cache_key = f"market_data:{pair}"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached
            
        try:
            # Normalize pair for Binance API
            symbol = pair.replace('/', '')
            
            # Fetch ticker data
            ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with self.session.get(ticker_url) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch ticker for {symbol}: {resp.status}")
                    return None
                ticker_data = await resp.json()
            
            # Fetch order book
            depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
            async with self.session.get(depth_url) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch order book for {symbol}: {resp.status}")
                    return None
                depth_data = await resp.json()
            
            market_data = {
                'current_price': float(ticker_data['lastPrice']),
                'bid': float(ticker_data['bidPrice']),
                'ask': float(ticker_data['askPrice']),
                'volume_24h': float(ticker_data['quoteVolume']),
                'price_change_24h': float(ticker_data['priceChangePercent']),
                'high_24h': float(ticker_data['highPrice']),
                'low_24h': float(ticker_data['lowPrice']),
                'order_book_depth': {
                    'bids': [[float(p), float(q)] for p, q in depth_data['bids'][:10]],
                    'asks': [[float(p), float(q)] for p, q in depth_data['asks'][:10]]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            await market_cache.set(cache_key, market_data, ttl=self._cache_ttl)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {pair}: {e}")
            return None
    
    async def _check_price_deviation(
        self,
        signal_price: float,
        current_price: float
    ) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Check if signal price deviates too much from current market price.
        
        Args:
            signal_price: Price in the signal
            current_price: Current market price
            
        Returns:
            Validation result and details
        """
        if current_price == 0:
            return ValidationResult.FAIL, {'error': 'Invalid current price'}
            
        deviation = abs(signal_price - current_price) / current_price
        deviation_percent = deviation * 100
        
        details = {
            'signal_price': signal_price,
            'current_price': current_price,
            'deviation': deviation,
            'deviation_percent': deviation_percent,
            'max_allowed': self.MAX_PRICE_DEVIATION * 100
        }
        
        if deviation > self.MAX_PRICE_DEVIATION:
            return ValidationResult.FAIL, details
        elif deviation > self.MAX_PRICE_DEVIATION * 0.5:  # Warning if > 1%
            return ValidationResult.WARNING, details
        else:
            return ValidationResult.PASS, details
    
    async def _analyze_spread(
        self,
        bid: float,
        ask: float
    ) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Analyze bid-ask spread.
        
        Args:
            bid: Current bid price
            ask: Current ask price
            
        Returns:
            Validation result and details
        """
        if bid == 0 or ask == 0:
            return ValidationResult.FAIL, {'error': 'Invalid bid/ask prices'}
            
        spread = ask - bid
        mid_price = (ask + bid) / 2
        spread_percent = (spread / mid_price) * 100
        
        details = {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'spread_percent': spread_percent,
            'mid_price': mid_price,
            'max_allowed': self.MAX_SPREAD * 100
        }
        
        if spread_percent > self.MAX_SPREAD * 100:
            return ValidationResult.FAIL, details
        elif spread_percent > self.MAX_SPREAD * 50:  # Warning if > 0.25%
            return ValidationResult.WARNING, details
        else:
            return ValidationResult.PASS, details
    
    async def _check_liquidity(
        self,
        volume_24h: float
    ) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Check if market has sufficient liquidity.
        
        Args:
            volume_24h: 24-hour trading volume in quote currency
            
        Returns:
            Validation result and details
        """
        details = {
            'volume_24h': volume_24h,
            'min_required': self.MIN_DAILY_VOLUME,
            'liquidity_ratio': volume_24h / self.MIN_DAILY_VOLUME if self.MIN_DAILY_VOLUME > 0 else 0
        }
        
        if volume_24h < self.MIN_DAILY_VOLUME:
            return ValidationResult.FAIL, details
        elif volume_24h < self.MIN_DAILY_VOLUME * 2:  # Warning if < $200k
            return ValidationResult.WARNING, details
        else:
            return ValidationResult.PASS, details
    
    async def _estimate_slippage(
        self,
        position_size: float,
        order_book: Dict[str, List],
        current_price: float
    ) -> float:
        """
        Estimate potential slippage based on order book depth.
        
        Args:
            position_size: Size of the position
            order_book: Order book data
            current_price: Current market price
            
        Returns:
            Estimated slippage as percentage
        """
        if position_size == 0 or not order_book:
            return 0.0
            
        # Calculate how much of the order book we'd consume
        total_volume = 0
        weighted_price = 0
        
        for price, quantity in order_book.get('asks', []):
            if total_volume >= position_size:
                break
            volume = min(quantity, position_size - total_volume)
            weighted_price += price * volume
            total_volume += volume
        
        if total_volume > 0:
            avg_execution_price = weighted_price / total_volume
            slippage = abs(avg_execution_price - current_price) / current_price
            return slippage * 100  # Return as percentage
        
        return 0.0
    
    async def _check_market_hours(self, pair: str) -> ValidationResult:
        """
        Check if market is open for trading.
        
        Args:
            pair: Trading pair
            
        Returns:
            Validation result
        """
        # Crypto markets are always open
        if 'USDT' in pair or 'BTC' in pair or 'ETH' in pair:
            return ValidationResult.PASS
            
        # Check forex market hours
        now = datetime.utcnow()
        current_time = now.time()
        weekday = now.weekday()
        
        # Forex closed on weekends (Saturday and Sunday until 21:00 UTC)
        if weekday == 5:  # Saturday
            return ValidationResult.FAIL
        elif weekday == 6 and current_time < self.FOREX_OPEN:  # Sunday before open
            return ValidationResult.FAIL
        elif weekday == 4 and current_time > self.FOREX_CLOSE:  # Friday after close
            return ValidationResult.FAIL
            
        return ValidationResult.PASS
    
    def _create_validation_result(
        self,
        overall: ValidationResult,
        **kwargs
    ) -> MarketValidation:
        """Create a validation result with default values."""
        return MarketValidation(
            price_deviation_check=kwargs.get('price_check', ValidationResult.FAIL),
            spread_analysis=kwargs.get('spread_check', ValidationResult.FAIL),
            liquidity_check=kwargs.get('liquidity_check', ValidationResult.FAIL),
            slippage_estimate=kwargs.get('slippage', 0.0),
            market_hours_check=kwargs.get('hours_check', ValidationResult.FAIL),
            overall_result=overall,
            details=kwargs.get('details', {}),
            warnings=kwargs.get('warnings', []),
            errors=kwargs.get('errors', [])
        )
