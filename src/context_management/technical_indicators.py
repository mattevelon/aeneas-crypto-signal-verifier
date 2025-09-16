"""Technical Indicators Service for comprehensive market analysis."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TechnicalAnalysis:
    """Technical analysis results for a trading pair."""
    pair: str
    timeframe: str
    indicators: Dict[str, float]
    oscillators: Dict[str, Any]
    moving_averages: Dict[str, float]
    support_resistance: Dict[str, List[float]]
    patterns: List[str]
    divergences: List[Dict[str, Any]]
    confluence_score: float
    signal_strength: str  # strong_buy, buy, neutral, sell, strong_sell
    timestamp: datetime


class TechnicalIndicatorService:
    """
    Calculates technical indicators using TA-Lib equivalent implementations.
    Supports multi-timeframe analysis and divergence detection.
    """
    
    def __init__(self):
        """Initialize technical indicator service."""
        self.timeframes = ['15m', '1h', '4h', '1d']
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Indicator parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        
        logger.info("Initialized TechnicalIndicatorService")
    
    async def analyze(self, pair: str, timeframe: str = '1h') -> TechnicalAnalysis:
        """
        Perform comprehensive technical analysis on a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            timeframe: Analysis timeframe
            
        Returns:
            Complete technical analysis
        """
        # Check cache
        cache_key = f"ta:{pair}:{timeframe}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        # Fetch price data
        price_data = await self._fetch_price_data(pair, timeframe)
        
        if not price_data or len(price_data) < 100:
            logger.warning(f"Insufficient price data for {pair}")
            return self._create_empty_analysis(pair, timeframe)
        
        # Calculate all indicators
        tasks = [
            self._calculate_momentum_indicators(price_data),
            self._calculate_moving_averages(price_data),
            self._calculate_volatility_indicators(price_data),
            self._identify_support_resistance(price_data),
            self._detect_patterns(price_data),
            self._detect_divergences(price_data)
        ]
        
        results = await asyncio.gather(*tasks)
        
        momentum = results[0]
        moving_avgs = results[1]
        volatility = results[2]
        support_resistance = results[3]
        patterns = results[4]
        divergences = results[5]
        
        # Combine all indicators
        indicators = {**momentum, **volatility}
        
        # Calculate confluence score
        confluence_score = self._calculate_confluence(
            indicators, moving_avgs, patterns, divergences
        )
        
        # Determine signal strength
        signal_strength = self._determine_signal_strength(
            indicators, moving_avgs, confluence_score
        )
        
        # Build analysis result
        analysis = TechnicalAnalysis(
            pair=pair,
            timeframe=timeframe,
            indicators=indicators,
            oscillators=momentum,
            moving_averages=moving_avgs,
            support_resistance=support_resistance,
            patterns=patterns,
            divergences=divergences,
            confluence_score=confluence_score,
            signal_strength=signal_strength,
            timestamp=datetime.now()
        )
        
        # Cache result
        self.cache[cache_key] = (analysis, datetime.now())
        
        return analysis
    
    async def _fetch_price_data(self, pair: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch historical price data for analysis."""
        symbol = pair.replace('/', '')
        
        # Map timeframe to Binance interval
        interval_map = {
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        interval = interval_map.get(timeframe, '1h')
        
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(klines, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])
                        
                        # Convert to numeric types
                        df['open'] = pd.to_numeric(df['open'])
                        df['high'] = pd.to_numeric(df['high'])
                        df['low'] = pd.to_numeric(df['low'])
                        df['close'] = pd.to_numeric(df['close'])
                        df['volume'] = pd.to_numeric(df['volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        return df
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
        
        return pd.DataFrame()
    
    async def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators (RSI, MACD, Stochastic)."""
        indicators = {}
        
        # RSI
        rsi = self._calculate_rsi(df['close'].values, self.rsi_period)
        indicators['rsi'] = round(rsi[-1], 2) if len(rsi) > 0 else 50
        indicators['rsi_overbought'] = indicators['rsi'] > 70
        indicators['rsi_oversold'] = indicators['rsi'] < 30
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            df['close'].values,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        
        if len(macd_line) > 0:
            indicators['macd'] = round(macd_line[-1], 4)
            indicators['macd_signal'] = round(signal_line[-1], 4)
            indicators['macd_histogram'] = round(histogram[-1], 4)
            indicators['macd_cross'] = 'bullish' if histogram[-1] > 0 else 'bearish'
        
        # Stochastic
        k, d = self._calculate_stochastic(df['high'].values, df['low'].values, df['close'].values)
        if len(k) > 0:
            indicators['stoch_k'] = round(k[-1], 2)
            indicators['stoch_d'] = round(d[-1], 2)
            indicators['stoch_overbought'] = k[-1] > 80
            indicators['stoch_oversold'] = k[-1] < 20
        
        # CCI (Commodity Channel Index)
        cci = self._calculate_cci(df['high'].values, df['low'].values, df['close'].values)
        indicators['cci'] = round(cci, 2)
        indicators['cci_overbought'] = cci > 100
        indicators['cci_oversold'] = cci < -100
        
        return indicators
    
    async def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages."""
        mas = {}
        close_prices = df['close'].values
        
        # Simple Moving Averages
        for period in [7, 20, 50, 100, 200]:
            if len(close_prices) >= period:
                sma = np.mean(close_prices[-period:])
                mas[f'sma_{period}'] = round(sma, 2)
        
        # Exponential Moving Averages
        for period in [9, 21, 50, 100]:
            if len(close_prices) >= period:
                ema = self._calculate_ema(close_prices, period)
                if len(ema) > 0:
                    mas[f'ema_{period}'] = round(ema[-1], 2)
        
        # Current price position relative to MAs
        current_price = close_prices[-1]
        mas['price_above_sma_20'] = current_price > mas.get('sma_20', current_price)
        mas['price_above_sma_50'] = current_price > mas.get('sma_50', current_price)
        mas['price_above_sma_200'] = current_price > mas.get('sma_200', current_price)
        
        # Golden/Death cross detection
        if 'sma_50' in mas and 'sma_200' in mas:
            mas['golden_cross'] = mas['sma_50'] > mas['sma_200']
            mas['death_cross'] = mas['sma_50'] < mas['sma_200']
        
        return mas
    
    async def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility indicators (Bollinger Bands, ATR)."""
        indicators = {}
        
        # Bollinger Bands
        close_prices = df['close'].values
        if len(close_prices) >= self.bb_period:
            sma = np.mean(close_prices[-self.bb_period:])
            std = np.std(close_prices[-self.bb_period:])
            
            indicators['bb_upper'] = round(sma + (self.bb_std * std), 2)
            indicators['bb_middle'] = round(sma, 2)
            indicators['bb_lower'] = round(sma - (self.bb_std * std), 2)
            
            # Position within bands
            current_price = close_prices[-1]
            bb_width = indicators['bb_upper'] - indicators['bb_lower']
            if bb_width > 0:
                indicators['bb_position'] = round(
                    (current_price - indicators['bb_lower']) / bb_width * 100, 2
                )
        
        # ATR (Average True Range)
        atr = self._calculate_atr(df['high'].values, df['low'].values, df['close'].values)
        indicators['atr'] = round(atr, 4)
        
        # Volatility percentage
        if len(close_prices) >= 20:
            returns = np.diff(close_prices[-20:]) / close_prices[-20:-1]
            indicators['volatility'] = round(np.std(returns) * 100, 2)
        
        return indicators
    
    async def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify support and resistance levels."""
        levels = {
            'support': [],
            'resistance': [],
            'pivot_points': []
        }
        
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        if len(close_prices) < 20:
            return levels
        
        # Find local minima and maxima
        window = 10
        for i in range(window, len(close_prices) - window):
            # Check for local maximum (resistance)
            if high_prices[i] == max(high_prices[i-window:i+window+1]):
                levels['resistance'].append(round(high_prices[i], 2))
            
            # Check for local minimum (support)
            if low_prices[i] == min(low_prices[i-window:i+window+1]):
                levels['support'].append(round(low_prices[i], 2))
        
        # Remove duplicates and sort
        levels['support'] = sorted(list(set(levels['support'])))[-5:]  # Keep top 5
        levels['resistance'] = sorted(list(set(levels['resistance'])))[:5]  # Keep bottom 5
        
        # Calculate pivot points
        last_high = high_prices[-1]
        last_low = low_prices[-1]
        last_close = close_prices[-1]
        
        pivot = (last_high + last_low + last_close) / 3
        levels['pivot_points'] = {
            'pivot': round(pivot, 2),
            'r1': round(2 * pivot - last_low, 2),
            'r2': round(pivot + (last_high - last_low), 2),
            's1': round(2 * pivot - last_high, 2),
            's2': round(pivot - (last_high - last_low), 2)
        }
        
        return levels
    
    async def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect chart patterns."""
        patterns = []
        
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        if len(close_prices) < 20:
            return patterns
        
        # Trend detection
        sma_20 = np.mean(close_prices[-20:])
        sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma_20
        
        if close_prices[-1] > sma_20 > sma_50:
            patterns.append('uptrend')
        elif close_prices[-1] < sma_20 < sma_50:
            patterns.append('downtrend')
        else:
            patterns.append('ranging')
        
        # Candlestick patterns (simplified)
        last_open = df['open'].values[-1]
        last_close = close_prices[-1]
        last_high = high_prices[-1]
        last_low = low_prices[-1]
        
        body = abs(last_close - last_open)
        upper_shadow = last_high - max(last_open, last_close)
        lower_shadow = min(last_open, last_close) - last_low
        
        # Doji
        if body < (last_high - last_low) * 0.1:
            patterns.append('doji')
        
        # Hammer
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns.append('hammer')
        
        # Shooting star
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns.append('shooting_star')
        
        # Breakout detection
        if len(close_prices) >= 20:
            recent_high = max(high_prices[-20:-1])
            recent_low = min(low_prices[-20:-1])
            
            if close_prices[-1] > recent_high:
                patterns.append('breakout_up')
            elif close_prices[-1] < recent_low:
                patterns.append('breakout_down')
        
        return patterns
    
    async def _detect_divergences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect divergences between price and indicators."""
        divergences = []
        
        close_prices = df['close'].values
        if len(close_prices) < 50:
            return divergences
        
        # Calculate RSI for divergence
        rsi = self._calculate_rsi(close_prices, self.rsi_period)
        
        if len(rsi) < 20:
            return divergences
        
        # Look for divergences in last 20 periods
        for i in range(-20, -2):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if close_prices[i] < close_prices[i-5] and rsi[i] > rsi[i-5]:
                if close_prices[i] == min(close_prices[i-5:i+1]):
                    divergences.append({
                        'type': 'bullish_divergence',
                        'indicator': 'rsi',
                        'strength': 'regular',
                        'period': abs(i)
                    })
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if close_prices[i] > close_prices[i-5] and rsi[i] < rsi[i-5]:
                if close_prices[i] == max(close_prices[i-5:i+1]):
                    divergences.append({
                        'type': 'bearish_divergence',
                        'indicator': 'rsi',
                        'strength': 'regular',
                        'period': abs(i)
                    })
        
        return divergences[:3]  # Limit to 3 most recent
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return np.array([50])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period:]
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        if len(prices) < slow + signal:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast[slow-fast:] - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line[signal-1:] - signal_line
        
        return macd_line[signal-1:], signal_line, histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema[period-1:]
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            period: int = 14, smooth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic oscillator."""
        if len(close) < period + smooth:
            return np.array([50]), np.array([50])
        
        k_values = []
        for i in range(period - 1, len(close)):
            highest = max(high[i-period+1:i+1])
            lowest = min(low[i-period+1:i+1])
            
            if highest - lowest != 0:
                k = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                k = 50
            k_values.append(k)
        
        k_values = np.array(k_values)
        d_values = np.convolve(k_values, np.ones(smooth)/smooth, mode='valid')
        
        return k_values[-len(d_values):], d_values
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        if len(close) < period:
            return 0
        
        typical_price = (high + low + close) / 3
        sma = np.mean(typical_price[-period:])
        mad = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mad != 0:
            cci = (typical_price[-1] - sma) / (0.015 * mad)
        else:
            cci = 0
        
        return cci
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return 0
        
        true_ranges = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return np.mean(true_ranges[-period:])
        return 0
    
    def _calculate_confluence(self, indicators: Dict, mas: Dict, patterns: List, divergences: List) -> float:
        """Calculate confluence score based on multiple indicators."""
        score = 50.0  # Base score
        
        # RSI signals
        if indicators.get('rsi_oversold'):
            score += 10
        elif indicators.get('rsi_overbought'):
            score -= 10
        
        # MACD signals
        if indicators.get('macd_cross') == 'bullish':
            score += 10
        elif indicators.get('macd_cross') == 'bearish':
            score -= 10
        
        # Moving average signals
        if mas.get('price_above_sma_50'):
            score += 5
        if mas.get('price_above_sma_200'):
            score += 5
        if mas.get('golden_cross'):
            score += 15
        elif mas.get('death_cross'):
            score -= 15
        
        # Pattern signals
        if 'uptrend' in patterns:
            score += 10
        elif 'downtrend' in patterns:
            score -= 10
        
        if 'breakout_up' in patterns:
            score += 15
        elif 'breakout_down' in patterns:
            score -= 15
        
        # Divergence signals
        for div in divergences:
            if div['type'] == 'bullish_divergence':
                score += 10
            elif div['type'] == 'bearish_divergence':
                score -= 10
        
        return max(0, min(100, score))
    
    def _determine_signal_strength(self, indicators: Dict, mas: Dict, confluence: float) -> str:
        """Determine overall signal strength."""
        if confluence >= 80:
            return 'strong_buy'
        elif confluence >= 65:
            return 'buy'
        elif confluence <= 20:
            return 'strong_sell'
        elif confluence <= 35:
            return 'sell'
        else:
            return 'neutral'
    
    def _create_empty_analysis(self, pair: str, timeframe: str) -> TechnicalAnalysis:
        """Create empty analysis when data is insufficient."""
        return TechnicalAnalysis(
            pair=pair,
            timeframe=timeframe,
            indicators={},
            oscillators={},
            moving_averages={},
            support_resistance={'support': [], 'resistance': [], 'pivot_points': []},
            patterns=[],
            divergences=[],
            confluence_score=50.0,
            signal_strength='neutral',
            timestamp=datetime.now()
        )
    
    async def multi_timeframe_analysis(self, pair: str) -> Dict[str, TechnicalAnalysis]:
        """
        Perform analysis across multiple timeframes.
        
        Args:
            pair: Trading pair
            
        Returns:
            Dictionary of analyses by timeframe
        """
        tasks = []
        for timeframe in self.timeframes:
            tasks.append(self.analyze(pair, timeframe))
        
        results = await asyncio.gather(*tasks)
        
        return {tf: analysis for tf, analysis in zip(self.timeframes, results)}
