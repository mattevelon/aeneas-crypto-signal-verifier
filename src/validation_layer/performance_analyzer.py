"""
Historical Performance Analysis Module

Analyzes historical trading performance to provide insights for signal validation
and backtesting capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import get_settings
from src.core.database import get_async_session
from src.models import Signal
from sqlalchemy import select, func, and_, or_

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic metrics
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    
    # Return metrics
    total_return: float
    average_return: float
    best_return: float
    worst_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    
    # Statistical metrics
    profit_factor: float
    expectancy: float
    kelly_criterion: float
    recovery_factor: float
    
    # Time-based metrics
    avg_holding_period: timedelta
    longest_winning_streak: int
    longest_losing_streak: int
    
    # Detailed breakdown
    returns_by_pair: Dict[str, float]
    performance_by_month: Dict[str, float]
    performance_by_channel: Dict[str, Dict[str, Any]]


@dataclass
class BacktestResult:
    """Result of backtesting a signal or strategy."""
    signal_id: str
    entry_price: float
    exit_price: float
    return_percentage: float
    holding_period: timedelta
    max_profit: float
    max_loss: float
    hit_stop_loss: bool
    hit_take_profit: int  # Which TP level was hit (0 if none)
    slippage: float
    fees: float
    net_return: float


class HistoricalPerformanceAnalyzer:
    """Analyzes historical performance for insights and backtesting."""
    
    # Performance calculation parameters
    RISK_FREE_RATE = 0.03  # 3% annual risk-free rate
    TRADING_DAYS_YEAR = 365  # Crypto trades 24/7
    DEFAULT_FEE_RATE = 0.001  # 0.1% trading fee
    DEFAULT_SLIPPAGE = 0.0005  # 0.05% slippage
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.performance_cache = {}
        self.backtest_cache = {}
        
    async def calculate_performance_metrics(
        self,
        channel_id: Optional[str] = None,
        pair: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            channel_id: Filter by channel
            pair: Filter by trading pair
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            PerformanceMetrics
        """
        # Get signals from database
        signals = await self._fetch_signals(channel_id, pair, start_date, end_date)
        
        if not signals:
            return self._empty_metrics()
            
        # Calculate returns for each signal
        returns = []
        signal_results = []
        
        for signal in signals:
            result = await self._calculate_signal_return(signal)
            signal_results.append(result)
            returns.append(result.return_percentage)
            
        # Basic metrics
        total_signals = len(signals)
        winning_signals = sum(1 for r in returns if r > 0)
        losing_signals = sum(1 for r in returns if r < 0)
        win_rate = winning_signals / total_signals if total_signals > 0 else 0
        
        # Return metrics
        total_return = sum(returns)
        average_return = np.mean(returns) if returns else 0
        best_return = max(returns) if returns else 0
        worst_return = min(returns) if returns else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95 = self._calculate_var(returns, 0.95)
        
        # Statistical metrics
        profit_factor = self._calculate_profit_factor(returns)
        expectancy = self._calculate_expectancy(returns, win_rate)
        kelly_criterion = self._calculate_kelly(win_rate, returns)
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
        
        # Time-based metrics
        holding_periods = [r.holding_period for r in signal_results]
        avg_holding_period = sum(holding_periods, timedelta()) / len(holding_periods) if holding_periods else timedelta()
        
        winning_streak, losing_streak = self._calculate_streaks(returns)
        
        # Breakdown by pair
        returns_by_pair = await self._calculate_returns_by_pair(signal_results)
        
        # Breakdown by month
        performance_by_month = await self._calculate_monthly_performance(signal_results)
        
        # Breakdown by channel
        performance_by_channel = await self._calculate_channel_performance(signal_results)
        
        return PerformanceMetrics(
            total_signals=total_signals,
            winning_signals=winning_signals,
            losing_signals=losing_signals,
            win_rate=win_rate,
            total_return=total_return,
            average_return=average_return,
            best_return=best_return,
            worst_return=worst_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion,
            recovery_factor=recovery_factor,
            avg_holding_period=avg_holding_period,
            longest_winning_streak=winning_streak,
            longest_losing_streak=losing_streak,
            returns_by_pair=returns_by_pair,
            performance_by_month=performance_by_month,
            performance_by_channel=performance_by_channel
        )
    
    async def backtest_signal(
        self,
        signal: Dict[str, Any],
        historical_data: Dict[str, Any],
        slippage: float = None,
        fees: float = None
    ) -> BacktestResult:
        """
        Backtest a signal against historical data.
        
        Args:
            signal: Signal to backtest
            historical_data: Historical price data
            slippage: Slippage percentage
            fees: Trading fee percentage
            
        Returns:
            BacktestResult
        """
        if slippage is None:
            slippage = self.DEFAULT_SLIPPAGE
        if fees is None:
            fees = self.DEFAULT_FEE_RATE
            
        entry_price = float(signal.get('entry_price', 0))
        stop_loss = float(signal.get('stop_loss', 0))
        take_profits = signal.get('take_profits', [])
        
        # Simulate entry with slippage
        actual_entry = entry_price * (1 + slippage)
        
        # Get price history
        prices = historical_data.get('prices', [])
        timestamps = historical_data.get('timestamps', [])
        
        if not prices or not timestamps:
            # No historical data, use current price
            exit_price = entry_price
            holding_period = timedelta(hours=24)
            hit_sl = False
            hit_tp = 0
        else:
            # Simulate price movement
            exit_price, exit_time, hit_sl, hit_tp = await self._simulate_trade(
                actual_entry, stop_loss, take_profits, prices, timestamps
            )
            
            entry_time = datetime.fromisoformat(signal.get('created_at', datetime.now().isoformat()))
            holding_period = exit_time - entry_time
            
        # Calculate returns
        gross_return = (exit_price - actual_entry) / actual_entry
        
        # Apply fees (entry + exit)
        total_fees = fees * 2
        net_return = gross_return - total_fees
        
        # Track max profit/loss during trade
        if prices:
            max_price = max(prices)
            min_price = min(prices)
            max_profit = (max_price - actual_entry) / actual_entry
            max_loss = (min_price - actual_entry) / actual_entry
        else:
            max_profit = max(0, net_return)
            max_loss = min(0, net_return)
            
        return BacktestResult(
            signal_id=signal.get('id', ''),
            entry_price=actual_entry,
            exit_price=exit_price,
            return_percentage=gross_return * 100,
            holding_period=holding_period,
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            hit_stop_loss=hit_sl,
            hit_take_profit=hit_tp,
            slippage=slippage * 100,
            fees=total_fees * 100,
            net_return=net_return * 100
        )
    
    async def _fetch_signals(
        self,
        channel_id: Optional[str],
        pair: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Signal]:
        """Fetch signals from database with filters."""
        async with get_async_session() as session:
            query = select(Signal)
            
            # Apply filters
            conditions = []
            if channel_id:
                conditions.append(Signal.source_channel_id == channel_id)
            if pair:
                conditions.append(Signal.pair == pair)
            if start_date:
                conditions.append(Signal.created_at >= start_date)
            if end_date:
                conditions.append(Signal.created_at <= end_date)
                
            if conditions:
                query = query.where(and_(*conditions))
                
            result = await session.execute(query.order_by(Signal.created_at))
            return result.scalars().all()
    
    async def _calculate_signal_return(self, signal: Signal) -> BacktestResult:
        """Calculate return for a single signal."""
        # Simplified return calculation
        # In production, this would use actual exit data
        entry = float(signal.entry_price)
        
        # Simulate exit based on risk/reward
        if signal.status == 'COMPLETED':
            # Use actual exit data if available
            exit_price = float(signal.metadata.get('exit_price', entry * 1.02))
        else:
            # Estimate based on targets
            if signal.take_profits:
                exit_price = float(signal.take_profits[0])  # First TP
            else:
                exit_price = entry * 1.02  # Default 2% profit
                
        return_pct = ((exit_price - entry) / entry) * 100
        
        return BacktestResult(
            signal_id=str(signal.id),
            entry_price=entry,
            exit_price=exit_price,
            return_percentage=return_pct,
            holding_period=timedelta(hours=24),  # Default
            max_profit=max(0, return_pct),
            max_loss=min(0, return_pct),
            hit_stop_loss=return_pct < 0,
            hit_take_profit=1 if return_pct > 0 else 0,
            slippage=0.05,
            fees=0.2,
            net_return=return_pct - 0.25
        )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns) / 100
        excess_returns = returns_array - (self.RISK_FREE_RATE / self.TRADING_DAYS_YEAR)
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
            
        return mean_excess / std_excess * np.sqrt(self.TRADING_DAYS_YEAR)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns) / 100
        excess_returns = returns_array - (self.RISK_FREE_RATE / self.TRADING_DAYS_YEAR)
        
        mean_excess = np.mean(excess_returns)
        
        # Calculate downside deviation
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
        else:
            return float('inf') if mean_excess > 0 else 0.0
            
        if downside_std == 0:
            return 0.0
            
        return mean_excess / downside_std * np.sqrt(self.TRADING_DAYS_YEAR)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0
            
        # Calculate cumulative returns
        cumulative = []
        cum_return = 1.0
        
        for r in returns:
            cum_return *= (1 + r / 100)
            cumulative.append(cum_return)
            
        # Calculate running maximum
        running_max = []
        max_val = cumulative[0]
        
        for val in cumulative:
            if val > max_val:
                max_val = val
            running_max.append(max_val)
            
        # Calculate drawdown
        drawdowns = [(cum - max_val) / max_val for cum, max_val in zip(cumulative, running_max)]
        
        return abs(min(drawdowns)) if drawdowns else 0.0
    
    def _calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if not returns:
            return 0.0
            
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence) * 100)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if not returns:
            return 0.0
            
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        
        if losses == 0:
            return float('inf') if profits > 0 else 0.0
            
        return profits / losses
    
    def _calculate_expectancy(self, returns: List[float], win_rate: float) -> float:
        """Calculate expectancy (expected value per trade)."""
        if not returns:
            return 0.0
            
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = abs(np.mean(losing_returns)) if losing_returns else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _calculate_kelly(self, win_rate: float, returns: List[float]) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        if not returns or win_rate == 0:
            return 0.0
            
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        if not winning_returns or not losing_returns:
            return 0.0
            
        avg_win = np.mean(winning_returns)
        avg_loss = abs(np.mean(losing_returns))
        
        if avg_loss == 0:
            return 0.0
            
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly (25%) for safety
        return max(0, min(kelly * 0.25, 0.1))  # Cap at 10%
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
            
        return abs(total_return / max_drawdown)
    
    def _calculate_streaks(self, returns: List[float]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        if not returns:
            return 0, 0
            
        current_win_streak = 0
        current_lose_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        
        for r in returns:
            if r > 0:
                current_win_streak += 1
                current_lose_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif r < 0:
                current_lose_streak += 1
                current_win_streak = 0
                max_lose_streak = max(max_lose_streak, current_lose_streak)
                
        return max_win_streak, max_lose_streak
    
    async def _calculate_returns_by_pair(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Calculate returns grouped by trading pair."""
        returns_by_pair = {}
        
        # Group by pair (would need signal data for actual pair info)
        # Simplified version
        for result in results:
            pair = "BTC/USDT"  # Would get from signal
            if pair not in returns_by_pair:
                returns_by_pair[pair] = []
            returns_by_pair[pair].append(result.return_percentage)
            
        # Calculate average return per pair
        for pair in returns_by_pair:
            returns_by_pair[pair] = np.mean(returns_by_pair[pair])
            
        return returns_by_pair
    
    async def _calculate_monthly_performance(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Calculate performance by month."""
        monthly_returns = {}
        
        # Simplified - would use actual dates from signals
        current_month = datetime.now().strftime("%Y-%m")
        monthly_returns[current_month] = np.mean([r.return_percentage for r in results])
        
        return monthly_returns
    
    async def _calculate_channel_performance(
        self,
        results: List[BacktestResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate performance by channel."""
        # Simplified - would group by actual channel
        return {
            "default_channel": {
                "total_signals": len(results),
                "win_rate": sum(1 for r in results if r.return_percentage > 0) / len(results) if results else 0,
                "avg_return": np.mean([r.return_percentage for r in results]) if results else 0
            }
        }
    
    async def _simulate_trade(
        self,
        entry: float,
        stop_loss: float,
        take_profits: List[float],
        prices: List[float],
        timestamps: List[datetime]
    ) -> Tuple[float, datetime, bool, int]:
        """Simulate trade execution against historical prices."""
        for i, price in enumerate(prices):
            # Check stop loss
            if price <= stop_loss:
                return stop_loss, timestamps[i], True, 0
                
            # Check take profits
            for tp_idx, tp in enumerate(take_profits):
                if price >= tp:
                    return tp, timestamps[i], False, tp_idx + 1
                    
        # No exit triggered, use last price
        return prices[-1], timestamps[-1], False, 0
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_signals=0,
            winning_signals=0,
            losing_signals=0,
            win_rate=0.0,
            total_return=0.0,
            average_return=0.0,
            best_return=0.0,
            worst_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            kelly_criterion=0.0,
            recovery_factor=0.0,
            avg_holding_period=timedelta(),
            longest_winning_streak=0,
            longest_losing_streak=0,
            returns_by_pair={},
            performance_by_month={},
            performance_by_channel={}
        )
