"""
P&L (Profit and Loss) Calculator

Calculates comprehensive profit and loss metrics for trading signals,
including realized/unrealized P&L, fees, slippage, and various performance ratios.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
import logging
from dataclasses import dataclass
import json

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import pandas as pd

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeeStructure(Enum):
    """Exchange fee structures"""
    MAKER = "maker"
    TAKER = "taker"
    VIP = "vip"


@dataclass
class TradingFees:
    """Trading fee configuration"""
    exchange: str
    maker_fee: Decimal = Decimal("0.001")  # 0.1%
    taker_fee: Decimal = Decimal("0.001")  # 0.1%
    vip_discount: Decimal = Decimal("0")   # VIP discount percentage
    
    def get_fee(self, fee_type: FeeStructure) -> Decimal:
        """Get fee rate based on type"""
        if fee_type == FeeStructure.MAKER:
            return self.maker_fee * (1 - self.vip_discount)
        elif fee_type == FeeStructure.TAKER:
            return self.taker_fee * (1 - self.vip_discount)
        else:  # VIP
            return min(self.maker_fee, self.taker_fee) * (1 - self.vip_discount)


@dataclass
class PnLMetrics:
    """Comprehensive P&L metrics"""
    gross_pnl: Decimal
    net_pnl: Decimal
    fees_paid: Decimal
    slippage_cost: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    roi_percentage: Decimal
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: Decimal
    win_rate: float
    profit_factor: float
    expectancy: Decimal
    risk_reward_ratio: float
    recovery_factor: Optional[float]


class PnLCalculator:
    """
    Comprehensive P&L calculation engine for trading signals
    """
    
    def __init__(self):
        self.redis = get_redis()
        self.risk_free_rate = Decimal("0.02")  # 2% annual risk-free rate
        
        # Exchange fees configuration
        self.fee_structures = {
            'binance': TradingFees('binance', Decimal("0.001"), Decimal("0.001")),
            'kucoin': TradingFees('kucoin', Decimal("0.001"), Decimal("0.001")),
            'coinbase': TradingFees('coinbase', Decimal("0.0025"), Decimal("0.0035")),
            'default': TradingFees('default', Decimal("0.001"), Decimal("0.001"))
        }
    
    async def calculate_signal_pnl(self, signal_id: str, 
                                   position_size: Optional[Decimal] = None,
                                   fee_type: FeeStructure = FeeStructure.TAKER) -> Optional[PnLMetrics]:
        """
        Calculate P&L for a single signal
        
        Args:
            signal_id: Signal identifier
            position_size: Position size in base currency
            fee_type: Fee structure type
            
        Returns:
            PnLMetrics object or None if calculation fails
        """
        try:
            async with get_async_session() as session:
                # Get signal and performance data
                signal = await self._get_signal(session, signal_id)
                performance = await self._get_performance(session, signal_id)
                
                if not signal or not performance:
                    logger.error(f"Signal or performance data not found for {signal_id}")
                    return None
                
                # Use provided position size or default
                if not position_size:
                    position_size = performance.position_size or Decimal("1000")
                
                # Get exchange fees
                exchange = self._extract_exchange_from_pair(signal.pair)
                fees = self.fee_structures.get(exchange, self.fee_structures['default'])
                
                # Calculate raw P&L
                gross_pnl = await self._calculate_gross_pnl(signal, performance, position_size)
                
                # Calculate fees
                total_fees = await self._calculate_total_fees(
                    signal, performance, position_size, fees, fee_type
                )
                
                # Calculate slippage
                slippage_cost = await self._calculate_slippage(signal, performance, position_size)
                
                # Net P&L
                net_pnl = gross_pnl - total_fees - slippage_cost
                
                # Realized vs Unrealized
                realized_pnl = net_pnl if performance.actual_exit else Decimal("0")
                unrealized_pnl = net_pnl if not performance.actual_exit else Decimal("0")
                
                # Calculate performance ratios
                ratios = await self._calculate_performance_ratios(
                    signal_id, net_pnl, position_size
                )
                
                # Create metrics object
                metrics = PnLMetrics(
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fees_paid=total_fees,
                    slippage_cost=slippage_cost,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    roi_percentage=(net_pnl / position_size * 100) if position_size else Decimal("0"),
                    sharpe_ratio=ratios.get('sharpe_ratio'),
                    sortino_ratio=ratios.get('sortino_ratio'),
                    max_drawdown=ratios.get('max_drawdown', Decimal("0")),
                    win_rate=ratios.get('win_rate', 0),
                    profit_factor=ratios.get('profit_factor', 0),
                    expectancy=ratios.get('expectancy', Decimal("0")),
                    risk_reward_ratio=ratios.get('risk_reward_ratio', 0),
                    recovery_factor=ratios.get('recovery_factor')
                )
                
                # Cache the metrics
                await self._cache_metrics(signal_id, metrics)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating P&L for signal {signal_id}: {str(e)}")
            return None
    
    async def calculate_portfolio_pnl(self, signal_ids: List[str],
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate aggregate P&L for a portfolio of signals
        
        Args:
            signal_ids: List of signal identifiers
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Dict containing portfolio metrics
        """
        try:
            portfolio_metrics = {
                'total_gross_pnl': Decimal("0"),
                'total_net_pnl': Decimal("0"),
                'total_fees': Decimal("0"),
                'total_slippage': Decimal("0"),
                'total_realized': Decimal("0"),
                'total_unrealized': Decimal("0"),
                'signals_count': len(signal_ids),
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
                'best_trade': {'signal_id': None, 'pnl': Decimal("-999999")},
                'worst_trade': {'signal_id': None, 'pnl': Decimal("999999")},
                'average_pnl': Decimal("0"),
                'portfolio_sharpe': 0,
                'portfolio_sortino': 0,
                'max_portfolio_drawdown': Decimal("0"),
                'overall_win_rate': 0,
                'overall_profit_factor': 0,
                'correlation_matrix': {}
            }
            
            # Calculate individual P&Ls
            individual_pnls = []
            for signal_id in signal_ids:
                metrics = await self.calculate_signal_pnl(signal_id)
                if metrics:
                    individual_pnls.append({
                        'signal_id': signal_id,
                        'metrics': metrics
                    })
                    
                    # Aggregate totals
                    portfolio_metrics['total_gross_pnl'] += metrics.gross_pnl
                    portfolio_metrics['total_net_pnl'] += metrics.net_pnl
                    portfolio_metrics['total_fees'] += metrics.fees_paid
                    portfolio_metrics['total_slippage'] += metrics.slippage_cost
                    portfolio_metrics['total_realized'] += metrics.realized_pnl
                    portfolio_metrics['total_unrealized'] += metrics.unrealized_pnl
                    
                    # Track winners/losers
                    if metrics.net_pnl > 0:
                        portfolio_metrics['winning_trades'] += 1
                        if metrics.net_pnl > portfolio_metrics['best_trade']['pnl']:
                            portfolio_metrics['best_trade'] = {
                                'signal_id': signal_id,
                                'pnl': metrics.net_pnl
                            }
                    elif metrics.net_pnl < 0:
                        portfolio_metrics['losing_trades'] += 1
                        if metrics.net_pnl < portfolio_metrics['worst_trade']['pnl']:
                            portfolio_metrics['worst_trade'] = {
                                'signal_id': signal_id,
                                'pnl': metrics.net_pnl
                            }
                    else:
                        portfolio_metrics['breakeven_trades'] += 1
            
            # Calculate portfolio-level metrics
            if individual_pnls:
                portfolio_metrics['average_pnl'] = (
                    portfolio_metrics['total_net_pnl'] / len(individual_pnls)
                )
                
                # Win rate
                total_trades = (portfolio_metrics['winning_trades'] + 
                              portfolio_metrics['losing_trades'])
                if total_trades > 0:
                    portfolio_metrics['overall_win_rate'] = (
                        portfolio_metrics['winning_trades'] / total_trades
                    )
                
                # Profit factor
                total_wins = sum(m['metrics'].net_pnl for m in individual_pnls 
                               if m['metrics'].net_pnl > 0)
                total_losses = abs(sum(m['metrics'].net_pnl for m in individual_pnls 
                                     if m['metrics'].net_pnl < 0))
                if total_losses > 0:
                    portfolio_metrics['overall_profit_factor'] = float(total_wins / total_losses)
                
                # Calculate portfolio Sharpe and Sortino
                returns = [float(m['metrics'].roi_percentage) for m in individual_pnls]
                if len(returns) > 1:
                    portfolio_metrics['portfolio_sharpe'] = self._calculate_sharpe_ratio(returns)
                    portfolio_metrics['portfolio_sortino'] = self._calculate_sortino_ratio(returns)
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio P&L: {str(e)}")
            return {}
    
    # P&L Calculation Methods
    async def _calculate_gross_pnl(self, signal: Signal, performance: SignalPerformance,
                                   position_size: Decimal) -> Decimal:
        """Calculate gross P&L before fees and slippage"""
        if not performance.actual_entry:
            return Decimal("0")
        
        entry_price = Decimal(str(performance.actual_entry))
        exit_price = Decimal(str(performance.actual_exit)) if performance.actual_exit else Decimal(str(signal.entry_price))
        
        if signal.direction == 'long':
            price_diff = exit_price - entry_price
        else:  # short
            price_diff = entry_price - exit_price
        
        gross_pnl = (price_diff / entry_price) * position_size
        return gross_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    async def _calculate_total_fees(self, signal: Signal, performance: SignalPerformance,
                                    position_size: Decimal, fees: TradingFees,
                                    fee_type: FeeStructure) -> Decimal:
        """Calculate total trading fees"""
        fee_rate = fees.get_fee(fee_type)
        
        # Entry fees
        entry_fees = position_size * fee_rate
        
        # Exit fees (if position closed)
        exit_fees = Decimal("0")
        if performance.actual_exit:
            # Calculate exit position value
            if signal.direction == 'long':
                exit_value = position_size * (Decimal(str(performance.actual_exit)) / Decimal(str(performance.actual_entry)))
            else:  # short
                exit_value = position_size * (Decimal(str(performance.actual_entry)) / Decimal(str(performance.actual_exit)))
            exit_fees = exit_value * fee_rate
        
        total_fees = entry_fees + exit_fees
        return total_fees.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    async def _calculate_slippage(self, signal: Signal, performance: SignalPerformance,
                                  position_size: Decimal) -> Decimal:
        """Calculate slippage costs"""
        if not performance.actual_entry:
            return Decimal("0")
        
        # Entry slippage
        planned_entry = Decimal(str(signal.entry_price))
        actual_entry = Decimal(str(performance.actual_entry))
        
        if signal.direction == 'long':
            entry_slippage = max(actual_entry - planned_entry, Decimal("0"))
        else:  # short
            entry_slippage = max(planned_entry - actual_entry, Decimal("0"))
        
        entry_slippage_cost = (entry_slippage / planned_entry) * position_size
        
        # Exit slippage (if applicable)
        exit_slippage_cost = Decimal("0")
        if performance.actual_exit and signal.take_profits:
            # Assume first take profit as planned exit
            planned_exit = Decimal(str(signal.take_profits[0].get('price', signal.entry_price)))
            actual_exit = Decimal(str(performance.actual_exit))
            
            if signal.direction == 'long':
                exit_slippage = max(planned_exit - actual_exit, Decimal("0"))
            else:  # short
                exit_slippage = max(actual_exit - planned_exit, Decimal("0"))
            
            exit_slippage_cost = (exit_slippage / planned_exit) * position_size
        
        total_slippage = entry_slippage_cost + exit_slippage_cost
        return total_slippage.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    async def _calculate_performance_ratios(self, signal_id: str, net_pnl: Decimal,
                                           position_size: Decimal) -> Dict[str, Any]:
        """Calculate various performance ratios"""
        ratios = {}
        
        try:
            async with get_async_session() as session:
                # Get all trades for calculating ratios
                all_trades = await self._get_recent_trades(session, limit=100)
                
                if all_trades:
                    # Win rate
                    wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
                    ratios['win_rate'] = wins / len(all_trades)
                    
                    # Profit factor
                    gross_wins = sum(t.get('pnl', 0) for t in all_trades if t.get('pnl', 0) > 0)
                    gross_losses = abs(sum(t.get('pnl', 0) for t in all_trades if t.get('pnl', 0) < 0))
                    ratios['profit_factor'] = gross_wins / gross_losses if gross_losses > 0 else float('inf')
                    
                    # Expectancy
                    ratios['expectancy'] = Decimal(str(sum(t.get('pnl', 0) for t in all_trades) / len(all_trades)))
                    
                    # Max drawdown
                    ratios['max_drawdown'] = self._calculate_max_drawdown(all_trades)
                    
                    # Risk/Reward ratio
                    avg_win = gross_wins / wins if wins > 0 else 0
                    losses = len(all_trades) - wins
                    avg_loss = abs(gross_losses / losses) if losses > 0 else 0
                    ratios['risk_reward_ratio'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
                    
                    # Sharpe and Sortino ratios (if enough data)
                    if len(all_trades) >= 20:
                        returns = [t.get('roi', 0) for t in all_trades]
                        ratios['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
                        ratios['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                
                return ratios
                
        except Exception as e:
            logger.error(f"Error calculating performance ratios: {str(e)}")
            return ratios
    
    # Statistical Calculation Methods
    def _calculate_sharpe_ratio(self, returns: List[float], periods: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio
        sharpe = (mean_return - float(self.risk_free_rate) / periods) / std_return
        return sharpe * np.sqrt(periods)
    
    def _calculate_sortino_ratio(self, returns: List[float], periods: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Downside returns only
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0
        
        sortino = (mean_return - float(self.risk_free_rate) / periods) / downside_std
        return sortino * np.sqrt(periods)
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> Decimal:
        """Calculate maximum drawdown"""
        if not trades:
            return Decimal("0")
        
        cumulative_returns = []
        cumsum = 0
        for trade in trades:
            cumsum += trade.get('pnl', 0)
            cumulative_returns.append(cumsum)
        
        if not cumulative_returns:
            return Decimal("0")
        
        peak = cumulative_returns[0]
        max_dd = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
        
        return Decimal(str(max_dd * 100))  # Return as percentage
    
    # Helper Methods
    def _extract_exchange_from_pair(self, pair: str) -> str:
        """Extract exchange name from trading pair"""
        # Simple heuristic - can be improved
        pair_lower = pair.lower()
        if 'binance' in pair_lower:
            return 'binance'
        elif 'kucoin' in pair_lower:
            return 'kucoin'
        elif 'coinbase' in pair_lower:
            return 'coinbase'
        return 'default'
    
    async def _get_signal(self, session: AsyncSession, signal_id: str) -> Optional[Signal]:
        """Get signal from database"""
        result = await session.execute(
            select(Signal).where(Signal.id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_performance(self, session: AsyncSession, signal_id: str) -> Optional[SignalPerformance]:
        """Get performance record from database"""
        result = await session.execute(
            select(SignalPerformance).where(SignalPerformance.signal_id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_recent_trades(self, session: AsyncSession, limit: int = 100) -> List[Dict]:
        """Get recent trades for ratio calculations"""
        # Query recent completed trades
        result = await session.execute(
            select(SignalPerformance)
            .where(SignalPerformance.closed_at.isnot(None))
            .order_by(SignalPerformance.closed_at.desc())
            .limit(limit)
        )
        
        trades = []
        for perf in result.scalars():
            trades.append({
                'pnl': float(perf.pnl_amount) if perf.pnl_amount else 0,
                'roi': float(perf.pnl_percentage) if perf.pnl_percentage else 0,
                'closed_at': perf.closed_at
            })
        
        return trades
    
    async def _cache_metrics(self, signal_id: str, metrics: PnLMetrics):
        """Cache calculated metrics"""
        cache_data = {
            'gross_pnl': str(metrics.gross_pnl),
            'net_pnl': str(metrics.net_pnl),
            'fees_paid': str(metrics.fees_paid),
            'slippage_cost': str(metrics.slippage_cost),
            'realized_pnl': str(metrics.realized_pnl),
            'unrealized_pnl': str(metrics.unrealized_pnl),
            'roi_percentage': str(metrics.roi_percentage),
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': str(metrics.max_drawdown),
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'expectancy': str(metrics.expectancy),
            'risk_reward_ratio': metrics.risk_reward_ratio,
            'recovery_factor': metrics.recovery_factor,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            f"pnl_metrics:{signal_id}",
            3600,  # 1 hour cache
            json.dumps(cache_data, default=str)
        )
