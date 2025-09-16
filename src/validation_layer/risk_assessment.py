"""
Risk Assessment Module

Implements comprehensive risk evaluation including Kelly Criterion for position sizing,
Value at Risk (VaR) calculation, and maximum drawdown estimation.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats

from src.config.settings import get_settings
from src.core.database import get_async_session
from sqlalchemy import select, func, and_

logger = logging.getLogger(__name__)
settings = get_settings()


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """Comprehensive risk assessment metrics."""
    position_size_kelly: float  # Kelly Criterion optimal position size
    position_size_recommended: float  # Adjusted for practical constraints
    value_at_risk_95: float  # 95% confidence VaR
    value_at_risk_99: float  # 99% confidence VaR
    max_drawdown_expected: float  # Expected maximum drawdown
    risk_reward_ratio: float  # Risk/reward ratio
    sharpe_ratio: float  # Risk-adjusted return metric
    risk_level: RiskLevel
    confidence_score: float  # Overall risk confidence (0-100)
    warnings: List[str]
    recommendations: List[str]


class RiskAssessmentModule:
    """Advanced risk assessment and position sizing calculations."""
    
    # Risk parameters
    MIN_RISK_REWARD_RATIO = 1.5  # Minimum 1:1.5 risk/reward
    MAX_POSITION_SIZE = 0.10  # Maximum 10% of capital per trade
    MIN_WIN_RATE = 0.40  # Minimum 40% win rate for Kelly
    RISK_FREE_RATE = 0.03  # 3% annual risk-free rate
    CONFIDENCE_THRESHOLD = 0.95  # 95% confidence level for VaR
    
    # Kelly Criterion adjustments
    KELLY_FRACTION = 0.25  # Use 25% of Kelly for safety (fractional Kelly)
    
    def __init__(self):
        """Initialize risk assessment module."""
        self.historical_data = {}
        self.performance_cache = {}
        
    async def assess_risk(
        self,
        signal: Dict[str, Any],
        account_balance: float,
        historical_performance: Optional[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """
        Perform comprehensive risk assessment for a trading signal.
        
        Args:
            signal: Trading signal data
            account_balance: Current account balance
            historical_performance: Historical performance data
            
        Returns:
            RiskMetrics with comprehensive risk assessment
        """
        warnings = []
        recommendations = []
        
        # Extract signal parameters
        entry_price = float(signal.get('entry_price', 0))
        stop_loss = float(signal.get('stop_loss', 0))
        take_profits = signal.get('take_profits', [])
        confidence = float(signal.get('confidence_score', 50)) / 100
        
        # Calculate basic risk metrics
        risk_amount = abs(entry_price - stop_loss)
        risk_percent = risk_amount / entry_price
        
        # Calculate average take profit
        if take_profits:
            avg_tp = np.mean([float(tp) for tp in take_profits])
            reward_amount = abs(avg_tp - entry_price)
            reward_percent = reward_amount / entry_price
        else:
            reward_amount = risk_amount * 2  # Default 2:1 reward
            reward_percent = risk_percent * 2
            
        # Risk/Reward ratio
        risk_reward_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        if risk_reward_ratio < self.MIN_RISK_REWARD_RATIO:
            warnings.append(f"Poor risk/reward ratio: {risk_reward_ratio:.2f}")
            recommendations.append(f"Consider adjusting TP to achieve minimum {self.MIN_RISK_REWARD_RATIO}:1")
        
        # Get historical win rate and metrics
        if historical_performance:
            win_rate = historical_performance.get('win_rate', 0.5)
            avg_win = historical_performance.get('avg_win_percent', reward_percent)
            avg_loss = historical_performance.get('avg_loss_percent', risk_percent)
            historical_returns = historical_performance.get('returns', [])
        else:
            # Use default estimates
            win_rate = 0.5 * confidence  # Adjust by confidence
            avg_win = reward_percent
            avg_loss = risk_percent
            historical_returns = []
            
        # Kelly Criterion calculation
        kelly_position = await self._calculate_kelly_criterion(
            win_rate, avg_win, avg_loss
        )
        
        # Practical position size (considering constraints)
        recommended_position = await self._calculate_recommended_position(
            kelly_position, risk_percent, account_balance, confidence
        )
        
        # Value at Risk calculation
        var_95, var_99 = await self._calculate_value_at_risk(
            recommended_position * account_balance,
            historical_returns if historical_returns else [risk_percent, -reward_percent],
            confidence
        )
        
        # Maximum drawdown estimation
        max_drawdown = await self._estimate_max_drawdown(
            historical_returns,
            risk_percent,
            win_rate
        )
        
        # Sharpe ratio calculation
        sharpe_ratio = await self._calculate_sharpe_ratio(
            historical_returns if historical_returns else [reward_percent * win_rate - risk_percent * (1-win_rate)],
            self.RISK_FREE_RATE
        )
        
        # Determine risk level
        risk_level = self._classify_risk_level(
            risk_reward_ratio, kelly_position, max_drawdown, confidence
        )
        
        # Calculate overall confidence score
        risk_confidence = self._calculate_risk_confidence(
            risk_reward_ratio, win_rate, confidence, sharpe_ratio
        )
        
        # Add recommendations based on analysis
        if kelly_position < 0:
            warnings.append("Negative Kelly: Signal has negative expected value")
            recommendations.append("Avoid this trade or reduce position significantly")
        
        if max_drawdown > 0.20:
            warnings.append(f"High expected drawdown: {max_drawdown:.1%}")
            recommendations.append("Consider reducing position size to limit drawdown")
            
        if sharpe_ratio < 0.5:
            warnings.append(f"Low Sharpe ratio: {sharpe_ratio:.2f}")
            recommendations.append("Risk-adjusted returns are below acceptable threshold")
            
        return RiskMetrics(
            position_size_kelly=kelly_position,
            position_size_recommended=recommended_position,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            max_drawdown_expected=max_drawdown,
            risk_reward_ratio=risk_reward_ratio,
            sharpe_ratio=sharpe_ratio,
            risk_level=risk_level,
            confidence_score=risk_confidence * 100,
            warnings=warnings,
            recommendations=recommendations
        )
    
    async def _calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly formula: f = (p*b - q) / b
        where:
        - f = fraction of capital to wager
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = ratio of win to loss
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            
        Returns:
            Optimal position size as fraction of capital
        """
        if avg_loss == 0 or win_rate < self.MIN_WIN_RATE:
            return 0.0
            
        # Calculate odds (b)
        b = avg_win / avg_loss
        
        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly for safety
        kelly = kelly * self.KELLY_FRACTION
        
        # Ensure non-negative and within bounds
        kelly = max(0, min(kelly, self.MAX_POSITION_SIZE))
        
        logger.debug(f"Kelly calculation: win_rate={win_rate:.2%}, b={b:.2f}, kelly={kelly:.4f}")
        
        return kelly
    
    async def _calculate_recommended_position(
        self,
        kelly_position: float,
        risk_percent: float,
        account_balance: float,
        confidence: float
    ) -> float:
        """
        Calculate practical position size considering all constraints.
        
        Args:
            kelly_position: Kelly Criterion optimal position
            risk_percent: Risk per trade as percentage
            account_balance: Current account balance
            confidence: Signal confidence (0-1)
            
        Returns:
            Recommended position size as fraction of capital
        """
        # Start with Kelly position
        position = kelly_position
        
        # Adjust by confidence
        position = position * confidence
        
        # Apply maximum position size constraint
        position = min(position, self.MAX_POSITION_SIZE)
        
        # Apply risk-based constraint (max 2% risk per trade)
        max_risk_position = 0.02 / risk_percent if risk_percent > 0 else self.MAX_POSITION_SIZE
        position = min(position, max_risk_position)
        
        # Ensure minimum viable position (0.5% of capital)
        min_position = 0.005
        if position < min_position and kelly_position > 0:
            position = min_position
            
        return position
    
    async def _calculate_value_at_risk(
        self,
        position_value: float,
        historical_returns: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) at different confidence levels.
        
        Args:
            position_value: Value of the position
            historical_returns: Historical return percentages
            confidence_level: Confidence level for VaR
            
        Returns:
            Tuple of (VaR_95, VaR_99)
        """
        if not historical_returns or len(historical_returns) < 2:
            # Use default risk estimate
            return position_value * 0.05, position_value * 0.10
            
        returns = np.array(historical_returns)
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0, 0
            
        # Calculate VaR using parametric method (assuming normal distribution)
        # VaR_95 = mean - 1.65 * std
        # VaR_99 = mean - 2.33 * std
        var_95 = position_value * (mean_return - 1.65 * std_return)
        var_99 = position_value * (mean_return - 2.33 * std_return)
        
        # Ensure VaR is negative (represents loss)
        var_95 = min(0, var_95)
        var_99 = min(0, var_99)
        
        return abs(var_95), abs(var_99)
    
    async def _estimate_max_drawdown(
        self,
        historical_returns: List[float],
        risk_per_trade: float,
        win_rate: float
    ) -> float:
        """
        Estimate maximum drawdown based on historical data or statistical model.
        
        Args:
            historical_returns: Historical return percentages
            risk_per_trade: Risk percentage per trade
            win_rate: Historical win rate
            
        Returns:
            Expected maximum drawdown as percentage
        """
        if historical_returns and len(historical_returns) > 10:
            # Calculate actual max drawdown from historical data
            cumulative = np.cumprod(1 + np.array(historical_returns) / 100)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(np.min(drawdown))
            return max_dd
        else:
            # Estimate using statistical model
            # Simplified model: max_dd ≈ risk * sqrt(losing_streak)
            # Expected losing streak = -log(0.01) / log(1 - win_rate)
            if win_rate >= 0.99:
                losing_streak = 10
            else:
                losing_streak = -np.log(0.01) / np.log(1 - win_rate)
                
            max_dd = risk_per_trade * np.sqrt(losing_streak)
            return min(max_dd, 0.50)  # Cap at 50% drawdown
    
    async def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.03
    ) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted return assessment.
        
        Args:
            returns: List of return percentages
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns) / 100  # Convert to decimal
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0.0
            
        sharpe = mean_excess / std_excess * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def _classify_risk_level(
        self,
        risk_reward_ratio: float,
        kelly_position: float,
        max_drawdown: float,
        confidence: float
    ) -> RiskLevel:
        """
        Classify overall risk level based on multiple factors.
        
        Args:
            risk_reward_ratio: Risk/reward ratio
            kelly_position: Kelly Criterion position size
            max_drawdown: Expected maximum drawdown
            confidence: Signal confidence
            
        Returns:
            RiskLevel classification
        """
        risk_score = 0
        
        # Risk/reward contribution
        if risk_reward_ratio < 1:
            risk_score += 3
        elif risk_reward_ratio < 1.5:
            risk_score += 2
        elif risk_reward_ratio < 2:
            risk_score += 1
            
        # Kelly position contribution
        if kelly_position < 0:
            risk_score += 3
        elif kelly_position < 0.01:
            risk_score += 2
        elif kelly_position < 0.02:
            risk_score += 1
            
        # Drawdown contribution
        if max_drawdown > 0.30:
            risk_score += 3
        elif max_drawdown > 0.20:
            risk_score += 2
        elif max_drawdown > 0.10:
            risk_score += 1
            
        # Confidence contribution
        if confidence < 0.4:
            risk_score += 2
        elif confidence < 0.6:
            risk_score += 1
            
        # Classify based on total score
        if risk_score >= 8:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_risk_confidence(
        self,
        risk_reward_ratio: float,
        win_rate: float,
        signal_confidence: float,
        sharpe_ratio: float
    ) -> float:
        """
        Calculate overall risk assessment confidence score.
        
        Args:
            risk_reward_ratio: Risk/reward ratio
            win_rate: Historical win rate
            signal_confidence: Signal confidence
            sharpe_ratio: Sharpe ratio
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        # Risk/reward contribution (25%)
        rr_score = min(risk_reward_ratio / 3.0, 1.0) * 0.25
        confidence += rr_score
        
        # Win rate contribution (25%)
        wr_score = min(win_rate / 0.6, 1.0) * 0.25
        confidence += wr_score
        
        # Signal confidence contribution (25%)
        confidence += signal_confidence * 0.25
        
        # Sharpe ratio contribution (25%)
        sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1.0) * 0.25
        confidence += sharpe_score
        
        return min(max(confidence, 0), 1.0)
