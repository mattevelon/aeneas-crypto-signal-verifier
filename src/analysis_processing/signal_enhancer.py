"""Signal Enhancement Engine for optimizing trading signals."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with optimizations."""
    original_params: Dict[str, Any]
    enhanced_params: Dict[str, Any]
    adjustments: Dict[str, Any]
    enhancement_score: float
    risk_adjustments: Dict[str, Any]
    execution_strategy: Dict[str, Any]
    metadata: Dict[str, Any]


class SignalEnhancer:
    """
    Enhances validated signals with AI-suggested optimizations.
    Implements smart order routing and execution strategies.
    """
    
    def __init__(self):
        """Initialize signal enhancer."""
        self.enhancement_strategies = self._initialize_strategies()
        self.risk_parameters = {
            'max_position_size': 0.1,  # 10% max per position
            'max_leverage': 10,
            'min_risk_reward': 1.5,
            'max_slippage': 0.01  # 1% max slippage
        }
        
        # Enhancement statistics
        self.enhancement_stats = {
            'total_enhanced': 0,
            'avg_improvement': 0,
            'optimization_types': {}
        }
        
        logger.info("Initialized SignalEnhancer")
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize enhancement strategies."""
        return {
            'entry_optimization': self._optimize_entry,
            'stop_loss_optimization': self._optimize_stop_loss,
            'take_profit_optimization': self._optimize_take_profits,
            'position_sizing': self._optimize_position_size,
            'execution_timing': self._optimize_execution_timing,
            'order_type_selection': self._select_order_types
        }
    
    def enhance_signal(self,
                       signal_data: Dict[str, Any],
                       ai_analysis: Dict[str, Any],
                       validation_result: Any,
                       context: Dict[str, Any]) -> EnhancedSignal:
        """
        Enhance a validated signal with optimizations.
        
        Args:
            signal_data: Original signal parameters
            ai_analysis: AI analysis results
            validation_result: Validation framework results
            context: Full context data
            
        Returns:
            Enhanced signal with optimizations
        """
        original_params = signal_data.get('trading_params', {}).copy()
        enhanced_params = original_params.copy()
        adjustments = {}
        
        # Apply AI-suggested optimizations
        ai_optimizations = ai_analysis.get('analysis', {}).get('optimizations', {})
        if ai_optimizations:
            enhanced_params, ai_adjustments = self._apply_ai_optimizations(
                enhanced_params, ai_optimizations
            )
            adjustments.update(ai_adjustments)
        
        # Apply enhancement strategies
        for strategy_name, strategy_func in self.enhancement_strategies.items():
            try:
                enhanced_params, strategy_adjustments = strategy_func(
                    enhanced_params, context, validation_result
                )
                if strategy_adjustments:
                    adjustments[strategy_name] = strategy_adjustments
            except Exception as e:
                logger.error(f"Enhancement strategy {strategy_name} failed: {e}")
        
        # Calculate enhancement score
        enhancement_score = self._calculate_enhancement_score(
            original_params, enhanced_params, adjustments
        )
        
        # Generate risk adjustments
        risk_adjustments = self._generate_risk_adjustments(
            enhanced_params, context, ai_analysis
        )
        
        # Generate execution strategy
        execution_strategy = self._generate_execution_strategy(
            enhanced_params, context
        )
        
        # Update statistics
        self._update_statistics(enhancement_score, adjustments)
        
        return EnhancedSignal(
            original_params=original_params,
            enhanced_params=enhanced_params,
            adjustments=adjustments,
            enhancement_score=enhancement_score,
            risk_adjustments=risk_adjustments,
            execution_strategy=execution_strategy,
            metadata={
                'enhanced_at': datetime.now().isoformat(),
                'strategies_applied': list(adjustments.keys()),
                'ai_optimizations_used': bool(ai_optimizations)
            }
        )
    
    def _apply_ai_optimizations(self,
                               params: Dict[str, Any],
                               optimizations: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Apply AI-suggested optimizations."""
        enhanced = params.copy()
        adjustments = {}
        
        # Apply entry price optimization
        if 'entry_price' in optimizations and optimizations['entry_price']:
            original_entry = enhanced['entry_price']
            suggested_entry = optimizations['entry_price']
            
            # Validate suggestion is reasonable
            if abs(suggested_entry - original_entry) / original_entry < 0.02:  # Within 2%
                enhanced['entry_price'] = suggested_entry
                adjustments['entry_price'] = {
                    'original': original_entry,
                    'optimized': suggested_entry,
                    'improvement': f"{(original_entry - suggested_entry) / original_entry * 100:.2f}%"
                }
        
        # Apply stop loss optimization
        if 'stop_loss' in optimizations and optimizations['stop_loss']:
            original_sl = enhanced['stop_loss']
            suggested_sl = optimizations['stop_loss']
            
            # Validate stop loss is on correct side
            if enhanced['direction'] == 'long' and suggested_sl < enhanced['entry_price']:
                enhanced['stop_loss'] = suggested_sl
                adjustments['stop_loss'] = {
                    'original': original_sl,
                    'optimized': suggested_sl
                }
            elif enhanced['direction'] == 'short' and suggested_sl > enhanced['entry_price']:
                enhanced['stop_loss'] = suggested_sl
                adjustments['stop_loss'] = {
                    'original': original_sl,
                    'optimized': suggested_sl
                }
        
        # Apply take profit optimization
        if 'take_profits' in optimizations and optimizations['take_profits']:
            enhanced['take_profits'] = optimizations['take_profits']
            adjustments['take_profits'] = {
                'original': params.get('take_profits', []),
                'optimized': optimizations['take_profits']
            }
        
        return enhanced, adjustments
    
    def _optimize_entry(self,
                       params: Dict[str, Any],
                       context: Dict[str, Any],
                       validation: Any) -> Tuple[Dict, Dict]:
        """Optimize entry price based on market conditions."""
        adjustments = {}
        
        # Get market data
        market = context.get('components', {}).get('market', {})
        technical = context.get('components', {}).get('technical', {})
        
        # Check for better entry based on support/resistance
        support_resistance = technical.get('support_resistance', {})
        current_price = market.get('current_price', params['entry_price'])
        
        if params['direction'] == 'long':
            # Look for support level near entry
            supports = support_resistance.get('support', [])
            if supports:
                nearest_support = min(supports, key=lambda x: abs(x - current_price))
                if abs(nearest_support - current_price) / current_price < 0.01:  # Within 1%
                    params['entry_price'] = nearest_support
                    adjustments['support_entry'] = nearest_support
        else:  # short
            # Look for resistance level near entry
            resistances = support_resistance.get('resistance', [])
            if resistances:
                nearest_resistance = min(resistances, key=lambda x: abs(x - current_price))
                if abs(nearest_resistance - current_price) / current_price < 0.01:  # Within 1%
                    params['entry_price'] = nearest_resistance
                    adjustments['resistance_entry'] = nearest_resistance
        
        return params, adjustments
    
    def _optimize_stop_loss(self,
                           params: Dict[str, Any],
                           context: Dict[str, Any],
                           validation: Any) -> Tuple[Dict, Dict]:
        """Optimize stop loss placement."""
        adjustments = {}
        
        # Get technical indicators
        technical = context.get('components', {}).get('technical', {})
        atr = technical.get('indicators', {}).get('atr', 0)
        
        if atr > 0:
            # Use ATR-based stop loss
            entry = params['entry_price']
            
            if params['direction'] == 'long':
                atr_stop = entry - (atr * 2)  # 2x ATR stop
                if atr_stop > params['stop_loss']:
                    params['stop_loss'] = atr_stop
                    adjustments['atr_based'] = atr_stop
            else:  # short
                atr_stop = entry + (atr * 2)
                if atr_stop < params['stop_loss']:
                    params['stop_loss'] = atr_stop
                    adjustments['atr_based'] = atr_stop
        
        return params, adjustments
    
    def _optimize_take_profits(self,
                              params: Dict[str, Any],
                              context: Dict[str, Any],
                              validation: Any) -> Tuple[Dict, Dict]:
        """Optimize take profit levels."""
        adjustments = {}
        
        # Get technical data
        technical = context.get('components', {}).get('technical', {})
        support_resistance = technical.get('support_resistance', {})
        
        # Adjust take profits to key levels
        if params['direction'] == 'long':
            resistances = support_resistance.get('resistance', [])
            if resistances:
                # Set take profits at resistance levels
                entry = params['entry_price']
                valid_resistances = [r for r in resistances if r > entry]
                if valid_resistances:
                    params['take_profits'] = sorted(valid_resistances)[:3]  # Max 3 targets
                    adjustments['resistance_targets'] = params['take_profits']
        else:  # short
            supports = support_resistance.get('support', [])
            if supports:
                entry = params['entry_price']
                valid_supports = [s for s in supports if s < entry]
                if valid_supports:
                    params['take_profits'] = sorted(valid_supports, reverse=True)[:3]
                    adjustments['support_targets'] = params['take_profits']
        
        return params, adjustments
    
    def _optimize_position_size(self,
                               params: Dict[str, Any],
                               context: Dict[str, Any],
                               validation: Any) -> Tuple[Dict, Dict]:
        """Optimize position sizing using Kelly Criterion."""
        adjustments = {}
        
        # Get historical performance
        historical = context.get('components', {}).get('historical', {})
        perf = historical.get('performance_metrics', {})
        win_rate = perf.get('win_rate', 0.5)
        avg_win = abs(perf.get('avg_pnl', 2))  # Average win percentage
        
        # Calculate Kelly percentage
        if win_rate > 0 and avg_win > 0:
            avg_loss = avg_win / 2  # Assume 2:1 reward/risk
            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply Kelly fraction (25% of full Kelly for safety)
            position_size = min(kelly_pct * 0.25, self.risk_parameters['max_position_size'])
            
            if position_size > 0:
                params['position_size'] = round(position_size, 4)
                adjustments['kelly_criterion'] = position_size
        
        return params, adjustments
    
    def _optimize_execution_timing(self,
                                  params: Dict[str, Any],
                                  context: Dict[str, Any],
                                  validation: Any) -> Tuple[Dict, Dict]:
        """Optimize execution timing based on market conditions."""
        adjustments = {}
        
        # Get market data
        market = context.get('components', {}).get('market', {})
        volatility = market.get('volatility', 0)
        
        # Determine optimal execution timing
        if volatility > 75:
            adjustments['timing'] = 'scale_in'  # Scale into position during high volatility
            adjustments['scaling_strategy'] = {
                'orders': 3,
                'interval_minutes': 15
            }
        else:
            adjustments['timing'] = 'immediate'
        
        return params, adjustments
    
    def _select_order_types(self,
                           params: Dict[str, Any],
                           context: Dict[str, Any],
                           validation: Any) -> Tuple[Dict, Dict]:
        """Select optimal order types for execution."""
        adjustments = {}
        
        # Get market conditions
        market = context.get('components', {}).get('market', {})
        spread_pct = market.get('bid_ask', {}).get('spread_percentage', 0)
        liquidity_score = market.get('liquidity_score', 50)
        
        # Determine order types
        order_types = {
            'entry': 'limit' if spread_pct > 0.1 else 'market',
            'stop_loss': 'stop_limit',
            'take_profit': 'limit'
        }
        
        # Add iceberg orders for large positions
        if params.get('position_size', 0) > 0.05 and liquidity_score < 70:
            order_types['execution_type'] = 'iceberg'
            order_types['iceberg_size'] = 0.01  # 1% chunks
        
        params['order_types'] = order_types
        adjustments['order_strategy'] = order_types
        
        return params, adjustments
    
    def _calculate_enhancement_score(self,
                                    original: Dict[str, Any],
                                    enhanced: Dict[str, Any],
                                    adjustments: Dict[str, Any]) -> float:
        """Calculate enhancement score."""
        score = 50.0  # Base score
        
        # Check if entry was optimized
        if original.get('entry_price') != enhanced.get('entry_price'):
            score += 10
        
        # Check if stop loss was improved
        if original.get('stop_loss') != enhanced.get('stop_loss'):
            entry = enhanced['entry_price']
            original_risk = abs(entry - original['stop_loss']) / entry
            enhanced_risk = abs(entry - enhanced['stop_loss']) / entry
            
            if enhanced_risk < original_risk:
                score += 15  # Reduced risk
        
        # Check if take profits were optimized
        if original.get('take_profits') != enhanced.get('take_profits'):
            score += 10
        
        # Check if position sizing was added
        if 'position_size' in enhanced and 'position_size' not in original:
            score += 15
        
        # Check if execution strategy was added
        if 'order_types' in enhanced:
            score += 10
        
        return min(100, score)
    
    def _generate_risk_adjustments(self,
                                  params: Dict[str, Any],
                                  context: Dict[str, Any],
                                  ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management adjustments."""
        risk_level = ai_analysis.get('verdict', {}).get('risk_level', 'medium')
        
        adjustments = {
            'max_loss_percentage': 2 if risk_level == 'low' else 1,
            'trailing_stop_activation': 1.5,  # Activate at 1.5% profit
            'trailing_stop_distance': 0.5,    # 0.5% trailing distance
            'partial_take_profits': {
                'tp1_percentage': 50,  # Take 50% at TP1
                'tp2_percentage': 30,  # Take 30% at TP2
                'tp3_percentage': 20   # Take 20% at TP3
            }
        }
        
        # Adjust for high risk signals
        if risk_level == 'high':
            adjustments['reduce_position_by'] = 0.5  # Halve position size
            adjustments['tighten_stop_by'] = 0.5     # Tighten stop loss
        
        return adjustments
    
    def _generate_execution_strategy(self,
                                    params: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete execution strategy."""
        market = context.get('components', {}).get('market', {})
        
        strategy = {
            'order_routing': 'smart',  # Smart order routing
            'execution_algo': 'twap' if params.get('position_size', 0) > 0.05 else 'immediate',
            'slippage_tolerance': self.risk_parameters['max_slippage'],
            'retry_strategy': {
                'max_retries': 3,
                'retry_delay_seconds': 5
            },
            'monitoring': {
                'check_interval_seconds': 60,
                'alert_on_deviation': 0.02  # Alert if 2% deviation
            }
        }
        
        # Add specific instructions based on market conditions
        if market.get('volatility', 0) > 75:
            strategy['volatility_adjustments'] = {
                'widen_stops': True,
                'reduce_size': True,
                'use_limits': True
            }
        
        return strategy
    
    def _update_statistics(self, enhancement_score: float, adjustments: Dict[str, Any]):
        """Update enhancement statistics."""
        self.enhancement_stats['total_enhanced'] += 1
        
        # Update average improvement
        n = self.enhancement_stats['total_enhanced']
        prev_avg = self.enhancement_stats['avg_improvement']
        self.enhancement_stats['avg_improvement'] = ((prev_avg * (n - 1)) + enhancement_score) / n
        
        # Track optimization types
        for opt_type in adjustments.keys():
            if opt_type not in self.enhancement_stats['optimization_types']:
                self.enhancement_stats['optimization_types'][opt_type] = 0
            self.enhancement_stats['optimization_types'][opt_type] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return {
            'total_enhanced': self.enhancement_stats['total_enhanced'],
            'avg_improvement_score': round(self.enhancement_stats['avg_improvement'], 2),
            'optimization_types': self.enhancement_stats['optimization_types'],
            'most_common_optimization': (
                max(self.enhancement_stats['optimization_types'],
                    key=self.enhancement_stats['optimization_types'].get)
                if self.enhancement_stats['optimization_types'] else None
            )
        }
