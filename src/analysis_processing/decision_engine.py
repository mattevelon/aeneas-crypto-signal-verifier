"""Decision Engine for final trading decisions and action generation."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trading action types."""
    EXECUTE = "execute"
    MONITOR = "monitor"
    REJECT = "reject"
    PAPER_TRADE = "paper_trade"
    SCALE_IN = "scale_in"


class DecisionConfidence(Enum):
    """Decision confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class TradingDecision:
    """Final trading decision with execution instructions."""
    action: ActionType
    confidence: DecisionConfidence
    reasoning: Dict[str, str]
    execution_params: Dict[str, Any]
    risk_limits: Dict[str, Any]
    monitoring_rules: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DecisionEngine:
    """
    Final decision engine that processes all analysis results.
    Implements decision tree logic and generates actionable recommendations.
    """
    
    def __init__(self):
        """Initialize decision engine."""
        self.decision_thresholds = {
            'min_confidence': 70,
            'min_validation_score': 75,
            'min_enhancement_score': 60,
            'max_risk_score': 70
        }
        
        self.decision_weights = {
            'ai_analysis': 0.35,
            'validation': 0.30,
            'enhancement': 0.20,
            'market_conditions': 0.15
        }
        
        # Decision statistics
        self.decision_stats = {
            'total_decisions': 0,
            'execute': 0,
            'reject': 0,
            'monitor': 0,
            'paper_trade': 0
        }
        
        logger.info("Initialized DecisionEngine")
    
    def make_decision(self,
                      signal_data: Dict[str, Any],
                      ai_analysis: Dict[str, Any],
                      validation_result: Any,
                      enhanced_signal: Any,
                      context: Dict[str, Any]) -> TradingDecision:
        """
        Make final trading decision based on all analysis.
        
        Args:
            signal_data: Original signal data
            ai_analysis: AI analysis results
            validation_result: Validation framework results
            enhanced_signal: Enhanced signal with optimizations
            context: Full context data
            
        Returns:
            Final trading decision with execution instructions
        """
        # Calculate decision score
        decision_score = self._calculate_decision_score(
            ai_analysis, validation_result, enhanced_signal, context
        )
        
        # Determine action based on score and conditions
        action = self._determine_action(
            decision_score, ai_analysis, validation_result
        )
        
        # Determine confidence level
        confidence = self._determine_confidence(decision_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            action, decision_score, ai_analysis, validation_result
        )
        
        # Generate execution parameters
        execution_params = self._generate_execution_params(
            action, enhanced_signal, context
        )
        
        # Set risk limits
        risk_limits = self._set_risk_limits(
            enhanced_signal, ai_analysis, context
        )
        
        # Define monitoring rules
        monitoring_rules = self._define_monitoring_rules(
            action, enhanced_signal, context
        )
        
        # Generate alerts
        alerts = self._generate_alerts(
            ai_analysis, validation_result, context
        )
        
        # Build metadata
        metadata = self._build_metadata(
            decision_score, signal_data, ai_analysis
        )
        
        # Update statistics
        self._update_statistics(action)
        
        return TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            execution_params=execution_params,
            risk_limits=risk_limits,
            monitoring_rules=monitoring_rules,
            alerts=alerts,
            metadata=metadata
        )
    
    def _calculate_decision_score(self,
                                 ai_analysis: Dict[str, Any],
                                 validation_result: Any,
                                 enhanced_signal: Any,
                                 context: Dict[str, Any]) -> float:
        """Calculate weighted decision score."""
        scores = {}
        
        # AI analysis score
        ai_confidence = ai_analysis.get('verdict', {}).get('confidence_score', 0)
        scores['ai_analysis'] = ai_confidence
        
        # Validation score
        validation_score = validation_result.score if validation_result else 0
        scores['validation'] = validation_score
        
        # Enhancement score
        enhancement_score = enhanced_signal.enhancement_score if enhanced_signal else 50
        scores['enhancement'] = enhancement_score
        
        # Market conditions score
        market_score = self._calculate_market_score(context)
        scores['market_conditions'] = market_score
        
        # Calculate weighted total
        total_score = sum(
            score * self.decision_weights.get(key, 0.25)
            for key, score in scores.items()
        )
        
        return round(total_score, 2)
    
    def _calculate_market_score(self, context: Dict[str, Any]) -> float:
        """Calculate market conditions score."""
        score = 50.0  # Base score
        
        market = context.get('components', {}).get('market', {})
        technical = context.get('components', {}).get('technical', {})
        
        # Liquidity score
        liquidity = market.get('liquidity_score', 50)
        score += (liquidity - 50) * 0.3
        
        # Volatility adjustment
        volatility = market.get('volatility', 50)
        if volatility < 30:
            score += 10  # Low volatility is good
        elif volatility > 75:
            score -= 10  # High volatility is risky
        
        # Technical signal strength
        signal_strength = technical.get('signal_strength', 'neutral')
        strength_scores = {
            'strong_buy': 20,
            'buy': 10,
            'neutral': 0,
            'sell': -10,
            'strong_sell': -20
        }
        score += strength_scores.get(signal_strength, 0)
        
        # Cross-channel consensus
        cross_channel = context.get('components', {}).get('cross_channel', {})
        consensus = cross_channel.get('consensus_score', 50)
        score += (consensus - 50) * 0.2
        
        return max(0, min(100, score))
    
    def _determine_action(self,
                         decision_score: float,
                         ai_analysis: Dict[str, Any],
                         validation_result: Any) -> ActionType:
        """Determine trading action based on score and conditions."""
        # Check for critical failures
        if validation_result and validation_result.metadata.get('critical_failures'):
            return ActionType.REJECT
        
        # Check if signal is valid
        is_valid = ai_analysis.get('verdict', {}).get('is_valid', False)
        if not is_valid:
            return ActionType.REJECT
        
        # Determine action based on score
        if decision_score >= 80:
            return ActionType.EXECUTE
        elif decision_score >= 70:
            # Check risk level for high score but not excellent
            risk_level = ai_analysis.get('verdict', {}).get('risk_level', 'high')
            if risk_level == 'low':
                return ActionType.EXECUTE
            elif risk_level == 'medium':
                return ActionType.SCALE_IN
            else:
                return ActionType.PAPER_TRADE
        elif decision_score >= 60:
            return ActionType.PAPER_TRADE
        elif decision_score >= 50:
            return ActionType.MONITOR
        else:
            return ActionType.REJECT
    
    def _determine_confidence(self, decision_score: float) -> DecisionConfidence:
        """Determine confidence level based on score."""
        if decision_score >= 85:
            return DecisionConfidence.VERY_HIGH
        elif decision_score >= 75:
            return DecisionConfidence.HIGH
        elif decision_score >= 65:
            return DecisionConfidence.MEDIUM
        elif decision_score >= 55:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW
    
    def _generate_reasoning(self,
                           action: ActionType,
                           decision_score: float,
                           ai_analysis: Dict[str, Any],
                           validation_result: Any) -> Dict[str, str]:
        """Generate reasoning for the decision."""
        reasoning = {
            'primary': '',
            'supporting': [],
            'concerns': []
        }
        
        # Primary reasoning based on action
        if action == ActionType.EXECUTE:
            reasoning['primary'] = f"Strong signal with {decision_score:.1f}% confidence. All validation checks passed."
        elif action == ActionType.SCALE_IN:
            reasoning['primary'] = f"Good signal ({decision_score:.1f}%) but scaling in recommended due to market conditions."
        elif action == ActionType.PAPER_TRADE:
            reasoning['primary'] = f"Moderate signal ({decision_score:.1f}%). Paper trading recommended for validation."
        elif action == ActionType.MONITOR:
            reasoning['primary'] = f"Weak signal ({decision_score:.1f}%). Continue monitoring for better entry."
        else:  # REJECT
            reasoning['primary'] = f"Signal rejected due to low confidence ({decision_score:.1f}%) or critical failures."
        
        # Supporting reasons
        ai_confidence = ai_analysis.get('verdict', {}).get('confidence_score', 0)
        reasoning['supporting'].append(f"AI confidence: {ai_confidence}%")
        
        if validation_result:
            reasoning['supporting'].append(f"Validation score: {validation_result.score}%")
            reasoning['supporting'].append(f"{len(validation_result.checks_passed)} validation checks passed")
        
        # Concerns
        if validation_result and validation_result.warnings:
            reasoning['concerns'] = validation_result.warnings[:3]  # Top 3 concerns
        
        risk_factors = ai_analysis.get('analysis', {}).get('risk_factors', [])
        if risk_factors:
            reasoning['concerns'].extend(risk_factors[:2])
        
        return reasoning
    
    def _generate_execution_params(self,
                                  action: ActionType,
                                  enhanced_signal: Any,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution parameters based on action."""
        if action == ActionType.REJECT or action == ActionType.MONITOR:
            return {}
        
        params = enhanced_signal.enhanced_params.copy() if enhanced_signal else {}
        
        # Add execution-specific parameters
        execution = {
            'trading_params': params,
            'execution_type': 'live' if action == ActionType.EXECUTE else 'paper',
            'order_strategy': enhanced_signal.execution_strategy if enhanced_signal else {},
            'timing': 'immediate' if action == ActionType.EXECUTE else 'scheduled'
        }
        
        # Scale-in specific parameters
        if action == ActionType.SCALE_IN:
            execution['scaling'] = {
                'initial_size': 0.3,  # 30% initial
                'increments': [0.3, 0.4],  # 30%, 40% additional
                'conditions': ['price_improvement', 'confirmation']
            }
        
        # Paper trade specific parameters
        if action == ActionType.PAPER_TRADE:
            execution['paper_trade_config'] = {
                'track_slippage': True,
                'simulate_fees': True,
                'duration_hours': 24
            }
        
        return execution
    
    def _set_risk_limits(self,
                        enhanced_signal: Any,
                        ai_analysis: Dict[str, Any],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Set risk management limits."""
        risk_level = ai_analysis.get('verdict', {}).get('risk_level', 'medium')
        
        limits = {
            'max_position_size': 0.05 if risk_level == 'high' else 0.10,
            'max_loss_percentage': 1 if risk_level == 'high' else 2,
            'max_leverage': 5 if risk_level == 'high' else 10,
            'daily_loss_limit': 3,  # 3% daily loss limit
            'correlation_limit': 0.7,  # Max correlation with existing positions
            'time_stop_hours': 48,  # Close position after 48 hours if no profit
            'volatility_adjustment': {
                'enabled': True,
                'reduce_size_above': 75  # Reduce size if volatility > 75%
            }
        }
        
        # Apply enhanced signal's risk adjustments
        if enhanced_signal and enhanced_signal.risk_adjustments:
            limits.update(enhanced_signal.risk_adjustments)
        
        return limits
    
    def _define_monitoring_rules(self,
                                action: ActionType,
                                enhanced_signal: Any,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Define monitoring rules for the position."""
        rules = {
            'check_interval_seconds': 60,
            'price_alerts': {
                'deviation_percentage': 2,  # Alert on 2% price deviation
                'stop_loss_proximity': 0.5,  # Alert when 0.5% from stop loss
                'take_profit_proximity': 0.5
            },
            'volume_alerts': {
                'spike_threshold': 3,  # Alert on 3x average volume
                'dry_up_threshold': 0.3  # Alert on 30% of average volume
            },
            'technical_alerts': {
                'rsi_extreme': [30, 70],
                'macd_cross': True,
                'support_resistance_break': True
            },
            'news_monitoring': {
                'enabled': True,
                'keywords': [],  # Will be populated based on pair
                'sentiment_threshold': -0.5  # Alert on negative sentiment
            }
        }
        
        # Add action-specific monitoring
        if action == ActionType.SCALE_IN:
            rules['scaling_triggers'] = {
                'price_improvement': 0.5,  # Scale in on 0.5% better price
                'confirmation_indicators': ['rsi', 'macd'],
                'max_wait_minutes': 30
            }
        
        if action == ActionType.PAPER_TRADE:
            rules['paper_trade_monitoring'] = {
                'track_actual_price': True,
                'compare_execution': True,
                'log_all_events': True
            }
        
        return rules
    
    def _generate_alerts(self,
                        ai_analysis: Dict[str, Any],
                        validation_result: Any,
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate important alerts."""
        alerts = []
        
        # High risk alert
        risk_level = ai_analysis.get('verdict', {}).get('risk_level')
        if risk_level == 'high':
            alerts.append({
                'type': 'risk',
                'severity': 'high',
                'message': 'High risk signal - use reduced position size',
                'timestamp': datetime.now().isoformat()
            })
        
        # Low liquidity alert
        market = context.get('components', {}).get('market', {})
        if market.get('liquidity_score', 100) < 30:
            alerts.append({
                'type': 'liquidity',
                'severity': 'medium',
                'message': 'Low liquidity detected - expect higher slippage',
                'timestamp': datetime.now().isoformat()
            })
        
        # Validation warnings as alerts
        if validation_result and validation_result.warnings:
            for warning in validation_result.warnings[:2]:  # Top 2 warnings
                alerts.append({
                    'type': 'validation',
                    'severity': 'medium',
                    'message': warning,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Cross-channel conflict alert
        cross_channel = context.get('components', {}).get('cross_channel', {})
        if cross_channel.get('conflicts'):
            alerts.append({
                'type': 'consensus',
                'severity': 'medium',
                'message': f"Conflicting signals detected: {', '.join(cross_channel['conflicts'])}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _build_metadata(self,
                       decision_score: float,
                       signal_data: Dict[str, Any],
                       ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build decision metadata."""
        return {
            'decision_score': decision_score,
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal_data.get('signal_id'),
            'ai_confidence': ai_analysis.get('verdict', {}).get('confidence_score', 0),
            'risk_level': ai_analysis.get('verdict', {}).get('risk_level'),
            'processing_chain': [
                'signal_detection',
                'context_building',
                'ai_analysis',
                'validation',
                'enhancement',
                'decision'
            ],
            'decision_factors': {
                'primary': 'ai_analysis',
                'weights_used': self.decision_weights
            }
        }
    
    def _update_statistics(self, action: ActionType):
        """Update decision statistics."""
        self.decision_stats['total_decisions'] += 1
        
        if action == ActionType.EXECUTE:
            self.decision_stats['execute'] += 1
        elif action == ActionType.REJECT:
            self.decision_stats['reject'] += 1
        elif action == ActionType.MONITOR:
            self.decision_stats['monitor'] += 1
        elif action == ActionType.PAPER_TRADE:
            self.decision_stats['paper_trade'] += 1
    
    def evaluate_decision_performance(self,
                                     decision: TradingDecision,
                                     actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate decision performance against actual outcome.
        
        Args:
            decision: Original trading decision
            actual_outcome: Actual trading outcome
            
        Returns:
            Performance evaluation
        """
        evaluation = {
            'decision_quality': 'unknown',
            'outcome_match': False,
            'profit_loss': actual_outcome.get('pnl', 0),
            'lessons': []
        }
        
        # Evaluate decision quality
        if decision.action == ActionType.EXECUTE:
            if actual_outcome.get('pnl', 0) > 0:
                evaluation['decision_quality'] = 'good'
                evaluation['outcome_match'] = True
            else:
                evaluation['decision_quality'] = 'poor'
                evaluation['lessons'].append('Signal executed but resulted in loss')
        
        elif decision.action == ActionType.REJECT:
            if actual_outcome.get('missed_profit', 0) > 0:
                evaluation['decision_quality'] = 'conservative'
                evaluation['lessons'].append('Rejected signal that would have been profitable')
            else:
                evaluation['decision_quality'] = 'good'
                evaluation['outcome_match'] = True
        
        # Learn from the outcome
        if actual_outcome.get('stop_hit'):
            evaluation['lessons'].append('Stop loss was triggered')
        
        if actual_outcome.get('take_profit_hit'):
            evaluation['lessons'].append('Take profit target achieved')
        
        return evaluation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics."""
        total = self.decision_stats['total_decisions']
        
        return {
            'total_decisions': total,
            'actions': {
                'execute': self.decision_stats['execute'],
                'reject': self.decision_stats['reject'],
                'monitor': self.decision_stats['monitor'],
                'paper_trade': self.decision_stats['paper_trade']
            },
            'execution_rate': self.decision_stats['execute'] / total if total > 0 else 0,
            'rejection_rate': self.decision_stats['reject'] / total if total > 0 else 0,
            'caution_rate': (self.decision_stats['monitor'] + self.decision_stats['paper_trade']) / total if total > 0 else 0
        }
