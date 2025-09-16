"""Validation Framework for comprehensive signal validation."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of validation checks."""
    status: ValidationStatus
    score: float
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ValidationFramework:
    """
    Multi-layer validation framework for signal quality assurance.
    Implements validation rules engine with scoring mechanism.
    """
    
    def __init__(self):
        """Initialize validation framework with rules."""
        self.validation_rules = self._initialize_rules()
        self.weights = self._initialize_weights()
        self.thresholds = {
            'pass': 80,
            'warning': 60,
            'fail': 0
        }
        
        # Validation statistics
        self.validation_stats = {
            'total_validated': 0,
            'passed': 0,
            'warnings': 0,
            'failed': 0
        }
        
        logger.info("Initialized ValidationFramework")
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize validation rules."""
        return {
            'technical': {
                'price_sanity': self._validate_price_sanity,
                'stop_loss_placement': self._validate_stop_loss,
                'take_profit_levels': self._validate_take_profits,
                'risk_reward_ratio': self._validate_risk_reward,
                'leverage_appropriate': self._validate_leverage
            },
            'market': {
                'liquidity_sufficient': self._validate_liquidity,
                'spread_acceptable': self._validate_spread,
                'volatility_manageable': self._validate_volatility,
                'market_hours': self._validate_market_hours
            },
            'signal': {
                'confidence_threshold': self._validate_confidence,
                'cross_validation': self._validate_cross_channel,
                'pattern_quality': self._validate_patterns,
                'historical_performance': self._validate_historical
            },
            'risk': {
                'position_sizing': self._validate_position_size,
                'max_drawdown': self._validate_max_drawdown,
                'correlation_risk': self._validate_correlation,
                'black_swan_protection': self._validate_tail_risk
            }
        }
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize validation weights."""
        return {
            'technical': 0.35,
            'market': 0.25,
            'signal': 0.25,
            'risk': 0.15
        }
    
    def validate(self,
                signal_data: Dict[str, Any],
                ai_analysis: Dict[str, Any],
                context: Dict[str, Any]) -> ValidationResult:
        """
        Perform comprehensive validation.
        
        Args:
            signal_data: Signal parameters
            ai_analysis: AI analysis results
            context: Full context data
            
        Returns:
            Validation result with score and recommendations
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        category_scores = {}
        
        # Run validation for each category
        for category, rules in self.validation_rules.items():
            category_passed = []
            category_failed = []
            
            for rule_name, rule_func in rules.items():
                try:
                    passed, message = rule_func(signal_data, ai_analysis, context)
                    
                    if passed:
                        category_passed.append(f"{category}.{rule_name}")
                    else:
                        category_failed.append(f"{category}.{rule_name}")
                        if self._is_critical_rule(rule_name):
                            checks_failed.append(f"CRITICAL: {message}")
                        else:
                            warnings.append(message)
                            
                except Exception as e:
                    logger.error(f"Validation rule {rule_name} failed: {e}")
                    warnings.append(f"Could not validate {rule_name}")
            
            # Calculate category score
            total_rules = len(rules)
            passed_rules = len(category_passed)
            category_scores[category] = (passed_rules / total_rules * 100) if total_rules > 0 else 0
            
            checks_passed.extend(category_passed)
            checks_failed.extend(category_failed)
        
        # Calculate weighted total score
        total_score = sum(
            score * self.weights.get(category, 0.25)
            for category, score in category_scores.items()
        )
        
        # Determine status
        if total_score >= self.thresholds['pass']:
            status = ValidationStatus.PASSED
        elif total_score >= self.thresholds['warning']:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            checks_failed, warnings, signal_data, ai_analysis
        )
        
        # Update statistics
        self._update_statistics(status)
        
        return ValidationResult(
            status=status,
            score=round(total_score, 2),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                'category_scores': category_scores,
                'timestamp': datetime.now().isoformat(),
                'critical_failures': [c for c in checks_failed if 'CRITICAL' in c]
            }
        )
    
    def _validate_price_sanity(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate price levels are reasonable."""
        params = signal_data.get('trading_params', {})
        entry = params.get('entry_price', 0)
        
        if entry <= 0:
            return False, "Invalid entry price"
        
        # Check against current market price
        market_price = context.get('components', {}).get('market', {}).get('current_price', 0)
        if market_price > 0:
            deviation = abs(entry - market_price) / market_price
            if deviation > 0.05:  # More than 5% deviation
                return False, f"Entry price deviates {deviation*100:.1f}% from market"
        
        return True, "Price levels validated"
    
    def _validate_stop_loss(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate stop loss placement."""
        params = signal_data.get('trading_params', {})
        entry = params.get('entry_price', 0)
        stop_loss = params.get('stop_loss', 0)
        direction = params.get('direction', 'long')
        
        if stop_loss <= 0:
            return False, "No stop loss defined"
        
        # Check stop loss is on correct side
        if direction == 'long' and stop_loss >= entry:
            return False, "Stop loss above entry for long position"
        elif direction == 'short' and stop_loss <= entry:
            return False, "Stop loss below entry for short position"
        
        # Check stop loss distance (not too tight, not too wide)
        distance = abs(entry - stop_loss) / entry
        if distance < 0.005:  # Less than 0.5%
            return False, "Stop loss too tight (< 0.5%)"
        elif distance > 0.15:  # More than 15%
            return False, "Stop loss too wide (> 15%)"
        
        return True, "Stop loss placement validated"
    
    def _validate_take_profits(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate take profit levels."""
        params = signal_data.get('trading_params', {})
        entry = params.get('entry_price', 0)
        take_profits = params.get('take_profits', [])
        direction = params.get('direction', 'long')
        
        if not take_profits:
            return False, "No take profit levels defined"
        
        # Check take profits are on correct side
        for tp in take_profits:
            if direction == 'long' and tp <= entry:
                return False, "Take profit below entry for long position"
            elif direction == 'short' and tp >= entry:
                return False, "Take profit above entry for short position"
        
        # Check take profits are achievable
        max_tp_distance = max(abs(tp - entry) / entry for tp in take_profits)
        if max_tp_distance > 0.5:  # More than 50% move
            return False, "Take profit targets unrealistic (> 50% move)"
        
        return True, "Take profit levels validated"
    
    def _validate_risk_reward(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate risk/reward ratio."""
        params = signal_data.get('trading_params', {})
        rr_ratio = params.get('risk_reward_ratio')
        
        if not rr_ratio:
            return False, "No risk/reward ratio calculated"
        
        risk, reward = rr_ratio
        if reward / risk < 1.5:
            return False, f"Poor risk/reward ratio: 1:{reward/risk:.1f}"
        
        return True, f"Good risk/reward ratio: 1:{reward/risk:.1f}"
    
    def _validate_leverage(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate leverage is appropriate."""
        params = signal_data.get('trading_params', {})
        leverage = params.get('leverage')
        risk_level = ai_analysis.get('verdict', {}).get('risk_level', 'high')
        
        if not leverage:
            return True, "No leverage used"
        
        # Check leverage against risk level
        if risk_level == 'high' and leverage > 5:
            return False, f"Leverage too high ({leverage}x) for high risk signal"
        elif risk_level == 'medium' and leverage > 10:
            return False, f"Leverage too high ({leverage}x) for medium risk signal"
        elif leverage > 20:
            return False, f"Excessive leverage ({leverage}x)"
        
        return True, f"Leverage appropriate ({leverage}x)"
    
    def _validate_liquidity(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate market liquidity."""
        market = context.get('components', {}).get('market', {})
        liquidity_score = market.get('liquidity_score', 0)
        volume_24h = market.get('volume', {}).get('quote_24h', 0)
        
        if liquidity_score < 30:
            return False, f"Poor liquidity score: {liquidity_score}"
        
        if volume_24h < 100000:  # Less than $100k daily volume
            return False, f"Insufficient 24h volume: ${volume_24h:,.0f}"
        
        return True, "Liquidity validated"
    
    def _validate_spread(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate bid-ask spread."""
        market = context.get('components', {}).get('market', {})
        spread_pct = market.get('bid_ask', {}).get('spread_percentage', 0)
        
        if spread_pct > 0.5:
            return False, f"Wide spread: {spread_pct:.2f}%"
        
        return True, f"Acceptable spread: {spread_pct:.2f}%"
    
    def _validate_volatility(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate market volatility."""
        market = context.get('components', {}).get('market', {})
        volatility = market.get('volatility', 0)
        
        if volatility > 100:  # Annualized volatility > 100%
            return False, f"Extreme volatility: {volatility:.1f}%"
        
        return True, f"Manageable volatility: {volatility:.1f}%"
    
    def _validate_market_hours(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate market hours and timing."""
        # Crypto markets are 24/7, but check for low activity periods
        current_hour = datetime.now().hour
        
        # Warning for typically low volume hours (varies by exchange)
        if 2 <= current_hour <= 6:  # UTC early morning
            return True, "Warning: Low activity period"
        
        return True, "Active market hours"
    
    def _validate_confidence(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate AI confidence score."""
        confidence = ai_analysis.get('verdict', {}).get('confidence_score', 0)
        
        if confidence < 60:
            return False, f"Low confidence score: {confidence}%"
        
        return True, f"Good confidence score: {confidence}%"
    
    def _validate_cross_channel(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate cross-channel consensus."""
        cross_channel = context.get('components', {}).get('cross_channel', {})
        consensus = cross_channel.get('consensus_score', 0)
        validation_status = cross_channel.get('validation_status', 'insufficient')
        
        if validation_status == 'conflict':
            return False, "Conflicting signals across channels"
        
        if consensus < 50:
            return False, f"Poor cross-channel consensus: {consensus}%"
        
        return True, f"Cross-channel validated: {consensus}%"
    
    def _validate_patterns(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate pattern quality."""
        pattern_analysis = signal_data.get('pattern_analysis', {})
        pattern_count = pattern_analysis.get('total_patterns', 0)
        
        if pattern_count < 3:
            return False, f"Insufficient patterns detected: {pattern_count}"
        
        return True, f"Multiple patterns confirmed: {pattern_count}"
    
    def _validate_historical(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate historical performance."""
        historical = context.get('components', {}).get('historical', {})
        perf = historical.get('performance_metrics', {})
        win_rate = perf.get('win_rate', 0)
        
        if win_rate > 0 and win_rate < 0.4:
            return False, f"Poor historical win rate: {win_rate*100:.1f}%"
        
        return True, f"Historical performance acceptable"
    
    def _validate_position_size(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate position sizing."""
        risk = ai_analysis.get('risk', {})
        position_size = risk.get('position_size_percentage', 0)
        
        if position_size > 10:
            return False, f"Position size too large: {position_size}%"
        
        if position_size == 0:
            return False, "No position size calculated"
        
        return True, f"Position size appropriate: {position_size}%"
    
    def _validate_max_drawdown(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate maximum drawdown."""
        risk = ai_analysis.get('risk', {})
        max_dd = risk.get('max_drawdown_estimate', 0)
        
        if max_dd > 20:
            return False, f"Excessive potential drawdown: {max_dd}%"
        
        return True, f"Acceptable drawdown risk: {max_dd}%"
    
    def _validate_correlation(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate correlation risk."""
        market = context.get('components', {}).get('market', {})
        correlations = market.get('correlations', {})
        
        # Check for high correlation with major pairs
        high_correlations = [
            pair for pair, corr in correlations.items()
            if abs(corr) > 0.8
        ]
        
        if len(high_correlations) > 1:
            return False, f"High correlation with {', '.join(high_correlations)}"
        
        return True, "Correlation risk acceptable"
    
    def _validate_tail_risk(self, signal_data: Dict, ai_analysis: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate black swan protection."""
        params = signal_data.get('trading_params', {})
        stop_loss = params.get('stop_loss', 0)
        leverage = params.get('leverage', 1)
        
        # Check if stop loss provides adequate protection
        if stop_loss == 0:
            return False, "No black swan protection (no stop loss)"
        
        if leverage > 10:
            return False, f"High leverage ({leverage}x) increases tail risk"
        
        return True, "Tail risk protection in place"
    
    def _is_critical_rule(self, rule_name: str) -> bool:
        """Determine if a rule is critical."""
        critical_rules = [
            'price_sanity',
            'stop_loss_placement',
            'liquidity_sufficient',
            'confidence_threshold'
        ]
        return rule_name in critical_rules
    
    def _generate_recommendations(self,
                                 checks_failed: List[str],
                                 warnings: List[str],
                                 signal_data: Dict,
                                 ai_analysis: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Recommendations based on failures
        if any('stop loss' in f.lower() for f in checks_failed):
            recommendations.append("Adjust stop loss to appropriate level (1-5% from entry)")
        
        if any('liquidity' in f.lower() for f in checks_failed):
            recommendations.append("Consider reducing position size due to low liquidity")
        
        if any('confidence' in f.lower() for f in checks_failed):
            recommendations.append("Wait for stronger signal confirmation")
        
        if any('leverage' in f.lower() for f in checks_failed):
            recommendations.append("Reduce leverage to manage risk")
        
        # Recommendations based on warnings
        if any('volatility' in w.lower() for w in warnings):
            recommendations.append("Use wider stops and smaller position due to volatility")
        
        if any('spread' in w.lower() for w in warnings):
            recommendations.append("Use limit orders to avoid spread costs")
        
        # General recommendations
        if ai_analysis.get('verdict', {}).get('risk_level') == 'high':
            recommendations.append("Consider paper trading this signal first")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _update_statistics(self, status: ValidationStatus):
        """Update validation statistics."""
        self.validation_stats['total_validated'] += 1
        
        if status == ValidationStatus.PASSED:
            self.validation_stats['passed'] += 1
        elif status == ValidationStatus.WARNING:
            self.validation_stats['warnings'] += 1
        elif status == ValidationStatus.FAILED:
            self.validation_stats['failed'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_validated']
        
        return {
            'total_validated': total,
            'passed': self.validation_stats['passed'],
            'warnings': self.validation_stats['warnings'],
            'failed': self.validation_stats['failed'],
            'pass_rate': self.validation_stats['passed'] / total if total > 0 else 0,
            'fail_rate': self.validation_stats['failed'] / total if total > 0 else 0
        }
