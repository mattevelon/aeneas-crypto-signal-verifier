"""Signal Classification module for categorizing trading signals."""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalUrgency(Enum):
    """Signal urgency levels."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketCondition(Enum):
    """Market condition types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class SignalQuality(Enum):
    """Signal quality ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class SignalClassification:
    """Complete signal classification results."""
    signal_type: str  # scalp/swing/position
    urgency: SignalUrgency
    direction: str  # long/short
    market_condition: Optional[MarketCondition]
    quality: SignalQuality
    confidence_score: float
    risk_level: str  # low/medium/high
    metadata: Dict[str, Any]


class SignalClassifier:
    """
    Classifies trading signals based on detected patterns and parameters.
    Implements ensemble voting mechanism for confidence scoring.
    """
    
    def __init__(self):
        """Initialize signal classifier with classification rules."""
        self.classification_rules = self._initialize_rules()
        self.quality_thresholds = {
            'excellent': 85,
            'good': 70,
            'fair': 50,
            'poor': 0
        }
        logger.info("Initialized signal classifier")
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize classification rules and weights."""
        return {
            'urgency_keywords': {
                'immediate': ['now', 'urgent', 'asap', 'immediately', 'quick', 'fast'],
                'high': ['alert', 'important', 'attention', 'hot', 'strong'],
                'medium': ['setup', 'prepare', 'watch', 'monitor'],
                'low': ['potential', 'possible', 'maybe', 'consider']
            },
            'timeframe_mapping': {
                'scalp': ['1m', '5m', '15m', 'scalp', 'quick'],
                'swing': ['1h', '4h', '1d', 'swing', 'daily'],
                'position': ['1w', 'weekly', 'monthly', 'long-term', 'position']
            },
            'risk_indicators': {
                'low': ['safe', 'conservative', 'low risk', 'secure'],
                'medium': ['moderate', 'balanced', 'standard'],
                'high': ['risky', 'aggressive', 'high risk', 'volatile']
            },
            'market_conditions': {
                'trending_up': ['uptrend', 'bullish', 'ascending', 'rising'],
                'trending_down': ['downtrend', 'bearish', 'descending', 'falling'],
                'ranging': ['range', 'sideways', 'consolidation', 'flat'],
                'volatile': ['volatile', 'choppy', 'unstable', 'erratic'],
                'breakout': ['breakout', 'breakthrough', 'break above', 'break below'],
                'reversal': ['reversal', 'reverse', 'turn', 'flip']
            }
        }
    
    def classify(self, 
                 signal_params: Any,  # TradingSignal from parameter_extractor
                 detected_patterns: List[Dict[str, Any]],
                 text: str) -> SignalClassification:
        """
        Classify a trading signal based on multiple factors.
        
        Args:
            signal_params: Extracted signal parameters (TradingSignal object)
            detected_patterns: Patterns detected by pattern engine
            text: Original message text
            
        Returns:
            Complete signal classification
        """
        # Determine signal type
        signal_type = self._classify_signal_type(signal_params, detected_patterns, text)
        
        # Determine urgency
        urgency = self._classify_urgency(text, detected_patterns)
        
        # Determine market condition
        market_condition = self._classify_market_condition(text, detected_patterns)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(signal_params, detected_patterns)
        quality = self._determine_quality_rating(quality_score)
        
        # Determine risk level
        risk_level = self._classify_risk_level(signal_params, text, detected_patterns)
        
        # Calculate ensemble confidence
        confidence_score = self._calculate_ensemble_confidence(
            signal_params, detected_patterns, quality_score
        )
        
        # Build metadata
        metadata = self._build_classification_metadata(
            signal_params, detected_patterns, text
        )
        
        return SignalClassification(
            signal_type=signal_type,
            urgency=urgency,
            direction=signal_params.direction if signal_params else 'unknown',
            market_condition=market_condition,
            quality=quality,
            confidence_score=confidence_score,
            risk_level=risk_level,
            metadata=metadata
        )
    
    def _classify_signal_type(self, signal_params: Any, patterns: List[Dict], text: str) -> str:
        """Classify signal as scalp/swing/position."""
        # Check explicit signal type in parameters
        if signal_params and signal_params.signal_type:
            return signal_params.signal_type
        
        # Check patterns for signal type
        for pattern in patterns:
            if pattern.get('signal_type'):
                return pattern['signal_type']
        
        # Check timeframe
        if signal_params and signal_params.timeframe:
            tf = signal_params.timeframe.lower()
            for sig_type, keywords in self.classification_rules['timeframe_mapping'].items():
                if any(kw in tf for kw in keywords):
                    return sig_type
        
        # Check text for keywords
        text_lower = text.lower()
        for sig_type, keywords in self.classification_rules['timeframe_mapping'].items():
            if any(kw in text_lower for kw in keywords):
                return sig_type
        
        # Default based on leverage
        if signal_params and signal_params.leverage:
            if signal_params.leverage > 10:
                return 'scalp'
            elif signal_params.leverage > 5:
                return 'swing'
            else:
                return 'position'
        
        return 'swing'  # Default
    
    def _classify_urgency(self, text: str, patterns: List[Dict]) -> SignalUrgency:
        """Determine signal urgency level."""
        text_lower = text.lower()
        
        # Check for urgency keywords
        for urgency_level, keywords in self.classification_rules['urgency_keywords'].items():
            if any(kw in text_lower for kw in keywords):
                return SignalUrgency[urgency_level.upper()]
        
        # Check pattern names for urgency indicators
        pattern_names = [p['pattern_name'] for p in patterns]
        if any('urgent' in name or 'alert' in name for name in pattern_names):
            return SignalUrgency.HIGH
        
        # Check for time-sensitive indicators
        if any(indicator in text_lower for indicator in ['expires', 'limited', 'ending']):
            return SignalUrgency.HIGH
        
        return SignalUrgency.MEDIUM  # Default
    
    def _classify_market_condition(self, text: str, patterns: List[Dict]) -> Optional[MarketCondition]:
        """Identify market condition from signal context."""
        text_lower = text.lower()
        
        # Check for market condition keywords
        for condition, keywords in self.classification_rules['market_conditions'].items():
            if any(kw in text_lower for kw in keywords):
                return MarketCondition[condition.upper()]
        
        # Check patterns for market indicators
        pattern_names = [p['pattern_name'] for p in patterns]
        if any('breakout' in name for name in pattern_names):
            return MarketCondition.BREAKOUT
        if any('trend' in name for name in pattern_names):
            if any('up' in name for name in pattern_names):
                return MarketCondition.TRENDING_UP
            elif any('down' in name for name in pattern_names):
                return MarketCondition.TRENDING_DOWN
        
        return None
    
    def _classify_risk_level(self, signal_params: Any, text: str, patterns: List[Dict]) -> str:
        """Determine risk level of the signal."""
        text_lower = text.lower()
        
        # Check for explicit risk keywords
        for risk_level, keywords in self.classification_rules['risk_indicators'].items():
            if any(kw in text_lower for kw in keywords):
                return risk_level
        
        # Calculate based on risk/reward ratio
        if signal_params and signal_params.risk_reward_ratio:
            risk, reward = signal_params.risk_reward_ratio
            if reward / risk >= 3:
                return 'low'
            elif reward / risk >= 1.5:
                return 'medium'
            else:
                return 'high'
        
        # Calculate based on stop loss distance
        if signal_params and signal_params.entry_price > 0 and signal_params.stop_loss > 0:
            sl_distance = abs(signal_params.entry_price - signal_params.stop_loss) / signal_params.entry_price
            if sl_distance < 0.02:  # Less than 2%
                return 'low'
            elif sl_distance < 0.05:  # Less than 5%
                return 'medium'
            else:
                return 'high'
        
        # Check leverage
        if signal_params and signal_params.leverage:
            if signal_params.leverage > 10:
                return 'high'
            elif signal_params.leverage > 5:
                return 'medium'
            else:
                return 'low'
        
        return 'medium'  # Default
    
    def _calculate_quality_score(self, signal_params: Any, patterns: List[Dict]) -> float:
        """Calculate signal quality score (0-100)."""
        score = 0
        max_score = 100
        
        # Check for essential components (40 points)
        if signal_params:
            if signal_params.pair and signal_params.pair != "UNKNOWN/USDT":
                score += 10
            if signal_params.entry_price > 0:
                score += 10
            if signal_params.stop_loss > 0:
                score += 10
            if signal_params.take_profits and len(signal_params.take_profits) > 0:
                score += 10
        
        # Pattern detection quality (30 points)
        pattern_count = len(patterns)
        if pattern_count >= 5:
            score += 30
        elif pattern_count >= 3:
            score += 20
        elif pattern_count >= 1:
            score += 10
        
        # Risk management (20 points)
        if signal_params:
            if signal_params.risk_reward_ratio:
                risk, reward = signal_params.risk_reward_ratio
                if reward / risk >= 2:
                    score += 20
                elif reward / risk >= 1.5:
                    score += 15
                elif reward / risk >= 1:
                    score += 10
        
        # Additional quality indicators (10 points)
        high_confidence_patterns = [p for p in patterns if p.get('confidence_weight', 0) > 0.8]
        if len(high_confidence_patterns) >= 3:
            score += 10
        elif len(high_confidence_patterns) >= 2:
            score += 7
        elif len(high_confidence_patterns) >= 1:
            score += 5
        
        return min(score, max_score)
    
    def _determine_quality_rating(self, score: float) -> SignalQuality:
        """Convert quality score to rating."""
        if score >= self.quality_thresholds['excellent']:
            return SignalQuality.EXCELLENT
        elif score >= self.quality_thresholds['good']:
            return SignalQuality.GOOD
        elif score >= self.quality_thresholds['fair']:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
    
    def _calculate_ensemble_confidence(self, 
                                      signal_params: Any,
                                      patterns: List[Dict],
                                      quality_score: float) -> float:
        """
        Calculate ensemble confidence score using voting mechanism.
        Target F1 score: 0.92
        """
        votes = []
        
        # Pattern-based confidence
        if patterns:
            pattern_confidence = sum(p.get('confidence_weight', 0) for p in patterns) / len(patterns)
            votes.append(pattern_confidence * 100)
        
        # Quality-based confidence
        votes.append(quality_score)
        
        # Parameter completeness confidence
        if signal_params:
            completeness = 0
            if signal_params.pair and signal_params.pair != "UNKNOWN/USDT":
                completeness += 20
            if signal_params.entry_price > 0:
                completeness += 20
            if signal_params.stop_loss > 0:
                completeness += 20
            if signal_params.take_profits:
                completeness += 20
            if signal_params.risk_reward_ratio:
                completeness += 20
            votes.append(completeness)
        
        # Risk-adjusted confidence
        if signal_params and signal_params.risk_reward_ratio:
            risk, reward = signal_params.risk_reward_ratio
            rr_confidence = min(100, (reward / risk) * 30)
            votes.append(rr_confidence)
        
        # Calculate weighted ensemble score
        if votes:
            # Use weighted average with more weight on pattern and quality
            weights = [0.3, 0.3, 0.2, 0.2][:len(votes)]
            weighted_sum = sum(v * w for v, w in zip(votes, weights))
            total_weight = sum(weights)
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = 0
        
        return round(min(100, max(0, ensemble_score)), 2)
    
    def _build_classification_metadata(self, 
                                      signal_params: Any,
                                      patterns: List[Dict],
                                      text: str) -> Dict[str, Any]:
        """Build metadata for classification results."""
        metadata = {
            'classified_at': datetime.now().isoformat(),
            'pattern_count': len(patterns),
            'text_length': len(text),
            'has_complete_params': False,
            'pattern_types': []
        }
        
        if signal_params:
            metadata['has_complete_params'] = all([
                signal_params.pair != "UNKNOWN/USDT",
                signal_params.entry_price > 0,
                signal_params.stop_loss > 0,
                bool(signal_params.take_profits)
            ])
            
            if signal_params.leverage:
                metadata['leverage'] = signal_params.leverage
            
            if signal_params.timeframe:
                metadata['timeframe'] = signal_params.timeframe
        
        # Add unique pattern types
        pattern_types = set()
        for pattern in patterns:
            pattern_name = pattern.get('pattern_name', '')
            # Extract pattern category
            if '_' in pattern_name:
                category = pattern_name.split('_')[0]
                pattern_types.add(category)
        metadata['pattern_types'] = list(pattern_types)
        
        # Add pattern confidence distribution
        confidences = [p.get('confidence_weight', 0) for p in patterns]
        if confidences:
            metadata['avg_pattern_confidence'] = sum(confidences) / len(confidences)
            metadata['max_pattern_confidence'] = max(confidences)
            metadata['min_pattern_confidence'] = min(confidences)
        
        return metadata
    
    def validate_classification(self, classification: SignalClassification) -> bool:
        """
        Validate classification results for consistency.
        
        Args:
            classification: Classification to validate
            
        Returns:
            True if classification is valid
        """
        # Check required fields
        if not classification.signal_type or not classification.direction:
            return False
        
        # Check confidence score range
        if not 0 <= classification.confidence_score <= 100:
            return False
        
        # Check quality consistency with confidence
        if classification.quality == SignalQuality.EXCELLENT and classification.confidence_score < 70:
            return False
        if classification.quality == SignalQuality.POOR and classification.confidence_score > 50:
            return False
        
        # Check risk level validity
        if classification.risk_level not in ['low', 'medium', 'high']:
            return False
        
        return True
