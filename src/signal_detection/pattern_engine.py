"""Pattern Recognition Engine for detecting trading signals using regex and ML."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    LONG = "long"
    SHORT = "short"
    SPOT = "spot"
    FUTURES = "futures"
    SCALP = "scalp"
    SWING = "swing"
    POSITION = "position"


@dataclass
class SignalPattern:
    """Represents a signal detection pattern."""
    name: str
    pattern: str
    confidence_weight: float
    signal_type: Optional[SignalType] = None
    flags: int = re.IGNORECASE | re.MULTILINE


class PatternRecognitionEngine:
    """
    Regex-based pattern matching engine for signal detection.
    Implements 50+ patterns for comprehensive signal identification.
    """
    
    def __init__(self):
        """Initialize pattern recognition engine with trading signal patterns."""
        self.patterns = self._initialize_patterns()
        self.compiled_patterns = self._compile_patterns()
        logger.info(f"Initialized pattern engine with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self) -> List[SignalPattern]:
        """Initialize comprehensive list of signal detection patterns."""
        patterns = [
            # Entry Signal Patterns
            SignalPattern(
                name="entry_buy_signal",
                pattern=r"(?:buy|long|enter\s+long|open\s+long)[\s:]*(?:@|at|price)?\s*([\d.,]+)",
                confidence_weight=0.9,
                signal_type=SignalType.LONG
            ),
            SignalPattern(
                name="entry_sell_signal", 
                pattern=r"(?:sell|short|enter\s+short|open\s+short)[\s:]*(?:@|at|price)?\s*([\d.,]+)",
                confidence_weight=0.9,
                signal_type=SignalType.SHORT
            ),
            SignalPattern(
                name="entry_zone",
                pattern=r"(?:entry|buy|long)\s*(?:zone|area|range)[\s:]*(\d+[\d.,]*)\s*[-–]\s*(\d+[\d.,]*)",
                confidence_weight=0.85
            ),
            
            # Trading Pair Patterns
            SignalPattern(
                name="crypto_pair_standard",
                pattern=r"([A-Z]{2,10})[/\-_]([A-Z]{3,6})",
                confidence_weight=0.95
            ),
            SignalPattern(
                name="crypto_pair_spaced",
                pattern=r"([A-Z]{2,10})\s+([A-Z]{3,6})\s+(?:pair|trade|signal)",
                confidence_weight=0.8
            ),
            SignalPattern(
                name="crypto_ticker",
                pattern=r"\$([A-Z]{2,10})",
                confidence_weight=0.7
            ),
            
            # Entry patterns with dollar signs
            SignalPattern(
                name="entry_dollar_single",
                pattern=r"(?:entry|buy|long)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="entry_dollar_range",
                pattern=r"(?:entry|buy|long)[\s:]*\$?([\d,]+(?:\.\d+)?)\s*[-–]\s*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.95
            ),
            
            # Direction patterns
            SignalPattern(
                name="direction_long",
                pattern=r"(?i)(?:direction|side|position)[\s:]*(long|buy)",
                confidence_weight=0.9,
                signal_type=SignalType.LONG
            ),
            SignalPattern(
                name="direction_short",
                pattern=r"(?i)(?:direction|side|position)[\s:]*(short|sell)",
                confidence_weight=0.9,
                signal_type=SignalType.SHORT
            ),
            SignalPattern(
                name="simple_long",
                pattern=r"(?i)\b(long)\b",
                confidence_weight=0.7,
                signal_type=SignalType.LONG
            ),
            SignalPattern(
                name="simple_short",
                pattern=r"(?i)\b(short)\b",
                confidence_weight=0.7,
                signal_type=SignalType.SHORT
            ),
            
            # Stop Loss Patterns
            SignalPattern(
                name="stop_loss_explicit",
                pattern=r"(?:stop\s*loss|sl|stop)[\s:]*(?:@|at)?\s*([\d.,]+)",
                confidence_weight=0.95
            ),
            SignalPattern(
                name="stop_loss_dollar",
                pattern=r"(?:stop\s*loss|sl|stop)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.95
            ),
            SignalPattern(
                name="stop_loss_percentage",
                pattern=r"(?:stop\s*loss|sl)[\s:]*(\d+(?:\.\d+)?)\s*%",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="invalidation_level",
                pattern=r"(?:invalid|invalidation)[\s:]*(?:below|above)?\s*([\d.,]+)",
                confidence_weight=0.85
            ),
            
            # Take Profit Patterns
            SignalPattern(
                name="take_profit_single",
                pattern=r"(?:take\s*profit|tp|target)[\s:]*(?:@|at)?\s*([\d.,]+)",
                confidence_weight=0.95
            ),
            SignalPattern(
                name="take_profit_dollar",
                pattern=r"(?:tp\d?|target\s*\d?|take\s*profit\s*\d?)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="take_profit_multiple",
                pattern=r"(?:tp|target)\s*(\d)[\s:]*(?:@|at)?\s*([\d.,]+)",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="take_profit_range",
                pattern=r"(?:targets?|tp)[\s:]*(\d+[\d.,]*)\s*[-–]\s*(\d+[\d.,]*)",
                confidence_weight=0.85
            ),
            
            # Leverage Patterns
            SignalPattern(
                name="leverage_explicit",
                pattern=r"(?:leverage|lev)[\s:]*(\d+)x?",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="leverage_cross",
                pattern=r"(?:cross|isolated)\s*(?:margin)?[\s:]*(\d+)x?",
                confidence_weight=0.85
            ),
            
            # Risk Management Patterns
            SignalPattern(
                name="risk_percentage",
                pattern=r"(?:risk|position\s*size)[\s:]*(\d+(?:\.\d+)?)\s*%",
                confidence_weight=0.85
            ),
            SignalPattern(
                name="risk_reward_ratio",
                pattern=r"(?:r:r|risk:reward|rr)[\s:]*(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)",
                confidence_weight=0.9
            ),
            
            # Market Type Patterns
            SignalPattern(
                name="spot_market",
                pattern=r"(?:spot|cash)\s*(?:market|trade|position)",
                confidence_weight=0.8,
                signal_type=SignalType.SPOT
            ),
            SignalPattern(
                name="futures_market",
                pattern=r"(?:futures?|perp|perpetual|contract)",
                confidence_weight=0.8,
                signal_type=SignalType.FUTURES
            ),
            
            # Time Frame Patterns
            SignalPattern(
                name="timeframe_minutes",
                pattern=r"(\d+)\s*(?:m|min|minute)s?",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="timeframe_hours",
                pattern=r"(\d+)\s*(?:h|hr|hour)s?",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="timeframe_days",
                pattern=r"(\d+)\s*(?:d|day)s?",
                confidence_weight=0.7
            ),
            
            # Trading Style Patterns
            SignalPattern(
                name="scalp_trade",
                pattern=r"scalp(?:ing)?|quick\s*trade",
                confidence_weight=0.75,
                signal_type=SignalType.SCALP
            ),
            SignalPattern(
                name="swing_trade",
                pattern=r"swing\s*(?:trade|position)",
                confidence_weight=0.75,
                signal_type=SignalType.SWING
            ),
            SignalPattern(
                name="position_trade",
                pattern=r"position\s*(?:trade|long\s*term)",
                confidence_weight=0.75,
                signal_type=SignalType.POSITION
            ),
            
            # Action Patterns
            SignalPattern(
                name="action_buy_now",
                pattern=r"(?:buy|long)\s*now|immediate\s*(?:buy|long)",
                confidence_weight=0.85
            ),
            SignalPattern(
                name="action_wait",
                pattern=r"wait\s*(?:for|until)|don'?t\s*(?:buy|enter)",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="action_close",
                pattern=r"(?:close|exit|take\s*profit)\s*(?:position|trade)?",
                confidence_weight=0.8
            ),
            
            # Confidence Indicators
            SignalPattern(
                name="high_confidence",
                pattern=r"(?:high|strong)\s*(?:confidence|probability|conviction)",
                confidence_weight=1.0
            ),
            SignalPattern(
                name="medium_confidence",
                pattern=r"(?:medium|moderate)\s*(?:confidence|probability|risk)",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="low_confidence",
                pattern=r"(?:low|weak)\s*(?:confidence|probability)|risky",
                confidence_weight=0.5
            ),
            
            # Alert Patterns
            SignalPattern(
                name="alert_urgent",
                pattern=r"(?:urgent|alert|immediate|asap|now)",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="alert_important",
                pattern=r"(?:important|attention|note|must\s*read)",
                confidence_weight=0.8
            ),
            
            # Breakout Patterns
            SignalPattern(
                name="breakout_pattern",
                pattern=r"break(?:out|ing)?\s*(?:above|below)?\s*([\d.,]+)",
                confidence_weight=0.85
            ),
            SignalPattern(
                name="resistance_level",
                pattern=r"resistance[\s:]*(?:@|at)?\s*([\d.,]+)",
                confidence_weight=0.8
            ),
            SignalPattern(
                name="support_level",
                pattern=r"support[\s:]*(?:@|at)?\s*([\d.,]+)",
                confidence_weight=0.8
            ),
            
            # Volume Patterns
            SignalPattern(
                name="volume_increase",
                pattern=r"(?:high|increasing|strong)\s*volume",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="volume_decrease",
                pattern=r"(?:low|decreasing|weak)\s*volume",
                confidence_weight=0.7
            ),
            
            # Trend Patterns
            SignalPattern(
                name="uptrend",
                pattern=r"(?:up\s*trend|bullish|ascending)",
                confidence_weight=0.75
            ),
            SignalPattern(
                name="downtrend",
                pattern=r"(?:down\s*trend|bearish|descending)",
                confidence_weight=0.75
            ),
            
            # Emoji Signal Patterns
            SignalPattern(
                name="emoji_bullish",
                pattern=r"[🚀🔥💰📈🟢⬆️]+",
                confidence_weight=0.6
            ),
            SignalPattern(
                name="emoji_bearish",
                pattern=r"[📉🔴⬇️🩸💸]+",
                confidence_weight=0.6
            ),
            
            # Complex Signal Formats
            SignalPattern(
                name="signal_format_1",
                pattern=r"pair:\s*([A-Z]+/[A-Z]+).*entry:\s*([\d.,]+).*sl:\s*([\d.,]+).*tp:\s*([\d.,]+)",
                confidence_weight=0.95,
                flags=re.IGNORECASE | re.DOTALL
            ),
            SignalPattern(
                name="signal_format_2",
                pattern=r"([A-Z]+/[A-Z]+)\s*\n.*buy\s*@?\s*([\d.,]+)\s*\n.*stop\s*@?\s*([\d.,]+)",
                confidence_weight=0.9,
                flags=re.IGNORECASE | re.DOTALL
            ),
            
            # Percentage Move Patterns
            SignalPattern(
                name="percentage_gain",
                pattern=r"\+(\d+(?:\.\d+)?)\s*%",
                confidence_weight=0.7
            ),
            SignalPattern(
                name="percentage_loss",
                pattern=r"-(\d+(?:\.\d+)?)\s*%",
                confidence_weight=0.7
            ),
            
            # Order Type Patterns
            SignalPattern(
                name="limit_order",
                pattern=r"limit\s*(?:order|buy|sell)",
                confidence_weight=0.8
            ),
            SignalPattern(
                name="market_order",
                pattern=r"market\s*(?:order|buy|sell)",
                confidence_weight=0.8
            ),
            
            # Update/Cancel Patterns
            SignalPattern(
                name="signal_update",
                pattern=r"(?:update|modify|change)\s*(?:signal|trade|position)",
                confidence_weight=0.85
            ),
            SignalPattern(
                name="signal_cancel",
                pattern=r"(?:cancel|close|exit|stop)\s*(?:signal|trade|position)",
                confidence_weight=0.85
            ),
        ]
        
        return patterns
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile all patterns for efficient matching."""
        compiled = {}
        for pattern in self.patterns:
            try:
                compiled[pattern.name] = re.compile(pattern.pattern, pattern.flags)
            except re.error as e:
                logger.error(f"Failed to compile pattern {pattern.name}: {e}")
        return compiled
    
    def detect_signals(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect trading signals in text using pattern matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected signals with confidence scores
        """
        signals = []
        text = self._preprocess_text(text)
        
        for pattern in self.patterns:
            if pattern.name not in self.compiled_patterns:
                continue
                
            regex = self.compiled_patterns[pattern.name]
            matches = regex.finditer(text)
            
            for match in matches:
                signal = {
                    'pattern_name': pattern.name,
                    'match': match.group(0),
                    'groups': match.groups(),
                    'position': (match.start(), match.end()),
                    'confidence_weight': pattern.confidence_weight,
                    'signal_type': pattern.signal_type.value if pattern.signal_type else None
                }
                signals.append(signal)
        
        return signals
    
    def calculate_confidence(self, signals: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence score based on detected patterns.
        
        Args:
            signals: List of detected signals
            
        Returns:
            Confidence score between 0 and 100
        """
        if not signals:
            return 0.0
        
        # Weight calculation based on pattern matches
        total_weight = sum(s['confidence_weight'] for s in signals)
        max_possible_weight = len(self.patterns) * 1.0
        
        # Normalize to 0-100 scale
        base_confidence = (total_weight / max_possible_weight) * 100
        
        # Boost confidence for certain combinations
        boost = 0
        pattern_names = {s['pattern_name'] for s in signals}
        
        # Essential components boost
        if any(p in pattern_names for p in ['entry_buy_signal', 'entry_sell_signal']):
            boost += 10
        if any(p in pattern_names for p in ['stop_loss_explicit', 'stop_loss_percentage']):
            boost += 10
        if any(p in pattern_names for p in ['take_profit_single', 'take_profit_multiple']):
            boost += 10
        if any(p in pattern_names for p in ['crypto_pair_standard', 'crypto_pair_spaced']):
            boost += 5
        
        confidence = min(100, base_confidence + boost)
        return round(confidence, 2)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better pattern matching.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common variations
        text = re.sub(r'[–—]', '-', text)  # Normalize dashes
        text = re.sub(r'[@#]', '', text)  # Remove @ and # symbols
        
        # Keep original case for now (patterns handle case)
        return text.strip()
    
    def extract_key_components(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract key trading components from detected signals.
        
        Args:
            signals: List of detected signals
            
        Returns:
            Dictionary with extracted components
        """
        components = {
            'pairs': [],
            'entry_prices': [],
            'stop_losses': [],
            'take_profits': [],
            'signal_types': [],
            'leverage': None,
            'risk_reward': None,
            'timeframe': None,
            'confidence': self.calculate_confidence(signals)
        }
        
        for signal in signals:
            pattern_name = signal['pattern_name']
            groups = signal['groups']
            
            # Extract pairs
            if 'crypto_pair' in pattern_name and groups:
                if len(groups) >= 2:
                    pair = f"{groups[0]}/{groups[1]}"
                    components['pairs'].append(pair)
            
            # Extract entry prices
            if 'entry' in pattern_name and groups and groups[0]:
                try:
                    price = float(groups[0].replace(',', ''))
                    components['entry_prices'].append(price)
                except ValueError:
                    pass
            
            # Extract stop losses
            if 'stop_loss' in pattern_name and groups and groups[0]:
                try:
                    if '%' in signal['match']:
                        components['stop_losses'].append(f"{groups[0]}%")
                    else:
                        sl = float(groups[0].replace(',', ''))
                        components['stop_losses'].append(sl)
                except ValueError:
                    pass
            
            # Extract take profits
            if 'take_profit' in pattern_name and groups:
                for group in groups:
                    if group:
                        try:
                            tp = float(group.replace(',', ''))
                            components['take_profits'].append(tp)
                        except ValueError:
                            pass
            
            # Extract signal types
            if signal['signal_type']:
                components['signal_types'].append(signal['signal_type'])
            
            # Extract leverage
            if 'leverage' in pattern_name and groups and groups[0]:
                try:
                    components['leverage'] = int(groups[0])
                except ValueError:
                    pass
            
            # Extract risk/reward ratio
            if 'risk_reward_ratio' in pattern_name and len(groups) >= 2:
                try:
                    components['risk_reward'] = (float(groups[0]), float(groups[1]))
                except ValueError:
                    pass
            
            # Extract timeframe
            if 'timeframe' in pattern_name and groups and groups[0]:
                components['timeframe'] = signal['match']
        
        # Deduplicate and clean
        components['pairs'] = list(set(components['pairs']))
        components['signal_types'] = list(set(components['signal_types']))
        
        return components
