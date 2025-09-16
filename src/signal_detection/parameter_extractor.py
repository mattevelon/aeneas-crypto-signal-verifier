"""Signal Parameter Extractor for parsing trading signal components."""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Structured trading signal with all parameters."""
    pair: str
    direction: str  # long/short
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    leverage: Optional[int] = None
    position_size: Optional[float] = None
    timeframe: Optional[str] = None
    signal_type: Optional[str] = None  # scalp/swing/position
    confidence: float = 0.0
    risk_reward_ratio: Optional[Tuple[float, float]] = None
    timestamp: datetime = None
    source_message: str = ""
    metadata: Dict[str, Any] = None


class SignalParameterExtractor:
    """
    Extracts and normalizes trading parameters from detected signals.
    Processing time target: <100ms per signal.
    """
    
    def __init__(self):
        """Initialize parameter extractor with normalization rules."""
        self.pair_normalizer = self._initialize_pair_normalizer()
        self.exchange_suffixes = ['PERP', 'USD', 'USDT', 'BUSD', 'TUSD', 'USDC']
        logger.info("Initialized signal parameter extractor")
    
    def _initialize_pair_normalizer(self) -> Dict[str, str]:
        """Initialize cryptocurrency pair normalization mappings."""
        return {
            # Common variations
            'BITCOIN': 'BTC',
            'ETHEREUM': 'ETH',
            'RIPPLE': 'XRP',
            'LITECOIN': 'LTC',
            'BITCOINCASH': 'BCH',
            'CHAINLINK': 'LINK',
            'POLYGON': 'MATIC',
            'AVALANCHE': 'AVAX',
            'FANTOM': 'FTM',
            'SANDBOX': 'SAND',
            'DECENTRALAND': 'MANA',
            'AXIEINFINITY': 'AXS',
            'AXIE': 'AXS',
            # Stablecoins
            'TETHER': 'USDT',
            'CIRCLE': 'USDC',
            'BINANCEUSD': 'BUSD',
            'DAI': 'DAI',
            'TRUEUSD': 'TUSD',
        }
    
    def extract_parameters(self, text: str, detected_signals: List[Dict[str, Any]]) -> Optional[TradingSignal]:
        """
        Extract structured trading parameters from text and detected signals.
        
        Args:
            text: Original message text
            detected_signals: Signals detected by pattern engine
            
        Returns:
            Structured TradingSignal or None if extraction fails
        """
        try:
            # Extract components from detected signals
            components = self._aggregate_components(detected_signals)
            
            # Validate required fields
            if not self._validate_minimum_requirements(components):
                logger.debug("Signal doesn't meet minimum requirements")
                return None
            
            # Build trading signal
            signal = self._build_trading_signal(text, components)
            
            # Post-process and validate
            signal = self._post_process_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return None
    
    def _aggregate_components(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate components from multiple pattern matches."""
        components = {
            'pairs': [],
            'directions': [],
            'entry_prices': [],
            'stop_losses': [],
            'take_profits': [],
            'leverage': None,
            'position_size': None,
            'timeframe': None,
            'signal_types': [],
            'risk_reward': None
        }
        
        for signal in signals:
            pattern_name = signal['pattern_name']
            groups = signal.get('groups', [])
            match_text = signal.get('match', '')
            
            # Direction detection
            if 'buy' in pattern_name.lower() or 'long' in pattern_name.lower():
                components['directions'].append('long')
            elif 'sell' in pattern_name.lower() or 'short' in pattern_name.lower():
                components['directions'].append('short')
            
            # Price extraction
            if groups:
                self._extract_prices_from_groups(pattern_name, groups, components)
            
            # Additional extraction from match text
            self._extract_from_match_text(pattern_name, match_text, components)
        
        return components
    
    def _extract_prices_from_groups(self, pattern_name: str, groups: Tuple, components: Dict[str, Any]):
        """Extract price values from regex groups."""
        for i, group in enumerate(groups):
            if not group:
                continue
                
            # Try to parse as number
            try:
                value = self._parse_number(group)
                
                if 'entry' in pattern_name.lower():
                    components['entry_prices'].append(value)
                elif 'stop' in pattern_name.lower() or 'sl' in pattern_name.lower():
                    if '%' not in str(group):
                        components['stop_losses'].append(value)
                elif 'profit' in pattern_name.lower() or 'tp' in pattern_name.lower() or 'target' in pattern_name.lower():
                    components['take_profits'].append(value)
                elif 'leverage' in pattern_name.lower():
                    components['leverage'] = int(value)
                    
            except (ValueError, TypeError):
                # Not a number, check if it's a pair
                if '/' in group or self._is_crypto_pair(group):
                    components['pairs'].append(self._normalize_pair(group))
    
    def _extract_from_match_text(self, pattern_name: str, match_text: str, components: Dict[str, Any]):
        """Extract additional information from matched text."""
        # Signal type extraction
        if 'scalp' in pattern_name.lower():
            components['signal_types'].append('scalp')
        elif 'swing' in pattern_name.lower():
            components['signal_types'].append('swing')
        elif 'position' in pattern_name.lower():
            components['signal_types'].append('position')
        
        # Timeframe extraction
        if 'timeframe' in pattern_name:
            components['timeframe'] = match_text
        
        # Risk/Reward extraction
        if 'risk_reward' in pattern_name:
            rr_match = re.search(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', match_text)
            if rr_match:
                components['risk_reward'] = (
                    float(rr_match.group(1)),
                    float(rr_match.group(2))
                )
    
    def _validate_minimum_requirements(self, components: Dict[str, Any]) -> bool:
        """Check if components meet minimum requirements for valid signal."""
        # Must have at least a pair or entry price
        has_pair = bool(components['pairs'])
        has_entry = bool(components['entry_prices'])
        has_direction = bool(components['directions'])
        
        return (has_pair or has_entry) and has_direction
    
    def _build_trading_signal(self, text: str, components: Dict[str, Any]) -> TradingSignal:
        """Build TradingSignal object from components."""
        # Determine pair
        pair = self._determine_pair(text, components['pairs'])
        
        # Determine direction
        direction = self._determine_direction(components['directions'])
        
        # Determine prices
        entry_price = self._determine_entry_price(components['entry_prices'])
        stop_loss = self._determine_stop_loss(components['stop_losses'], entry_price, direction)
        take_profits = self._determine_take_profits(components['take_profits'], entry_price, direction)
        
        # Determine signal type
        signal_type = components['signal_types'][0] if components['signal_types'] else None
        
        # Build signal
        signal = TradingSignal(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            leverage=components['leverage'],
            position_size=components['position_size'],
            timeframe=components['timeframe'],
            signal_type=signal_type,
            risk_reward_ratio=components['risk_reward'],
            timestamp=datetime.now(),
            source_message=text[:500],  # Store first 500 chars
            metadata={}
        )
        
        return signal
    
    def _determine_pair(self, text: str, detected_pairs: List[str]) -> str:
        """Determine the trading pair from detected pairs or text."""
        if detected_pairs:
            # Return most common pair if multiple detected
            return max(set(detected_pairs), key=detected_pairs.count)
        
        # Try to extract from text directly
        pair = self._extract_pair_from_text(text)
        return pair if pair else "UNKNOWN/USDT"
    
    def _extract_pair_from_text(self, text: str) -> Optional[str]:
        """Extract trading pair directly from text."""
        # Common pair patterns
        patterns = [
            r'([A-Z]{2,10})/([A-Z]{3,6})',
            r'([A-Z]{2,10})-([A-Z]{3,6})',
            r'([A-Z]{2,10})([A-Z]{3,6})(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                base = match.group(1)
                quote = match.group(2) if len(match.groups()) > 1 else 'USDT'
                return self._normalize_pair(f"{base}/{quote}")
        
        return None
    
    def _normalize_pair(self, pair: str) -> str:
        """Normalize cryptocurrency pair format."""
        # Remove spaces and convert to uppercase
        pair = pair.upper().replace(' ', '')
        
        # Handle different separators
        for sep in ['-', '_', '/']:
            if sep in pair:
                parts = pair.split(sep)
                if len(parts) == 2:
                    base, quote = parts
                    break
        else:
            # No separator, try to split based on known quotes
            base, quote = self._split_pair_without_separator(pair)
        
        # Normalize base and quote
        base = self.pair_normalizer.get(base, base)
        quote = self.pair_normalizer.get(quote, quote)
        
        # Ensure quote is a valid stablecoin/quote currency
        if quote not in ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB']:
            if base in ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB']:
                # Swap if reversed
                base, quote = quote, base
            else:
                # Default to USDT if unknown
                quote = 'USDT'
        
        return f"{base}/{quote}"
    
    def _split_pair_without_separator(self, pair: str) -> Tuple[str, str]:
        """Split pair without separator based on known patterns."""
        # Check for known suffixes
        for suffix in self.exchange_suffixes:
            if pair.endswith(suffix):
                return pair[:-len(suffix)], suffix
        
        # Default split (assume last 3-4 chars are quote)
        if len(pair) > 6:
            return pair[:-4], pair[-4:]
        elif len(pair) > 5:
            return pair[:-3], pair[-3:]
        else:
            return pair, 'USDT'
    
    def _is_crypto_pair(self, text: str) -> bool:
        """Check if text represents a crypto pair."""
        text = text.upper()
        # Check for known bases and quotes
        known_cryptos = set(self.pair_normalizer.keys()) | set(self.pair_normalizer.values())
        return any(crypto in text for crypto in known_cryptos)
    
    def _determine_direction(self, directions: List[str]) -> str:
        """Determine signal direction from detected directions."""
        if not directions:
            return 'long'  # Default to long
        
        # Return most common direction
        return max(set(directions), key=directions.count)
    
    def _determine_entry_price(self, entry_prices: List[float]) -> float:
        """Determine entry price from detected prices."""
        if not entry_prices:
            return 0.0
        
        # Use average if multiple prices detected
        return sum(entry_prices) / len(entry_prices)
    
    def _determine_stop_loss(self, stop_losses: List, entry_price: float, direction: str) -> float:
        """Determine stop loss from detected values."""
        if not stop_losses:
            # Default stop loss (3% from entry)
            return entry_price * 0.97 if direction == 'long' else entry_price * 1.03
        
        # Handle percentage-based stop loss
        if isinstance(stop_losses[0], str) and '%' in stop_losses[0]:
            percentage = float(stop_losses[0].replace('%', ''))
            if direction == 'long':
                return entry_price * (1 - percentage / 100)
            else:
                return entry_price * (1 + percentage / 100)
        
        # Use first stop loss
        return float(stop_losses[0])
    
    def _determine_take_profits(self, take_profits: List[float], entry_price: float, direction: str) -> List[float]:
        """Determine take profit levels from detected values."""
        if not take_profits:
            # Default take profits (1.5%, 3%, 5% from entry)
            if direction == 'long':
                return [
                    entry_price * 1.015,
                    entry_price * 1.03,
                    entry_price * 1.05
                ]
            else:
                return [
                    entry_price * 0.985,
                    entry_price * 0.97,
                    entry_price * 0.95
                ]
        
        # Sort take profits based on direction
        take_profits = sorted(take_profits)
        if direction == 'short':
            take_profits = list(reversed(take_profits))
        
        return take_profits[:5]  # Limit to 5 targets
    
    def _post_process_signal(self, signal: TradingSignal) -> TradingSignal:
        """Post-process and validate signal parameters."""
        # Validate price relationships
        if signal.direction == 'long':
            # For long: stop loss < entry < take profits
            if signal.stop_loss >= signal.entry_price:
                signal.stop_loss = signal.entry_price * 0.97
            
            signal.take_profits = [tp for tp in signal.take_profits if tp > signal.entry_price]
            if not signal.take_profits:
                signal.take_profits = [signal.entry_price * 1.03]
                
        else:  # short
            # For short: stop loss > entry > take profits
            if signal.stop_loss <= signal.entry_price:
                signal.stop_loss = signal.entry_price * 1.03
            
            signal.take_profits = [tp for tp in signal.take_profits if tp < signal.entry_price]
            if not signal.take_profits:
                signal.take_profits = [signal.entry_price * 0.97]
        
        # Calculate risk/reward if not provided
        if not signal.risk_reward_ratio and signal.entry_price > 0:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profits[0] - signal.entry_price) if signal.take_profits else 0
            if risk > 0:
                signal.risk_reward_ratio = (1, round(reward / risk, 2))
        
        # Add metadata
        signal.metadata = {
            'processed_at': datetime.now().isoformat(),
            'price_validated': True,
            'risk_calculated': signal.risk_reward_ratio is not None
        }
        
        return signal
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text, handling various formats."""
        # Remove common separators and convert to float
        text = str(text).replace(',', '').replace(' ', '').replace('$', '')
        return float(text)
