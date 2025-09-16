"""Main Signal Detector module that orchestrates pattern recognition, parameter extraction, and classification."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import time

from .pattern_engine import PatternRecognitionEngine
from .parameter_extractor import SignalParameterExtractor, TradingSignal
from .signal_classifier import SignalClassifier, SignalClassification

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Main signal detection orchestrator that combines pattern recognition,
    parameter extraction, and classification.
    Processing target: <100ms per message.
    """
    
    def __init__(self):
        """Initialize signal detector with all components."""
        self.pattern_engine = PatternRecognitionEngine()
        self.parameter_extractor = SignalParameterExtractor()
        self.signal_classifier = SignalClassifier()
        
        # Performance tracking
        self.processing_times = []
        self.detection_stats = {
            'total_processed': 0,
            'signals_detected': 0,
            'avg_processing_time': 0,
            'avg_confidence': 0
        }
        
        logger.info("Initialized SignalDetector with all components")
    
    async def detect_signal(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Detect and analyze trading signal from text.
        
        Args:
            text: Message text to analyze
            metadata: Optional metadata (channel_id, message_id, timestamp, etc.)
            
        Returns:
            Complete signal analysis or None if no signal detected
        """
        start_time = time.time()
        
        try:
            # Step 1: Pattern recognition
            detected_patterns = self.pattern_engine.detect_signals(text)
            
            if not detected_patterns:
                logger.debug("No patterns detected in message")
                return None
            
            # Step 2: Calculate initial confidence
            pattern_confidence = self.pattern_engine.calculate_confidence(detected_patterns)
            
            # Early exit if confidence too low
            if pattern_confidence < 30:
                logger.debug(f"Pattern confidence too low: {pattern_confidence}")
                return None
            
            # Step 3: Extract parameters
            extracted_components = self.pattern_engine.extract_key_components(detected_patterns)
            trading_signal = self.parameter_extractor.extract_parameters(text, detected_patterns)
            
            if not trading_signal:
                logger.debug("Failed to extract valid trading parameters")
                return None
            
            # Step 4: Classify signal
            classification = self.signal_classifier.classify(
                trading_signal,
                detected_patterns,
                text
            )
            
            # Step 5: Validate classification
            if not self.signal_classifier.validate_classification(classification):
                logger.warning("Signal classification validation failed")
                return None
            
            # Step 6: Build complete signal result
            result = self._build_signal_result(
                trading_signal,
                classification,
                detected_patterns,
                extracted_components,
                metadata
            )
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_statistics(processing_time, classification.confidence_score)
            
            logger.info(f"Signal detected with {classification.confidence_score}% confidence in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in signal detection: {e}", exc_info=True)
            return None
    
    async def batch_detect(self, messages: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Process multiple messages in batch for efficiency.
        
        Args:
            messages: List of messages with 'text' and optional 'metadata'
            
        Returns:
            List of signal results (None for messages without signals)
        """
        tasks = []
        for message in messages:
            text = message.get('text', '')
            metadata = message.get('metadata', {})
            tasks.append(self.detect_signal(text, metadata))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing message {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _build_signal_result(self,
                            trading_signal: TradingSignal,
                            classification: SignalClassification,
                            patterns: List[Dict[str, Any]],
                            components: Dict[str, Any],
                            metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete signal result dictionary."""
        result = {
            'signal_id': self._generate_signal_id(),
            'detected_at': datetime.now().isoformat(),
            
            # Trading parameters
            'trading_params': {
                'pair': trading_signal.pair,
                'direction': trading_signal.direction,
                'entry_price': trading_signal.entry_price,
                'stop_loss': trading_signal.stop_loss,
                'take_profits': trading_signal.take_profits,
                'leverage': trading_signal.leverage,
                'position_size': trading_signal.position_size,
                'timeframe': trading_signal.timeframe,
                'risk_reward_ratio': trading_signal.risk_reward_ratio
            },
            
            # Classification results
            'classification': {
                'signal_type': classification.signal_type,
                'urgency': classification.urgency.value,
                'market_condition': classification.market_condition.value if classification.market_condition else None,
                'quality': classification.quality.value,
                'risk_level': classification.risk_level,
                'confidence_score': classification.confidence_score
            },
            
            # Pattern detection details
            'pattern_analysis': {
                'total_patterns': len(patterns),
                'pattern_types': list(set(p['pattern_name'] for p in patterns)),
                'pattern_confidence': components.get('confidence', 0),
                'high_confidence_patterns': len([p for p in patterns if p.get('confidence_weight', 0) > 0.8])
            },
            
            # Source information
            'source': {
                'message': trading_signal.source_message,
                'metadata': metadata or {},
                'processing_version': '1.0.0'
            },
            
            # Validation status
            'validation': {
                'is_valid': True,
                'has_required_fields': all([
                    trading_signal.pair != "UNKNOWN/USDT",
                    trading_signal.entry_price > 0,
                    trading_signal.stop_loss > 0,
                    len(trading_signal.take_profits) > 0
                ]),
                'risk_reward_valid': trading_signal.risk_reward_ratio is not None,
                'classification_valid': True
            }
        }
        
        return result
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID."""
        import uuid
        return f"sig_{uuid.uuid4().hex[:12]}"
    
    def _update_statistics(self, processing_time: float, confidence: float):
        """Update detection statistics."""
        self.detection_stats['total_processed'] += 1
        self.detection_stats['signals_detected'] += 1
        
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.detection_stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
        
        # Update average confidence
        prev_avg = self.detection_stats['avg_confidence']
        n = self.detection_stats['signals_detected']
        self.detection_stats['avg_confidence'] = ((prev_avg * (n - 1)) + confidence) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.detection_stats,
            'detection_rate': (
                self.detection_stats['signals_detected'] / self.detection_stats['total_processed'] * 100
                if self.detection_stats['total_processed'] > 0 else 0
            ),
            'performance': {
                'avg_processing_ms': round(self.detection_stats['avg_processing_time'], 2),
                'max_processing_ms': round(max(self.processing_times), 2) if self.processing_times else 0,
                'min_processing_ms': round(min(self.processing_times), 2) if self.processing_times else 0,
                'target_met': self.detection_stats['avg_processing_time'] < 100  # Target: <100ms
            }
        }
    
    async def validate_signal(self, signal: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a detected signal for completeness and consistency.
        
        Args:
            signal: Signal to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not signal.get('trading_params'):
            issues.append("Missing trading parameters")
            return False, issues
        
        params = signal['trading_params']
        
        # Validate pair
        if not params.get('pair') or params['pair'] == "UNKNOWN/USDT":
            issues.append("Invalid or missing trading pair")
        
        # Validate prices
        if not params.get('entry_price') or params['entry_price'] <= 0:
            issues.append("Invalid entry price")
        
        if not params.get('stop_loss') or params['stop_loss'] <= 0:
            issues.append("Invalid stop loss")
        
        if not params.get('take_profits') or len(params['take_profits']) == 0:
            issues.append("Missing take profit levels")
        
        # Validate price relationships
        if params.get('direction') == 'long':
            if params.get('stop_loss', 0) >= params.get('entry_price', 0):
                issues.append("Stop loss should be below entry for long position")
            
            for tp in params.get('take_profits', []):
                if tp <= params.get('entry_price', 0):
                    issues.append("Take profit should be above entry for long position")
                    break
        
        elif params.get('direction') == 'short':
            if params.get('stop_loss', 0) <= params.get('entry_price', 0):
                issues.append("Stop loss should be above entry for short position")
            
            for tp in params.get('take_profits', []):
                if tp >= params.get('entry_price', 0):
                    issues.append("Take profit should be below entry for short position")
                    break
        
        # Validate risk/reward
        if params.get('risk_reward_ratio'):
            risk, reward = params['risk_reward_ratio']
            if reward / risk < 1:
                issues.append("Risk/reward ratio is below 1:1")
        
        # Validate classification
        classification = signal.get('classification', {})
        if classification.get('confidence_score', 0) < 30:
            issues.append("Confidence score too low")
        
        if classification.get('quality') == 'poor':
            issues.append("Signal quality rated as poor")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def filter_signals(self, 
                      signals: List[Dict[str, Any]], 
                      min_confidence: float = 50,
                      quality_threshold: str = 'fair',
                      risk_levels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter signals based on criteria.
        
        Args:
            signals: List of detected signals
            min_confidence: Minimum confidence score
            quality_threshold: Minimum quality rating
            risk_levels: Acceptable risk levels (default: all)
            
        Returns:
            Filtered list of signals
        """
        if risk_levels is None:
            risk_levels = ['low', 'medium', 'high']
        
        quality_order = ['poor', 'fair', 'good', 'excellent']
        min_quality_index = quality_order.index(quality_threshold)
        
        filtered = []
        for signal in signals:
            classification = signal.get('classification', {})
            
            # Check confidence
            if classification.get('confidence_score', 0) < min_confidence:
                continue
            
            # Check quality
            signal_quality = classification.get('quality', 'poor')
            if quality_order.index(signal_quality) < min_quality_index:
                continue
            
            # Check risk level
            if classification.get('risk_level') not in risk_levels:
                continue
            
            filtered.append(signal)
        
        return filtered
