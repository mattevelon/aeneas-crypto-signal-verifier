"""Signal Detection Module for identifying and extracting trading signals from messages."""

from .pattern_engine import PatternRecognitionEngine
from .parameter_extractor import SignalParameterExtractor
from .signal_classifier import SignalClassifier
from .signal_detector import SignalDetector

__all__ = [
    'PatternRecognitionEngine',
    'SignalParameterExtractor', 
    'SignalClassifier',
    'SignalDetector'
]
