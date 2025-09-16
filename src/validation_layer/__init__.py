"""
Phase 4: Validation and Enrichment Layer

This module provides comprehensive validation, risk assessment, and manipulation detection
for cryptocurrency trading signals.
"""

from .market_validator import MarketDataValidator
from .risk_assessment import RiskAssessmentModule
from .manipulation_detector import ManipulationDetector
from .performance_analyzer import HistoricalPerformanceAnalyzer
from .justification_generator import JustificationGenerator

__all__ = [
    'MarketDataValidator',
    'RiskAssessmentModule',
    'ManipulationDetector',
    'HistoricalPerformanceAnalyzer',
    'JustificationGenerator'
]
