"""
Justification Generator Module

Generates comprehensive, multi-tiered explanations for trading signals
with support for multiple languages and user expertise levels.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import json

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ExpertiseLevel(Enum):
    """User expertise levels for explanations."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class Language(Enum):
    """Supported languages for explanations."""
    ENGLISH = "en"
    RUSSIAN = "ru"
    CHINESE = "zh"
    SPANISH = "es"


@dataclass
class Justification:
    """Multi-tiered justification for a trading signal."""
    novice_explanation: str
    intermediate_explanation: str
    expert_explanation: str
    
    # Key points
    key_factors: List[str]
    risk_warnings: List[str]
    opportunities: List[str]
    
    # Technical details
    technical_analysis: Dict[str, Any]
    market_context: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    
    # Confidence and reasoning
    confidence_reasoning: str
    decision_rationale: str
    
    # Structured data for visualization
    decision_tree: Dict[str, Any]
    supporting_evidence: List[Dict[str, Any]]


class JustificationGenerator:
    """Generates comprehensive justifications for trading decisions."""
    
    def __init__(self):
        """Initialize justification generator."""
        self.templates = self._load_templates()
        self.technical_terms = self._load_technical_glossary()
        
    async def generate_justification(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any],
        language: Language = Language.ENGLISH
    ) -> Justification:
        """
        Generate comprehensive multi-tiered justification.
        
        Args:
            signal: Trading signal data
            validation_results: Results from all validation modules
            language: Target language for explanations
            
        Returns:
            Justification with multi-level explanations
        """
        # Extract validation components
        market_validation = validation_results.get('market_validation', {})
        risk_assessment = validation_results.get('risk_assessment', {})
        manipulation_detection = validation_results.get('manipulation_detection', {})
        performance_metrics = validation_results.get('performance_metrics', {})
        
        # Generate key factors
        key_factors = await self._identify_key_factors(
            signal, market_validation, risk_assessment
        )
        
        # Generate risk warnings
        risk_warnings = await self._generate_risk_warnings(
            risk_assessment, manipulation_detection, market_validation
        )
        
        # Identify opportunities
        opportunities = await self._identify_opportunities(
            signal, market_validation, performance_metrics
        )
        
        # Compile technical analysis
        technical_analysis = await self._compile_technical_analysis(
            signal, validation_results
        )
        
        # Compile market context
        market_context = await self._compile_market_context(
            market_validation, signal
        )
        
        # Compile risk metrics
        risk_metrics = await self._compile_risk_metrics(
            risk_assessment, manipulation_detection
        )
        
        # Generate confidence reasoning
        confidence_reasoning = await self._generate_confidence_reasoning(
            signal, validation_results
        )
        
        # Generate decision rationale
        decision_rationale = await self._generate_decision_rationale(
            signal, validation_results
        )
        
        # Build decision tree
        decision_tree = await self._build_decision_tree(
            signal, validation_results
        )
        
        # Compile supporting evidence
        supporting_evidence = await self._compile_supporting_evidence(
            validation_results
        )
        
        # Generate tiered explanations
        novice_explanation = await self._generate_novice_explanation(
            signal, key_factors, risk_warnings, opportunities, language
        )
        
        intermediate_explanation = await self._generate_intermediate_explanation(
            signal, technical_analysis, market_context, risk_metrics, language
        )
        
        expert_explanation = await self._generate_expert_explanation(
            signal, validation_results, decision_tree, language
        )
        
        return Justification(
            novice_explanation=novice_explanation,
            intermediate_explanation=intermediate_explanation,
            expert_explanation=expert_explanation,
            key_factors=key_factors,
            risk_warnings=risk_warnings,
            opportunities=opportunities,
            technical_analysis=technical_analysis,
            market_context=market_context,
            risk_metrics=risk_metrics,
            confidence_reasoning=confidence_reasoning,
            decision_rationale=decision_rationale,
            decision_tree=decision_tree,
            supporting_evidence=supporting_evidence
        )
    
    async def _generate_novice_explanation(
        self,
        signal: Dict[str, Any],
        key_factors: List[str],
        risk_warnings: List[str],
        opportunities: List[str],
        language: Language
    ) -> str:
        """
        Generate explanation for novice traders (2-3 sentences).
        
        Simple, clear language without technical jargon.
        Focus on what to do and main risks.
        """
        pair = signal.get('pair', 'the asset')
        direction = signal.get('direction', 'buy')
        confidence = signal.get('confidence_score', 50)
        
        # Base template for novice
        if language == Language.ENGLISH:
            explanation = f"This is a {direction} signal for {pair} with {confidence}% confidence. "
            
            if confidence >= 70:
                explanation += f"The main reasons to consider this trade are: {', '.join(key_factors[:2])}. "
            else:
                explanation += f"This signal has moderate confidence due to: {', '.join(risk_warnings[:2])}. "
                
            if risk_warnings:
                explanation += f"Be careful of: {risk_warnings[0]}. "
                
            explanation += "Always use stop-loss orders and only risk what you can afford to lose."
            
        elif language == Language.RUSSIAN:
            direction_ru = "покупки" if direction == "long" else "продажи"
            explanation = f"Это сигнал {direction_ru} для {pair} с уверенностью {confidence}%. "
            
            if confidence >= 70:
                explanation += f"Основные причины рассмотреть эту сделку: {', '.join(key_factors[:2])}. "
            else:
                explanation += f"Сигнал имеет среднюю уверенность из-за: {', '.join(risk_warnings[:2])}. "
                
            if risk_warnings:
                explanation += f"Будьте осторожны с: {risk_warnings[0]}. "
                
            explanation += "Всегда используйте стоп-лосс и рискуйте только тем, что можете позволить себе потерять."
            
        else:
            # Default to English for unsupported languages
            explanation = await self._generate_novice_explanation(
                signal, key_factors, risk_warnings, opportunities, Language.ENGLISH
            )
            
        return explanation
    
    async def _generate_intermediate_explanation(
        self,
        signal: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        market_context: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        language: Language
    ) -> str:
        """
        Generate explanation for intermediate traders (1-2 paragraphs).
        
        Include technical context, risk/reward, and market conditions.
        """
        pair = signal.get('pair', 'the asset')
        entry = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profits = signal.get('take_profits', [])
        
        if language == Language.ENGLISH:
            # Technical setup
            explanation = f"**Technical Setup**: {pair} is showing a {signal.get('direction', 'bullish')} setup "
            explanation += f"with entry at ${entry:,.2f}, stop-loss at ${stop_loss:,.2f} "
            
            if take_profits:
                explanation += f"and take-profit targets at ${', '.join(f'{tp:,.2f}' for tp in take_profits[:3])}. "
            
            # Risk metrics
            risk_reward = risk_metrics.get('risk_reward_ratio', 0)
            position_size = risk_metrics.get('recommended_position_size', 0)
            
            explanation += f"\n\n**Risk Management**: Risk/reward ratio is {risk_reward:.2f}:1 "
            explanation += f"with recommended position size of {position_size:.1%} of portfolio. "
            
            # Market context
            spread = market_context.get('spread_percent', 0)
            volume = market_context.get('volume_24h', 0)
            
            explanation += f"Current market spread is {spread:.3f}% with 24h volume of ${volume:,.0f}. "
            
            # Technical indicators
            if technical_analysis:
                rsi = technical_analysis.get('rsi', 0)
                macd = technical_analysis.get('macd_signal', 'neutral')
                
                explanation += f"\n\n**Technical Indicators**: RSI at {rsi:.1f}, MACD showing {macd} signal. "
                
                if rsi > 70:
                    explanation += "Caution: RSI indicates overbought conditions. "
                elif rsi < 30:
                    explanation += "Note: RSI indicates oversold conditions. "
                    
        elif language == Language.RUSSIAN:
            direction_ru = "бычья" if signal.get('direction') == 'long' else "медвежья"
            explanation = f"**Техническая установка**: {pair} показывает {direction_ru} паттерн "
            explanation += f"с входом на ${entry:,.2f}, стоп-лосс на ${stop_loss:,.2f} "
            
            if take_profits:
                explanation += f"и целями на ${', '.join(f'{tp:,.2f}' for tp in take_profits[:3])}. "
                
            # Continue with Russian translation...
            # (Abbreviated for brevity)
            
        else:
            explanation = await self._generate_intermediate_explanation(
                signal, technical_analysis, market_context, risk_metrics, Language.ENGLISH
            )
            
        return explanation
    
    async def _generate_expert_explanation(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any],
        decision_tree: Dict[str, Any],
        language: Language
    ) -> str:
        """
        Generate comprehensive explanation for expert traders.
        
        Full technical breakdown with all metrics and detailed analysis.
        """
        if language == Language.ENGLISH:
            explanation = "# Comprehensive Signal Analysis\n\n"
            
            # Signal Overview
            explanation += "## Signal Parameters\n"
            explanation += f"- **Pair**: {signal.get('pair')}\n"
            explanation += f"- **Direction**: {signal.get('direction')}\n"
            explanation += f"- **Entry**: ${signal.get('entry_price', 0):,.4f}\n"
            explanation += f"- **Stop Loss**: ${signal.get('stop_loss', 0):,.4f}\n"
            explanation += f"- **Take Profits**: {signal.get('take_profits', [])}\n"
            explanation += f"- **Confidence Score**: {signal.get('confidence_score', 0)}%\n\n"
            
            # Market Validation
            market_val = validation_results.get('market_validation', {})
            explanation += "## Market Validation\n"
            explanation += f"- **Price Deviation**: {market_val.get('price_deviation', {}).get('deviation_percent', 0):.2f}%\n"
            explanation += f"- **Spread**: {market_val.get('spread', {}).get('spread_percent', 0):.3f}%\n"
            explanation += f"- **24h Volume**: ${market_val.get('liquidity', {}).get('volume_24h', 0):,.0f}\n"
            explanation += f"- **Slippage Estimate**: {market_val.get('slippage_estimate', 0):.2f}%\n\n"
            
            # Risk Assessment
            risk = validation_results.get('risk_assessment', {})
            explanation += "## Risk Metrics\n"
            explanation += f"- **Kelly Criterion**: {risk.get('position_size_kelly', 0):.4f}\n"
            explanation += f"- **Recommended Position**: {risk.get('position_size_recommended', 0):.2%}\n"
            explanation += f"- **VaR (95%)**: ${risk.get('value_at_risk_95', 0):,.2f}\n"
            explanation += f"- **VaR (99%)**: ${risk.get('value_at_risk_99', 0):,.2f}\n"
            explanation += f"- **Expected Max Drawdown**: {risk.get('max_drawdown_expected', 0):.2%}\n"
            explanation += f"- **Sharpe Ratio**: {risk.get('sharpe_ratio', 0):.3f}\n\n"
            
            # Manipulation Detection
            manip = validation_results.get('manipulation_detection', {})
            if manip.get('manipulation_detected'):
                explanation += "## ⚠️ Manipulation Warning\n"
                explanation += f"- **Type**: {manip.get('manipulation_type')}\n"
                explanation += f"- **Confidence**: {manip.get('confidence_score', 0):.1f}%\n"
                explanation += f"- **Severity**: {manip.get('severity_level')}\n"
                for indicator in manip.get('indicators', []):
                    explanation += f"- {indicator.get('description')}\n"
                explanation += "\n"
            
            # Historical Performance
            perf = validation_results.get('performance_metrics', {})
            if perf:
                explanation += "## Historical Performance\n"
                explanation += f"- **Win Rate**: {perf.get('win_rate', 0):.1%}\n"
                explanation += f"- **Average Return**: {perf.get('average_return', 0):.2f}%\n"
                explanation += f"- **Profit Factor**: {perf.get('profit_factor', 0):.2f}\n"
                explanation += f"- **Max Drawdown**: {perf.get('max_drawdown', 0):.2%}\n\n"
            
            # Decision Tree
            explanation += "## Decision Logic\n"
            explanation += "```\n"
            explanation += self._format_decision_tree(decision_tree)
            explanation += "```\n\n"
            
            # Final Recommendation
            explanation += "## Recommendation\n"
            confidence = signal.get('confidence_score', 0)
            if confidence >= 70:
                explanation += "✅ **EXECUTE**: High confidence signal with favorable risk/reward.\n"
            elif confidence >= 50:
                explanation += "⚠️ **MONITOR**: Moderate confidence. Consider paper trading or reduced position.\n"
            else:
                explanation += "❌ **AVOID**: Low confidence signal with unfavorable conditions.\n"
                
        else:
            # Default to English for expert explanations
            explanation = await self._generate_expert_explanation(
                signal, validation_results, decision_tree, Language.ENGLISH
            )
            
        return explanation
    
    async def _identify_key_factors(
        self,
        signal: Dict[str, Any],
        market_validation: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Identify key positive factors for the trade."""
        factors = []
        
        # Check confidence
        if signal.get('confidence_score', 0) >= 70:
            factors.append("High AI confidence score")
            
        # Check risk/reward
        if risk_assessment.get('risk_reward_ratio', 0) >= 2:
            factors.append("Favorable risk/reward ratio")
            
        # Check market conditions
        if market_validation.get('overall_result') == 'pass':
            factors.append("Good market conditions")
            
        # Check liquidity
        if market_validation.get('liquidity', {}).get('volume_24h', 0) > 1000000:
            factors.append("High liquidity")
            
        # Check technical setup
        if signal.get('technical_score', 0) >= 7:
            factors.append("Strong technical setup")
            
        return factors[:5]  # Return top 5 factors
    
    async def _generate_risk_warnings(
        self,
        risk_assessment: Dict[str, Any],
        manipulation_detection: Dict[str, Any],
        market_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate risk warnings based on validation results."""
        warnings = []
        
        # Check manipulation
        if manipulation_detection.get('manipulation_detected'):
            warnings.append(f"Potential {manipulation_detection.get('manipulation_type', 'market manipulation')} detected")
            
        # Check risk level
        risk_level = risk_assessment.get('risk_level', '')
        if risk_level in ['high', 'extreme']:
            warnings.append(f"{risk_level.capitalize()} risk level")
            
        # Check market conditions
        if market_validation.get('spread', {}).get('spread_percent', 0) > 0.5:
            warnings.append("High spread may impact profitability")
            
        # Check drawdown
        if risk_assessment.get('max_drawdown_expected', 0) > 0.15:
            warnings.append("High expected drawdown")
            
        # Check slippage
        if market_validation.get('slippage_estimate', 0) > 1:
            warnings.append("High slippage expected")
            
        return warnings[:5]  # Return top 5 warnings
    
    async def _identify_opportunities(
        self,
        signal: Dict[str, Any],
        market_validation: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify potential opportunities in the trade."""
        opportunities = []
        
        # Check historical performance
        if performance_metrics.get('win_rate', 0) > 0.6:
            opportunities.append("High historical win rate for similar signals")
            
        # Check market momentum
        if signal.get('momentum_score', 0) > 7:
            opportunities.append("Strong market momentum")
            
        # Check volume surge
        if market_validation.get('volume_spike', False):
            opportunities.append("Increased trading interest")
            
        # Check support/resistance
        if signal.get('near_support', False):
            opportunities.append("Trading near strong support level")
            
        return opportunities[:5]
    
    async def _compile_technical_analysis(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile technical analysis data."""
        return {
            'rsi': signal.get('rsi', 50),
            'macd_signal': signal.get('macd_signal', 'neutral'),
            'moving_averages': signal.get('moving_averages', {}),
            'support_resistance': signal.get('support_resistance', {}),
            'patterns': signal.get('patterns_detected', []),
            'volume_analysis': validation_results.get('volume_analysis', {})
        }
    
    async def _compile_market_context(
        self,
        market_validation: Dict[str, Any],
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile market context information."""
        return {
            'current_price': market_validation.get('current_price', 0),
            'spread_percent': market_validation.get('spread', {}).get('spread_percent', 0),
            'volume_24h': market_validation.get('liquidity', {}).get('volume_24h', 0),
            'price_change_24h': market_validation.get('price_change_24h', 0),
            'market_cap': signal.get('market_cap', 0),
            'market_sentiment': signal.get('market_sentiment', 'neutral')
        }
    
    async def _compile_risk_metrics(
        self,
        risk_assessment: Dict[str, Any],
        manipulation_detection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile risk metrics."""
        return {
            'risk_reward_ratio': risk_assessment.get('risk_reward_ratio', 0),
            'recommended_position_size': risk_assessment.get('position_size_recommended', 0),
            'kelly_criterion': risk_assessment.get('position_size_kelly', 0),
            'var_95': risk_assessment.get('value_at_risk_95', 0),
            'max_drawdown': risk_assessment.get('max_drawdown_expected', 0),
            'manipulation_risk': manipulation_detection.get('risk_score', 0)
        }
    
    async def _generate_confidence_reasoning(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> str:
        """Generate reasoning for confidence score."""
        confidence = signal.get('confidence_score', 0)
        factors = []
        
        # Positive factors
        if validation_results.get('market_validation', {}).get('overall_result') == 'pass':
            factors.append("market conditions are favorable")
        if validation_results.get('risk_assessment', {}).get('risk_level') == 'low':
            factors.append("risk level is acceptable")
            
        # Negative factors
        if validation_results.get('manipulation_detection', {}).get('manipulation_detected'):
            factors.append("potential manipulation was detected")
            
        reasoning = f"The {confidence}% confidence score reflects that "
        reasoning += ", ".join(factors) if factors else "multiple validation factors"
        reasoning += "."
        
        return reasoning
    
    async def _generate_decision_rationale(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> str:
        """Generate rationale for trading decision."""
        confidence = signal.get('confidence_score', 0)
        
        if confidence >= 70:
            rationale = "This signal should be executed based on strong technical setup, "
            rationale += "favorable market conditions, and acceptable risk parameters."
        elif confidence >= 50:
            rationale = "This signal warrants monitoring with potential for execution "
            rationale += "if market conditions improve or with reduced position size."
        else:
            rationale = "This signal should be avoided due to unfavorable conditions, "
            rationale += "high risk factors, or insufficient confidence in the setup."
            
        return rationale
    
    async def _build_decision_tree(
        self,
        signal: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build decision tree for signal evaluation."""
        return {
            'signal_detected': True,
            'validations': {
                'market': validation_results.get('market_validation', {}).get('overall_result', 'fail'),
                'risk': validation_results.get('risk_assessment', {}).get('risk_level', 'unknown'),
                'manipulation': not validation_results.get('manipulation_detection', {}).get('manipulation_detected', False)
            },
            'confidence_score': signal.get('confidence_score', 0),
            'decision': self._determine_decision(signal.get('confidence_score', 0)),
            'alternative_actions': self._get_alternative_actions(signal.get('confidence_score', 0))
        }
    
    async def _compile_supporting_evidence(
        self,
        validation_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile supporting evidence for the decision."""
        evidence = []
        
        # Add market validation evidence
        if 'market_validation' in validation_results:
            evidence.append({
                'type': 'market_validation',
                'weight': 0.3,
                'result': validation_results['market_validation'].get('overall_result'),
                'details': validation_results['market_validation'].get('details', {})
            })
            
        # Add risk assessment evidence
        if 'risk_assessment' in validation_results:
            evidence.append({
                'type': 'risk_assessment',
                'weight': 0.3,
                'result': validation_results['risk_assessment'].get('risk_level'),
                'details': validation_results['risk_assessment']
            })
            
        # Add manipulation detection evidence
        if 'manipulation_detection' in validation_results:
            evidence.append({
                'type': 'manipulation_detection',
                'weight': 0.2,
                'result': 'clean' if not validation_results['manipulation_detection'].get('manipulation_detected') else 'detected',
                'details': validation_results['manipulation_detection']
            })
            
        # Add performance evidence
        if 'performance_metrics' in validation_results:
            evidence.append({
                'type': 'historical_performance',
                'weight': 0.2,
                'result': 'positive' if validation_results['performance_metrics'].get('win_rate', 0) > 0.5 else 'negative',
                'details': validation_results['performance_metrics']
            })
            
        return evidence
    
    def _format_decision_tree(self, tree: Dict[str, Any]) -> str:
        """Format decision tree for display."""
        formatted = "Signal Detected\n"
        formatted += "├── Market Validation: " + tree['validations']['market'] + "\n"
        formatted += "├── Risk Assessment: " + tree['validations']['risk'] + "\n"
        formatted += "├── Manipulation Check: " + ("Pass" if tree['validations']['manipulation'] else "Fail") + "\n"
        formatted += "├── Confidence Score: " + str(tree['confidence_score']) + "%\n"
        formatted += "└── Decision: " + tree['decision'] + "\n"
        
        if tree['alternative_actions']:
            formatted += "    └── Alternatives: " + ", ".join(tree['alternative_actions'])
            
        return formatted
    
    def _determine_decision(self, confidence: float) -> str:
        """Determine trading decision based on confidence."""
        if confidence >= 70:
            return "EXECUTE"
        elif confidence >= 50:
            return "MONITOR"
        else:
            return "AVOID"
    
    def _get_alternative_actions(self, confidence: float) -> List[str]:
        """Get alternative actions based on confidence."""
        if confidence >= 70:
            return ["Scale in gradually", "Set tight stop-loss"]
        elif confidence >= 50:
            return ["Paper trade", "Wait for confirmation", "Reduce position size"]
        else:
            return ["Find better opportunity", "Wait for market conditions to improve"]
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates."""
        # Templates would be loaded from files or database
        return {
            'novice_en': "Basic template for novice English explanation",
            'intermediate_en': "Intermediate template for English",
            'expert_en': "Expert template for English"
        }
    
    def _load_technical_glossary(self) -> Dict[str, Dict[str, str]]:
        """Load technical term glossary for different languages."""
        return {
            'rsi': {
                'en': 'Relative Strength Index',
                'ru': 'Индекс относительной силы',
                'zh': '相对强弱指数',
                'es': 'Índice de Fuerza Relativa'
            },
            'macd': {
                'en': 'Moving Average Convergence Divergence',
                'ru': 'Схождение/расхождение скользящих средних',
                'zh': '移动平均收敛散度',
                'es': 'Convergencia/Divergencia de Medias Móviles'
            }
        }
