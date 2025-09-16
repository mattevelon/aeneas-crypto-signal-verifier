"""Main AI Analyzer that orchestrates LLM-based signal analysis."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json

from .prompt_engine import PromptEngine, PromptType
from .llm_client import LLMClient, LLMProvider, LLMResponse
from .response_processor import ResponseProcessor, ProcessedResponse
from .token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """
    Main AI analysis orchestrator for signal verification.
    Coordinates prompt generation, LLM interaction, and response processing.
    """
    
    def __init__(self):
        """Initialize AI analyzer with all components."""
        self.prompt_engine = PromptEngine()
        self.llm_client = LLMClient()
        self.response_processor = ResponseProcessor()
        self.token_optimizer = TokenOptimizer()
        
        # Analysis configuration
        self.analysis_types = ['signal_analysis', 'risk_assessment', 'justification']
        self.confidence_threshold = 70  # Minimum confidence for validation
        
        # Performance metrics
        self.analysis_metrics = {
            'total_analyses': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0,
            'avg_latency': 0
        }
        
        logger.info("Initialized AIAnalyzer")
    
    async def analyze_signal(self,
                            signal_data: Dict[str, Any],
                            context: Dict[str, Any],
                            analysis_depth: str = 'full') -> Dict[str, Any]:
        """
        Perform comprehensive AI analysis of a trading signal.
        
        Args:
            signal_data: Signal parameters and detection results
            context: Full context from ContextManager
            analysis_depth: 'quick', 'standard', or 'full'
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # Optimize context for token budget
            optimized_context, token_metrics = self.token_optimizer.optimize_context(context)
            
            # Determine which analyses to perform
            analyses_to_perform = self._determine_analyses(analysis_depth)
            
            # Perform analyses in parallel where possible
            analysis_results = {}
            
            # Primary signal analysis (always performed)
            if 'signal_analysis' in analyses_to_perform:
                signal_result = await self._analyze_signal_validity(
                    signal_data, optimized_context
                )
                analysis_results['signal_analysis'] = signal_result
            
            # Risk assessment (if depth >= standard)
            if 'risk_assessment' in analyses_to_perform:
                risk_result = await self._assess_risk(
                    signal_data, optimized_context, analysis_results.get('signal_analysis')
                )
                analysis_results['risk_assessment'] = risk_result
            
            # Generate justification (always for full analysis)
            if 'justification' in analyses_to_perform:
                justification = await self._generate_justification(
                    analysis_results, signal_data
                )
                analysis_results['justification'] = justification
            
            # Combine results
            final_result = self._combine_results(
                analysis_results,
                signal_data,
                token_metrics
            )
            
            # Update metrics
            self._update_metrics(final_result, start_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            self.analysis_metrics['failed'] += 1
            
            return {
                'status': 'error',
                'error': str(e),
                'signal_data': signal_data,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_signal_validity(self,
                                      signal_data: Dict[str, Any],
                                      context: Dict[str, Any]) -> ProcessedResponse:
        """Analyze signal validity using LLM."""
        # Build prompt
        prompt = self.prompt_engine.build_prompt(
            'signal_analysis',
            {
                'signal_details': json.dumps(signal_data.get('trading_params', {}), indent=2),
                'market_context': json.dumps(context.get('components', {}).get('market', {}), indent=2),
                'technical_indicators': json.dumps(context.get('components', {}).get('technical', {}), indent=2),
                'historical_data': json.dumps(context.get('components', {}).get('historical', {}), indent=2),
                'cross_channel_data': json.dumps(context.get('components', {}).get('cross_channel', {}), indent=2)
            }
        )
        
        # Make LLM request
        async with self.llm_client as client:
            response = await client.analyze_signal(prompt)
        
        # Process response
        processed = self.response_processor.process_response(
            response.content,
            'signal_analysis'
        )
        
        # Post-process
        processed = self.response_processor.post_process(processed)
        
        return processed
    
    async def _assess_risk(self,
                          signal_data: Dict[str, Any],
                          context: Dict[str, Any],
                          signal_analysis: Optional[ProcessedResponse]) -> ProcessedResponse:
        """Perform risk assessment using LLM."""
        # Prepare risk-specific context
        risk_context = {
            'signal_params': signal_data.get('trading_params', {}),
            'volatility_data': {
                'market_volatility': context.get('components', {}).get('market', {}).get('volatility', 0),
                'atr': context.get('components', {}).get('technical', {}).get('indicators', {}).get('atr', 0)
            },
            'liquidity_data': {
                'liquidity_score': context.get('components', {}).get('market', {}).get('liquidity_score', 0),
                'volume_24h': context.get('components', {}).get('market', {}).get('volume', {}).get('24h', 0)
            },
            'correlation_data': context.get('components', {}).get('market', {}).get('correlations', {})
        }
        
        # Build prompt
        prompt = self.prompt_engine.build_prompt('risk_assessment', risk_context)
        
        # Make LLM request
        async with self.llm_client as client:
            response = await client.analyze_signal(prompt)
        
        # Process response
        processed = self.response_processor.process_response(
            response.content,
            'risk_assessment'
        )
        
        return processed
    
    async def _generate_justification(self,
                                     analysis_results: Dict[str, Any],
                                     signal_data: Dict[str, Any]) -> ProcessedResponse:
        """Generate multi-level justification."""
        # Extract key results
        signal_analysis = analysis_results.get('signal_analysis')
        risk_assessment = analysis_results.get('risk_assessment')
        
        # Prepare justification context
        justification_context = {
            'analysis_results': {
                'confidence_score': signal_analysis.confidence_score if signal_analysis else 50,
                'risk_level': signal_analysis.risk_level if signal_analysis else 'medium',
                'signal_validity': signal_analysis.signal_validity if signal_analysis else {},
                'optimizations': signal_analysis.optimizations if signal_analysis else {},
                'risk_factors': signal_analysis.risk_factors if signal_analysis else [],
                'recommendations': signal_analysis.recommendations if signal_analysis else []
            },
            'risk_results': {
                'risk_scores': risk_assessment.metadata.get('risk_scores', {}) if risk_assessment else {},
                'position_size': risk_assessment.metadata.get('position_size_percentage', 0) if risk_assessment else 0
            } if risk_assessment else {},
            'confidence_score': signal_analysis.confidence_score if signal_analysis else 50
        }
        
        # Build prompt
        prompt = self.prompt_engine.build_prompt('justification', justification_context)
        
        # Make LLM request
        async with self.llm_client as client:
            response = await client.analyze_signal(prompt)
        
        # Process response
        processed = self.response_processor.process_response(
            response.content,
            'justification'
        )
        
        return processed
    
    def _determine_analyses(self, depth: str) -> List[str]:
        """Determine which analyses to perform based on depth."""
        if depth == 'quick':
            return ['signal_analysis']
        elif depth == 'standard':
            return ['signal_analysis', 'risk_assessment']
        else:  # full
            return ['signal_analysis', 'risk_assessment', 'justification']
    
    def _combine_results(self,
                        analysis_results: Dict[str, Any],
                        signal_data: Dict[str, Any],
                        token_metrics: Any) -> Dict[str, Any]:
        """Combine all analysis results into final output."""
        # Extract primary results
        signal_analysis = analysis_results.get('signal_analysis')
        risk_assessment = analysis_results.get('risk_assessment')
        justification_result = analysis_results.get('justification')
        
        # Build final result
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal_data.get('signal_id'),
            
            # Primary verdict
            'verdict': {
                'is_valid': signal_analysis.signal_validity.get('is_valid', False) if signal_analysis else False,
                'confidence_score': signal_analysis.confidence_score if signal_analysis else 0,
                'risk_level': signal_analysis.risk_level if signal_analysis else 'high',
                'recommendation': self._generate_recommendation(signal_analysis, risk_assessment)
            },
            
            # Detailed analysis
            'analysis': {
                'signal_validity': signal_analysis.signal_validity if signal_analysis else {},
                'optimizations': signal_analysis.optimizations if signal_analysis else {},
                'risk_factors': signal_analysis.risk_factors if signal_analysis else [],
                'recommendations': signal_analysis.recommendations if signal_analysis else []
            },
            
            # Risk assessment
            'risk': {
                'risk_scores': risk_assessment.metadata.get('risk_scores', {}) if risk_assessment else {},
                'position_size_percentage': risk_assessment.metadata.get('position_size_percentage', 0) if risk_assessment else 0,
                'max_drawdown_estimate': risk_assessment.metadata.get('max_drawdown_estimate', 0) if risk_assessment else 0
            } if risk_assessment else None,
            
            # Justification
            'justification': justification_result.justification if justification_result else {
                'novice': 'Analysis pending',
                'intermediate': 'Analysis pending',
                'expert': 'Analysis pending'
            },
            
            # Metadata
            'metadata': {
                'analyses_performed': list(analysis_results.keys()),
                'token_usage': {
                    'total': token_metrics.total_tokens,
                    'budget': token_metrics.remaining_budget + token_metrics.total_tokens,
                    'remaining': token_metrics.remaining_budget
                },
                'processing_quality': {
                    'signal_analysis': signal_analysis.metadata.get('quality_score', 0) if signal_analysis else 0,
                    'risk_assessment': risk_assessment.metadata.get('quality_score', 0) if risk_assessment else 0,
                    'justification': justification_result.metadata.get('quality_score', 0) if justification_result else 0
                }
            }
        }
        
        return result
    
    def _generate_recommendation(self,
                                signal_analysis: Optional[ProcessedResponse],
                                risk_assessment: Optional[ProcessedResponse]) -> str:
        """Generate final recommendation based on analyses."""
        if not signal_analysis:
            return "insufficient_data"
        
        confidence = signal_analysis.confidence_score
        risk_level = signal_analysis.risk_level
        is_valid = signal_analysis.signal_validity.get('is_valid', False)
        
        if not is_valid:
            return "avoid"
        
        if confidence >= 80 and risk_level == 'low':
            return "strong_buy"
        elif confidence >= 70 and risk_level in ['low', 'medium']:
            return "buy"
        elif confidence >= 60 and risk_level == 'low':
            return "consider"
        elif confidence < 50 or risk_level == 'high':
            return "avoid"
        else:
            return "monitor"
    
    def _update_metrics(self, result: Dict[str, Any], start_time: datetime):
        """Update analysis metrics."""
        self.analysis_metrics['total_analyses'] += 1
        
        if result.get('status') == 'success':
            self.analysis_metrics['successful'] += 1
            
            # Update average confidence
            confidence = result.get('verdict', {}).get('confidence_score', 0)
            n = self.analysis_metrics['successful']
            prev_avg = self.analysis_metrics['avg_confidence']
            self.analysis_metrics['avg_confidence'] = ((prev_avg * (n - 1)) + confidence) / n
        
        # Update latency
        latency = (datetime.now() - start_time).total_seconds()
        n = self.analysis_metrics['total_analyses']
        prev_avg = self.analysis_metrics['avg_latency']
        self.analysis_metrics['avg_latency'] = ((prev_avg * (n - 1)) + latency) / n
    
    async def batch_analyze(self,
                           signals: List[Dict[str, Any]],
                           contexts: List[Dict[str, Any]],
                           analysis_depth: str = 'standard') -> List[Dict[str, Any]]:
        """
        Analyze multiple signals in batch.
        
        Args:
            signals: List of signal data
            contexts: List of contexts (one per signal)
            analysis_depth: Depth of analysis
            
        Returns:
            List of analysis results
        """
        if len(signals) != len(contexts):
            raise ValueError("Number of signals must match number of contexts")
        
        # Process in batches of 5 for efficiency
        batch_size = 5
        results = []
        
        for i in range(0, len(signals), batch_size):
            batch_signals = signals[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            # Analyze batch in parallel
            tasks = []
            for signal, context in zip(batch_signals, batch_contexts):
                tasks.append(self.analyze_signal(signal, context, analysis_depth))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed for signal {i+j}: {result}")
                    results.append({
                        'status': 'error',
                        'error': str(result),
                        'signal_data': batch_signals[j]
                    })
                else:
                    results.append(result)
        
        return results
    
    def validate_analysis(self, analysis: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis results for consistency.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        # Check required fields
        if 'verdict' not in analysis:
            issues.append("Missing verdict")
        
        if 'analysis' not in analysis:
            issues.append("Missing analysis details")
        
        # Check verdict consistency
        verdict = analysis.get('verdict', {})
        if verdict.get('is_valid') and verdict.get('confidence_score', 0) < 30:
            issues.append("Valid signal with very low confidence")
        
        if verdict.get('risk_level') == 'high' and verdict.get('recommendation') in ['buy', 'strong_buy']:
            issues.append("High risk but positive recommendation")
        
        # Check optimization validity
        optimizations = analysis.get('analysis', {}).get('optimizations', {})
        if optimizations:
            if 'entry_price' in optimizations and optimizations['entry_price'] <= 0:
                issues.append("Invalid optimization: negative entry price")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'metrics': self.analysis_metrics,
            'success_rate': (
                self.analysis_metrics['successful'] / self.analysis_metrics['total_analyses']
                if self.analysis_metrics['total_analyses'] > 0 else 0
            ),
            'llm_stats': self.llm_client.get_statistics() if hasattr(self.llm_client, 'get_statistics') else {},
            'processor_stats': self.response_processor.get_statistics()
        }
