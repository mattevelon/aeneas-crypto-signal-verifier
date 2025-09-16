"""Prompt Engineering System for dynamic prompt generation and optimization."""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for different analysis needs."""
    SIGNAL_ANALYSIS = "signal_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_CONTEXT = "market_context"
    TECHNICAL_ANALYSIS = "technical_analysis"
    JUSTIFICATION = "justification"


@dataclass
class PromptTemplate:
    """Prompt template with versioning and metadata."""
    id: str
    type: PromptType
    version: str
    template: str
    variables: List[str]
    max_tokens: int
    temperature: float
    system_prompt: str
    metadata: Dict[str, Any]


class PromptEngine:
    """
    Dynamic prompt template engine for LLM interactions.
    Implements context injection with 8000 token budget management.
    """
    
    def __init__(self):
        """Initialize prompt engine with templates."""
        self.templates = self._initialize_templates()
        self.template_versions = {}
        self.active_templates = {}
        self.max_context_tokens = 8000
        
        # A/B testing configuration
        self.ab_tests = {}
        self.performance_metrics = {}
        
        logger.info("Initialized PromptEngine with templates")
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize prompt templates for different analysis types."""
        templates = {}
        
        # Signal Analysis Template
        templates['signal_analysis'] = PromptTemplate(
            id='signal_analysis_v1',
            type=PromptType.SIGNAL_ANALYSIS,
            version='1.0.0',
            template="""Analyze the following cryptocurrency trading signal and provide comprehensive verification:

SIGNAL DETAILS:
{signal_details}

MARKET CONTEXT:
{market_context}

TECHNICAL INDICATORS:
{technical_indicators}

HISTORICAL PERFORMANCE:
{historical_data}

CROSS-CHANNEL VALIDATION:
{cross_channel_data}

Please provide:
1. Signal Validity Assessment (0-100 confidence score)
2. Risk Level Analysis (low/medium/high with justification)
3. Entry Optimization (suggested adjustments if any)
4. Stop Loss Validation (is it appropriate?)
5. Take Profit Analysis (are targets realistic?)
6. Market Conditions Assessment
7. Key Risk Factors
8. Trading Recommendations

Format your response as JSON with the following structure:
{{
    "confidence_score": <0-100>,
    "risk_level": "<low/medium/high>",
    "signal_validity": {{
        "is_valid": <true/false>,
        "reasons": []
    }},
    "optimizations": {{
        "entry_price": <suggested_price_or_null>,
        "stop_loss": <suggested_sl_or_null>,
        "take_profits": <array_of_suggested_tps_or_null>
    }},
    "risk_factors": [],
    "recommendations": [],
    "justification": {{
        "novice": "<2-3 sentences>",
        "intermediate": "<1-2 paragraphs>",
        "expert": "<detailed analysis>"
    }}
}}""",
            variables=['signal_details', 'market_context', 'technical_indicators', 
                      'historical_data', 'cross_channel_data'],
            max_tokens=4000,
            temperature=0.3,
            system_prompt="""You are an expert cryptocurrency trading analyst with deep knowledge of technical analysis, 
market dynamics, and risk management. Provide accurate, data-driven analysis with clear justifications. 
Be conservative in risk assessment and prioritize capital preservation.""",
            metadata={'category': 'analysis', 'priority': 'high'}
        )
        
        # Risk Assessment Template
        templates['risk_assessment'] = PromptTemplate(
            id='risk_assessment_v1',
            type=PromptType.RISK_ASSESSMENT,
            version='1.0.0',
            template="""Perform comprehensive risk assessment for this trading signal:

SIGNAL PARAMETERS:
{signal_params}

MARKET VOLATILITY:
{volatility_data}

LIQUIDITY ANALYSIS:
{liquidity_data}

CORRELATION DATA:
{correlation_data}

Provide detailed risk analysis including:
1. Position Sizing (Kelly Criterion)
2. Maximum Drawdown Estimation
3. Risk/Reward Ratio Analysis
4. Black Swan Event Probability
5. Manipulation Risk Assessment
6. Liquidity Risk Evaluation
7. Correlation Risk Analysis

Response format:
{{
    "position_size_percentage": <0-100>,
    "max_drawdown_estimate": <percentage>,
    "risk_reward_ratio": {{
        "risk": <value>,
        "reward": <value>,
        "is_favorable": <true/false>
    }},
    "risk_scores": {{
        "market_risk": <0-100>,
        "liquidity_risk": <0-100>,
        "manipulation_risk": <0-100>,
        "correlation_risk": <0-100>,
        "overall_risk": <0-100>
    }},
    "recommendations": []
}}""",
            variables=['signal_params', 'volatility_data', 'liquidity_data', 'correlation_data'],
            max_tokens=2000,
            temperature=0.2,
            system_prompt="You are a risk management specialist focused on protecting capital and optimizing position sizing.",
            metadata={'category': 'risk', 'priority': 'high'}
        )
        
        # Market Context Template
        templates['market_context'] = PromptTemplate(
            id='market_context_v1',
            type=PromptType.MARKET_CONTEXT,
            version='1.0.0',
            template="""Analyze the current market context for {pair}:

PRICE ACTION:
{price_data}

VOLUME PROFILE:
{volume_data}

ORDER BOOK:
{orderbook_data}

NEWS SENTIMENT:
{news_sentiment}

Provide market context analysis:
1. Current market phase (accumulation/distribution/trend)
2. Institutional activity indicators
3. Retail sentiment assessment
4. Key support/resistance levels
5. Potential catalysts
6. Market manipulation indicators

Response format:
{{
    "market_phase": "<phase>",
    "trend_strength": <0-100>,
    "institutional_activity": "<low/medium/high>",
    "retail_sentiment": "<bearish/neutral/bullish>",
    "key_levels": {{
        "support": [],
        "resistance": []
    }},
    "manipulation_detected": <true/false>,
    "market_summary": "<brief summary>"
}}""",
            variables=['pair', 'price_data', 'volume_data', 'orderbook_data', 'news_sentiment'],
            max_tokens=1500,
            temperature=0.3,
            system_prompt="You are a market analyst specializing in cryptocurrency market microstructure and sentiment analysis.",
            metadata={'category': 'market', 'priority': 'medium'}
        )
        
        # Technical Analysis Template
        templates['technical_analysis'] = PromptTemplate(
            id='technical_analysis_v1',
            type=PromptType.TECHNICAL_ANALYSIS,
            version='1.0.0',
            template="""Perform technical analysis for {pair} on {timeframe}:

INDICATORS:
{indicators}

PATTERNS:
{patterns}

DIVERGENCES:
{divergences}

MULTI-TIMEFRAME:
{mtf_analysis}

Provide:
1. Primary trend direction and strength
2. Key indicator signals
3. Pattern recognition results
4. Divergence analysis
5. Confluence assessment
6. Entry/Exit timing
7. Technical targets

Response format:
{{
    "trend": {{
        "direction": "<up/down/sideways>",
        "strength": <0-100>,
        "timeframe_alignment": <true/false>
    }},
    "signals": {{
        "bullish": [],
        "bearish": [],
        "neutral": []
    }},
    "patterns_identified": [],
    "divergences": [],
    "confluence_score": <0-100>,
    "technical_targets": {{
        "support": [],
        "resistance": [],
        "price_targets": []
    }},
    "timing": "<immediate/wait/avoid>"
}}""",
            variables=['pair', 'timeframe', 'indicators', 'patterns', 'divergences', 'mtf_analysis'],
            max_tokens=2000,
            temperature=0.3,
            system_prompt="You are a technical analysis expert with deep knowledge of chart patterns, indicators, and market structure.",
            metadata={'category': 'technical', 'priority': 'medium'}
        )
        
        # Justification Generation Template
        templates['justification'] = PromptTemplate(
            id='justification_v1',
            type=PromptType.JUSTIFICATION,
            version='1.0.0',
            template="""Generate comprehensive trading signal justification:

SIGNAL ANALYSIS:
{analysis_results}

RISK ASSESSMENT:
{risk_results}

CONFIDENCE SCORE: {confidence_score}

Generate three-tiered justification:

1. NOVICE (2-3 sentences): Simple explanation focusing on direction and basic reasoning
2. INTERMEDIATE (1-2 paragraphs): Include technical context and risk considerations
3. EXPERT (full analysis): Complete breakdown with all metrics and advanced concepts

Also provide:
- Key strengths of the signal
- Main risk factors
- Recommended position management
- Alternative scenarios

Response format:
{{
    "justification": {{
        "novice": "<simple explanation>",
        "intermediate": "<detailed explanation>",
        "expert": "<comprehensive analysis>"
    }},
    "strengths": [],
    "risks": [],
    "position_management": {{
        "entry_strategy": "<strategy>",
        "exit_strategy": "<strategy>",
        "risk_management": "<approach>"
    }},
    "alternative_scenarios": []
}}""",
            variables=['analysis_results', 'risk_results', 'confidence_score'],
            max_tokens=3000,
            temperature=0.4,
            system_prompt="You are an expert trading educator who can explain complex concepts at multiple levels of sophistication.",
            metadata={'category': 'justification', 'priority': 'high'}
        )
        
        return templates
    
    def build_prompt(self, 
                     prompt_type: str,
                     context: Dict[str, Any],
                     version: Optional[str] = None) -> str:
        """
        Build a prompt from template with context injection.
        
        Args:
            prompt_type: Type of prompt to build
            context: Context data to inject
            version: Optional template version
            
        Returns:
            Formatted prompt string
        """
        # Get template
        template = self.templates.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Use specific version if requested
        if version and f"{prompt_type}_v{version}" in self.template_versions:
            template = self.template_versions[f"{prompt_type}_v{version}"]
        
        # Prepare variables
        formatted_vars = {}
        for var in template.variables:
            if var in context:
                # Format context data appropriately
                formatted_vars[var] = self._format_context_variable(var, context[var])
            else:
                formatted_vars[var] = "No data available"
        
        # Build prompt
        try:
            prompt = template.template.format(**formatted_vars)
            
            # Add system prompt
            full_prompt = {
                'system': template.system_prompt,
                'user': prompt,
                'max_tokens': template.max_tokens,
                'temperature': template.temperature
            }
            
            return full_prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def _format_context_variable(self, var_name: str, data: Any) -> str:
        """Format context variable for prompt injection."""
        if isinstance(data, dict):
            # Format as readable JSON
            return json.dumps(data, indent=2, default=str)
        elif isinstance(data, list):
            # Format as bullet points
            if data and isinstance(data[0], dict):
                return json.dumps(data, indent=2, default=str)
            else:
                return '\n'.join(f"- {item}" for item in data)
        else:
            return str(data)
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 8000) -> str:
        """
        Optimize prompt to fit within token budget.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum token budget
            
        Returns:
            Optimized prompt
        """
        # Rough token estimation (4 chars per token)
        estimated_tokens = len(prompt) / 4
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Truncation strategies
        lines = prompt.split('\n')
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Truncate long sections
        optimized_lines = []
        for line in lines:
            if len(line) > 500:
                # Truncate long lines
                optimized_lines.append(line[:500] + '...')
            else:
                optimized_lines.append(line)
        
        optimized = '\n'.join(optimized_lines)
        
        # Check if still over budget
        if len(optimized) / 4 > max_tokens:
            # More aggressive truncation
            max_chars = int(max_tokens * 3.5)  # Conservative estimate
            optimized = optimized[:max_chars] + '\n...[Content truncated to fit token limit]'
        
        return optimized
    
    def create_custom_template(self,
                              id: str,
                              type: PromptType,
                              template: str,
                              variables: List[str],
                              **kwargs) -> PromptTemplate:
        """
        Create a custom prompt template.
        
        Args:
            id: Template ID
            type: Prompt type
            template: Template string
            variables: Required variables
            **kwargs: Additional template parameters
            
        Returns:
            New PromptTemplate
        """
        new_template = PromptTemplate(
            id=id,
            type=type,
            version='custom',
            template=template,
            variables=variables,
            max_tokens=kwargs.get('max_tokens', 2000),
            temperature=kwargs.get('temperature', 0.3),
            system_prompt=kwargs.get('system_prompt', ''),
            metadata=kwargs.get('metadata', {})
        )
        
        # Store custom template
        self.templates[id] = new_template
        
        logger.info(f"Created custom template: {id}")
        return new_template
    
    def ab_test_prompt(self,
                      prompt_type: str,
                      context: Dict[str, Any],
                      variant_a: str,
                      variant_b: str) -> Tuple[str, str]:
        """
        Perform A/B testing on prompt variants.
        
        Args:
            prompt_type: Type of prompt
            context: Context data
            variant_a: First variant ID
            variant_b: Second variant ID
            
        Returns:
            Tuple of (selected_variant, prompt)
        """
        import random
        
        # Random selection for A/B test
        selected = variant_a if random.random() < 0.5 else variant_b
        
        # Track selection
        if prompt_type not in self.ab_tests:
            self.ab_tests[prompt_type] = {
                'variant_a': {'id': variant_a, 'count': 0, 'success': 0},
                'variant_b': {'id': variant_b, 'count': 0, 'success': 0}
            }
        
        if selected == variant_a:
            self.ab_tests[prompt_type]['variant_a']['count'] += 1
        else:
            self.ab_tests[prompt_type]['variant_b']['count'] += 1
        
        # Build prompt with selected variant
        prompt = self.build_prompt(prompt_type, context, selected)
        
        return selected, prompt
    
    def record_prompt_performance(self,
                                 prompt_type: str,
                                 variant: str,
                                 success: bool,
                                 metrics: Optional[Dict[str, Any]] = None):
        """
        Record performance metrics for prompt optimization.
        
        Args:
            prompt_type: Type of prompt
            variant: Variant ID used
            success: Whether the prompt was successful
            metrics: Additional performance metrics
        """
        if prompt_type not in self.performance_metrics:
            self.performance_metrics[prompt_type] = []
        
        self.performance_metrics[prompt_type].append({
            'variant': variant,
            'success': success,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        })
        
        # Update A/B test metrics if applicable
        if prompt_type in self.ab_tests:
            for variant_key, variant_data in self.ab_tests[prompt_type].items():
                if variant_data['id'] == variant and success:
                    variant_data['success'] += 1
    
    def get_best_performing_variant(self, prompt_type: str) -> Optional[str]:
        """
        Get the best performing variant for a prompt type.
        
        Args:
            prompt_type: Type of prompt
            
        Returns:
            Best variant ID or None
        """
        if prompt_type not in self.ab_tests:
            return None
        
        test_data = self.ab_tests[prompt_type]
        
        # Calculate success rates
        rates = {}
        for variant_key, variant_data in test_data.items():
            if variant_data['count'] > 0:
                rates[variant_data['id']] = variant_data['success'] / variant_data['count']
        
        if rates:
            return max(rates, key=rates.get)
        
        return None
    
    def validate_prompt(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a prompt for completeness and correctness.
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        # Check required fields
        if 'system' not in prompt:
            issues.append("Missing system prompt")
        if 'user' not in prompt:
            issues.append("Missing user prompt")
        
        # Check token limits
        if 'max_tokens' in prompt and prompt['max_tokens'] > 4000:
            issues.append("Max tokens exceeds limit (4000)")
        
        # Check temperature
        if 'temperature' in prompt:
            if prompt['temperature'] < 0 or prompt['temperature'] > 1:
                issues.append("Temperature must be between 0 and 1")
        
        # Estimate token usage
        total_text = prompt.get('system', '') + prompt.get('user', '')
        estimated_tokens = len(total_text) / 4
        
        if estimated_tokens > 8000:
            issues.append(f"Estimated tokens ({estimated_tokens:.0f}) exceeds budget (8000)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
