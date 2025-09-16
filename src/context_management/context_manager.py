"""Main Context Manager that orchestrates all context building components."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

from .historical_aggregator import HistoricalDataAggregator, HistoricalContext
from .market_integration import MarketDataIntegrator, MarketData
from .technical_indicators import TechnicalIndicatorService, TechnicalAnalysis
from .cross_channel_validator import CrossChannelValidator, CrossChannelValidation
from src.core.redis_client import get_redis

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Orchestrates context building for signal analysis.
    Aggregates historical, market, technical, and cross-channel data.
    Implements 8000 token budget management for LLM context.
    """
    
    def __init__(self, max_token_budget: int = 8000):
        """
        Initialize context manager with all components.
        
        Args:
            max_token_budget: Maximum tokens for LLM context
        """
        self.historical_aggregator = HistoricalDataAggregator()
        self.market_integrator = MarketDataIntegrator()
        self.technical_service = TechnicalIndicatorService()
        self.cross_validator = CrossChannelValidator()
        
        self.max_token_budget = max_token_budget
        self.redis_client = None  # Will use get_redis() when needed
        self.cache_ttl = 3600  # 1 hour
        
        # Token estimates per component (rough estimates)
        self.token_estimates = {
            'historical': 2000,
            'market': 1000,
            'technical': 1500,
            'cross_channel': 1000,
            'signal': 500,
            'metadata': 500
        }
        
        logger.info(f"Initialized ContextManager with {max_token_budget} token budget")
    
    async def build_context(self,
                           signal_params: Dict[str, Any],
                           channel_id: Optional[int] = None,
                           include_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build comprehensive context for signal analysis.
        
        Args:
            signal_params: Signal parameters (from signal detector)
            channel_id: Source channel ID
            include_components: Optional list of components to include
            
        Returns:
            Complete context for LLM analysis
        """
        # Default to all components
        if include_components is None:
            include_components = ['historical', 'market', 'technical', 'cross_channel']
        
        # Extract key parameters
        pair = signal_params.get('trading_params', {}).get('pair', 'BTC/USDT')
        direction = signal_params.get('trading_params', {}).get('direction', 'long')
        entry_price = signal_params.get('trading_params', {}).get('entry_price', 0)
        
        # Check cache
        cache_key = f"context:{pair}:{direction}:{entry_price}"
        cached_context = await self._get_cached_context(cache_key)
        if cached_context:
            return cached_context
        
        # Build context components in parallel
        tasks = []
        component_names = []
        
        if 'historical' in include_components:
            tasks.append(self.historical_aggregator.aggregate_context(pair, channel_id))
            component_names.append('historical')
        
        if 'market' in include_components:
            tasks.append(self.market_integrator.get_market_data(pair))
            component_names.append('market')
        
        if 'technical' in include_components:
            tasks.append(self.technical_service.analyze(pair))
            component_names.append('technical')
        
        if 'cross_channel' in include_components and channel_id:
            tasks.append(self.cross_validator.validate_signal(
                pair, direction, entry_price, channel_id
            ))
            component_names.append('cross_channel')
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build context dictionary
        context = {
            'signal': signal_params,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        for name, result in zip(component_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error building {name} context: {result}")
                context['components'][name] = {'error': str(result)}
            else:
                context['components'][name] = self._serialize_component(name, result)
        
        # Optimize context for token budget
        context = self._optimize_for_tokens(context)
        
        # Add summary and insights
        context['summary'] = self._generate_context_summary(context)
        context['key_insights'] = self._extract_key_insights(context)
        
        # Cache the context
        await self._cache_context(cache_key, context)
        
        return context
    
    def _serialize_component(self, name: str, component: Any) -> Dict[str, Any]:
        """Serialize component data for context."""
        if name == 'historical':
            return self._serialize_historical(component)
        elif name == 'market':
            return self._serialize_market(component)
        elif name == 'technical':
            return self._serialize_technical(component)
        elif name == 'cross_channel':
            return self._serialize_cross_channel(component)
        return {}
    
    def _serialize_historical(self, historical: HistoricalContext) -> Dict[str, Any]:
        """Serialize historical context."""
        return {
            'similar_signals': historical.similar_signals[:5],  # Limit to top 5
            'channel_performance': historical.channel_performance,
            'pair_statistics': historical.pair_statistics,
            'time_patterns': {
                'peak_hour': historical.time_patterns.get('peak_hour'),
                'signal_frequency': historical.time_patterns.get('signal_frequency'),
                'active_hours': historical.time_patterns.get('active_hours', [])
            },
            'anomalies': historical.anomalies[:3],  # Limit to top 3
            'performance_metrics': historical.performance_metrics
        }
    
    def _serialize_market(self, market: MarketData) -> Dict[str, Any]:
        """Serialize market data."""
        return {
            'current_price': market.current_price,
            'bid_ask': {
                'bid': market.bid,
                'ask': market.ask,
                'spread': market.spread,
                'spread_percentage': market.spread_percentage
            },
            'volume': {
                '24h': market.volume_24h,
                'quote_24h': market.volume_quote_24h
            },
            'price_change': {
                '24h': market.price_change_24h,
                'percentage_24h': market.price_change_percentage_24h
            },
            'liquidity_score': market.liquidity_score,
            'volatility': market.volatility,
            'correlations': market.correlation_data
        }
    
    def _serialize_technical(self, technical: TechnicalAnalysis) -> Dict[str, Any]:
        """Serialize technical analysis."""
        return {
            'timeframe': technical.timeframe,
            'key_indicators': {
                'rsi': technical.indicators.get('rsi'),
                'macd': technical.indicators.get('macd'),
                'macd_signal': technical.indicators.get('macd_signal'),
                'bb_position': technical.indicators.get('bb_position')
            },
            'moving_averages': {
                'sma_20': technical.moving_averages.get('sma_20'),
                'sma_50': technical.moving_averages.get('sma_50'),
                'sma_200': technical.moving_averages.get('sma_200'),
                'golden_cross': technical.moving_averages.get('golden_cross', False)
            },
            'support_resistance': technical.support_resistance,
            'patterns': technical.patterns[:3],  # Limit to top 3
            'divergences': technical.divergences[:2],  # Limit to top 2
            'confluence_score': technical.confluence_score,
            'signal_strength': technical.signal_strength
        }
    
    def _serialize_cross_channel(self, validation: CrossChannelValidation) -> Dict[str, Any]:
        """Serialize cross-channel validation."""
        return {
            'consensus_score': validation.consensus_score,
            'similar_signals_count': len(validation.similar_signals),
            'top_similar_signals': validation.similar_signals[:3],  # Limit to top 3
            'channel_agreement': validation.channel_agreement,
            'temporal_correlation': validation.temporal_correlation,
            'conflicts': validation.conflict_indicators,
            'validation_status': validation.validation_status
        }
    
    def _optimize_for_tokens(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize context to fit within token budget.
        
        Args:
            context: Full context dictionary
            
        Returns:
            Optimized context
        """
        # Estimate current token usage
        context_str = json.dumps(context)
        estimated_tokens = len(context_str) / 4  # Rough estimate: 4 chars per token
        
        if estimated_tokens <= self.max_token_budget:
            return context
        
        # Priority order for trimming
        trim_order = [
            ('components.historical.similar_signals', 3),
            ('components.technical.support_resistance', None),
            ('components.historical.anomalies', 1),
            ('components.cross_channel.top_similar_signals', 1),
            ('components.market.correlations', None)
        ]
        
        # Trim components until under budget
        for path, limit in trim_order:
            if estimated_tokens <= self.max_token_budget:
                break
            
            # Navigate to component
            parts = path.split('.')
            component = context
            for part in parts[:-1]:
                if part in component:
                    component = component[part]
                else:
                    break
            
            # Apply limit or remove
            if parts[-1] in component:
                if limit is None:
                    del component[parts[-1]]
                elif isinstance(component[parts[-1]], list):
                    component[parts[-1]] = component[parts[-1]][:limit]
            
            # Re-estimate tokens
            context_str = json.dumps(context)
            estimated_tokens = len(context_str) / 4
        
        return context
    
    def _generate_context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of context highlights."""
        summary = {
            'market_conditions': [],
            'technical_signals': [],
            'validation_status': None,
            'risk_factors': [],
            'opportunities': []
        }
        
        # Market conditions
        if 'market' in context['components']:
            market = context['components']['market']
            if market.get('volatility', 0) > 50:
                summary['market_conditions'].append('high_volatility')
            if market.get('liquidity_score', 0) < 30:
                summary['risk_factors'].append('low_liquidity')
            if abs(market.get('price_change', {}).get('percentage_24h', 0)) > 5:
                summary['market_conditions'].append('significant_price_movement')
        
        # Technical signals
        if 'technical' in context['components']:
            tech = context['components']['technical']
            summary['technical_signals'].append(tech.get('signal_strength', 'neutral'))
            
            if tech.get('patterns'):
                summary['technical_signals'].extend(tech['patterns'][:2])
            
            if tech.get('divergences'):
                summary['risk_factors'].append('divergence_detected')
        
        # Validation
        if 'cross_channel' in context['components']:
            cross = context['components']['cross_channel']
            summary['validation_status'] = cross.get('validation_status')
            
            if cross.get('conflicts'):
                summary['risk_factors'].extend(cross['conflicts'])
        
        # Historical performance
        if 'historical' in context['components']:
            hist = context['components']['historical']
            perf = hist.get('performance_metrics', {})
            
            if perf.get('win_rate', 0) > 0.6:
                summary['opportunities'].append('high_historical_win_rate')
            if perf.get('sharpe_ratio', 0) > 1.5:
                summary['opportunities'].append('good_risk_adjusted_returns')
        
        return summary
    
    def _extract_key_insights(self, context: Dict[str, Any]) -> List[str]:
        """Extract key insights from context."""
        insights = []
        
        # Market insights
        if 'market' in context['components']:
            market = context['components']['market']
            
            if market.get('spread_percentage', 0) > 0.5:
                insights.append(f"Wide spread of {market['spread_percentage']:.2f}% indicates lower liquidity")
            
            if market.get('volatility', 0) > 75:
                insights.append(f"High volatility ({market['volatility']:.1f}%) suggests increased risk")
        
        # Technical insights
        if 'technical' in context['components']:
            tech = context['components']['technical']
            
            if tech.get('key_indicators', {}).get('rsi', 50) > 70:
                insights.append("RSI indicates overbought conditions")
            elif tech.get('key_indicators', {}).get('rsi', 50) < 30:
                insights.append("RSI indicates oversold conditions")
            
            if tech.get('confluence_score', 0) > 70:
                insights.append(f"Strong technical confluence ({tech['confluence_score']:.0f}/100)")
        
        # Cross-channel insights
        if 'cross_channel' in context['components']:
            cross = context['components']['cross_channel']
            
            if cross.get('consensus_score', 0) > 70:
                insights.append(f"Strong cross-channel consensus ({cross['consensus_score']:.0f}%)")
            elif cross.get('conflicts'):
                insights.append(f"Conflicts detected: {', '.join(cross['conflicts'])}")
        
        # Historical insights
        if 'historical' in context['components']:
            hist = context['components']['historical']
            
            if hist.get('anomalies'):
                insights.append(f"{len(hist['anomalies'])} anomalies detected in recent signals")
            
            perf = hist.get('performance_metrics', {})
            if perf.get('win_rate', 0) > 0:
                insights.append(f"Historical win rate: {perf['win_rate']*100:.1f}%")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def _get_cached_context(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached context if available."""
        if self.redis_client:
            try:
                cached = await self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Cache get error: {e}")
        return None
    
    async def _cache_context(self, key: str, context: Dict[str, Any]):
        """Cache context data."""
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key,
                    self.cache_ttl,
                    json.dumps(context)
                )
            except Exception as e:
                logger.error(f"Cache set error: {e}")
    
    async def get_context_metrics(self) -> Dict[str, Any]:
        """Get metrics about context building performance."""
        return {
            'cache_ttl': self.cache_ttl,
            'max_token_budget': self.max_token_budget,
            'token_estimates': self.token_estimates,
            'components': {
                'historical': 'active',
                'market': 'active',
                'technical': 'active',
                'cross_channel': 'active'
            }
        }
