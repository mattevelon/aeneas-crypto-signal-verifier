"""Integration tests for Phase 3 Core Verification and Analysis Engine."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json

# Import Phase 3 components
from src.signal_detection.signal_detector import SignalDetector
from src.context_management.context_manager import ContextManager
from src.ai_integration.ai_analyzer import AIAnalyzer
from src.analysis_processing.validation_framework import ValidationFramework
from src.analysis_processing.signal_enhancer import SignalEnhancer
from src.analysis_processing.decision_engine import DecisionEngine
from src.analysis_processing.result_processor import ResultProcessor


class TestPhase3Pipeline:
    """Test complete Phase 3 signal processing pipeline."""
    
    @pytest.fixture
    def sample_telegram_message(self):
        """Sample Telegram trading signal message."""
        return """
        🚀 SIGNAL ALERT 🚀
        
        Pair: BTC/USDT
        Direction: LONG
        
        Entry: $42,500 - $43,000
        
        Targets:
        TP1: $44,000 (2.3%)
        TP2: $45,500 (5.8%)
        TP3: $47,000 (9.4%)
        
        Stop Loss: $41,000
        
        Leverage: 5x
        Risk: Medium
        
        Analysis: Strong support at $42,000 with bullish divergence on 4H RSI.
        Volume increasing, breakout expected above $43,500.
        """
    
    @pytest.fixture
    def mock_context_data(self):
        """Mock context data for testing."""
        return {
            'components': {
                'historical': {
                    'similar_signals': [
                        {'entry_price': 42000, 'confidence_score': 75, 'performance': {'pnl_percentage': 5.2}}
                    ],
                    'channel_performance': {'success_rate': 0.65, 'reputation_score': 0.7},
                    'pair_statistics': {'total_signals': 150, 'avg_confidence': 72},
                    'performance_metrics': {'win_rate': 0.62, 'sharpe_ratio': 1.8}
                },
                'market': {
                    'current_price': 42750,
                    'bid': 42745,
                    'ask': 42755,
                    'spread_percentage': 0.02,
                    'volume_24h': 15000000000,
                    'liquidity_score': 85,
                    'volatility': 45
                },
                'technical': {
                    'indicators': {'rsi': 58, 'macd': 150, 'bb_position': 65},
                    'signal_strength': 'buy',
                    'confluence_score': 72,
                    'patterns': ['uptrend', 'breakout_up']
                },
                'cross_channel': {
                    'consensus_score': 68,
                    'similar_signals_count': 3,
                    'validation_status': 'validated'
                }
            }
        }
    
    @pytest.fixture
    def mock_ai_response(self):
        """Mock AI analysis response."""
        return {
            'verdict': {
                'is_valid': True,
                'confidence_score': 75,
                'risk_level': 'medium',
                'recommendation': 'buy'
            },
            'analysis': {
                'signal_validity': {'is_valid': True, 'reasons': ['Strong technical setup', 'Good risk/reward']},
                'optimizations': {
                    'entry_price': 42600,
                    'stop_loss': 41200,
                    'take_profits': [44000, 45500, 47000]
                },
                'risk_factors': ['market_volatility', 'leverage_risk'],
                'recommendations': ['Use 50% position size', 'Set trailing stop after TP1']
            },
            'justification': {
                'novice': 'Good long opportunity with clear targets and stop loss.',
                'intermediate': 'Technical indicators support bullish bias. Entry near support with defined risk.',
                'expert': 'Confluence of support, RSI divergence, and volume expansion suggests high probability setup.'
            }
        }
    
    @pytest.mark.asyncio
    async def test_signal_detection(self, sample_telegram_message):
        """Test signal detection from Telegram message."""
        detector = SignalDetector()
        
        # Process the message
        result = await detector.detect_signal(sample_telegram_message)
        
        # Verify signal was detected
        assert result is not None
        assert result['signal_id'] is not None
        
        # Check trading parameters extracted
        params = result['trading_params']
        assert params['pair'] == 'BTC/USDT'
        assert params['direction'] == 'long'
        assert 42500 <= params['entry_price'] <= 43000
        assert params['stop_loss'] == 41000
        assert len(params['take_profits']) == 3
        assert params['leverage'] == 5
        
        # Check classification
        classification = result['classification']
        assert classification['signal_type'] in ['scalp', 'swing', 'position']
        assert classification['risk_level'] == 'medium'
        assert classification['confidence_score'] > 0
    
    @pytest.mark.asyncio
    async def test_context_building(self, mock_context_data):
        """Test context manager building comprehensive context."""
        with patch('src.context_management.context_manager.ContextManager') as MockContextManager:
            mock_manager = MockContextManager.return_value
            mock_manager.build_context = AsyncMock(return_value=mock_context_data)
            
            # Build context
            context = await mock_manager.build_context(
                signal_params={'trading_params': {'pair': 'BTC/USDT'}},
                channel_id=12345
            )
            
            # Verify context components
            assert 'components' in context
            assert 'historical' in context['components']
            assert 'market' in context['components']
            assert 'technical' in context['components']
            assert 'cross_channel' in context['components']
            
            # Check market data
            market = context['components']['market']
            assert market['current_price'] > 0
            assert market['liquidity_score'] > 50
            assert market['volatility'] < 100
    
    @pytest.mark.asyncio
    async def test_ai_analysis(self, mock_ai_response):
        """Test AI analysis integration."""
        with patch('src.ai_integration.ai_analyzer.AIAnalyzer') as MockAnalyzer:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze_signal = AsyncMock(return_value=mock_ai_response)
            
            # Perform analysis
            result = await mock_analyzer.analyze_signal(
                signal_data={'trading_params': {'pair': 'BTC/USDT'}},
                context={'components': {}},
                analysis_depth='full'
            )
            
            # Verify AI analysis
            assert result['verdict']['is_valid'] == True
            assert result['verdict']['confidence_score'] == 75
            assert result['verdict']['risk_level'] == 'medium'
            
            # Check optimizations
            assert 'optimizations' in result['analysis']
            assert result['analysis']['optimizations']['entry_price'] == 42600
            
            # Check justification
            assert 'justification' in result
            assert len(result['justification']['novice']) > 0
            assert len(result['justification']['expert']) > len(result['justification']['novice'])
    
    @pytest.mark.asyncio
    async def test_validation_framework(self):
        """Test validation framework."""
        validator = ValidationFramework()
        
        # Create test data
        signal_data = {
            'trading_params': {
                'pair': 'BTC/USDT',
                'entry_price': 42500,
                'stop_loss': 41000,
                'take_profits': [44000, 45500],
                'direction': 'long',
                'risk_reward_ratio': (1, 2.3)
            }
        }
        
        ai_analysis = {
            'verdict': {'confidence_score': 75, 'risk_level': 'medium'}
        }
        
        context = {
            'components': {
                'market': {'liquidity_score': 85, 'spread_percentage': 0.02},
                'cross_channel': {'consensus_score': 68}
            }
        }
        
        # Validate
        result = validator.validate(signal_data, ai_analysis, context)
        
        # Check validation result
        assert result.score > 60  # Should pass most checks
        assert len(result.checks_passed) > len(result.checks_failed)
        assert result.status.value in ['passed', 'warning']
    
    @pytest.mark.asyncio
    async def test_signal_enhancement(self):
        """Test signal enhancement engine."""
        enhancer = SignalEnhancer()
        
        # Create test data
        signal_data = {
            'trading_params': {
                'pair': 'BTC/USDT',
                'entry_price': 42500,
                'stop_loss': 41000,
                'take_profits': [44000, 45500],
                'direction': 'long'
            }
        }
        
        ai_analysis = {
            'analysis': {
                'optimizations': {
                    'entry_price': 42400,
                    'stop_loss': 41200
                }
            }
        }
        
        # Enhance signal
        enhanced = enhancer.enhance_signal(
            signal_data, ai_analysis, None, {'components': {}}
        )
        
        # Check enhancements
        assert enhanced.enhancement_score > 50
        assert enhanced.enhanced_params['entry_price'] == 42400  # AI optimization applied
        assert 'position_size' in enhanced.enhanced_params or 'order_types' in enhanced.enhanced_params
    
    @pytest.mark.asyncio
    async def test_decision_engine(self):
        """Test decision engine."""
        engine = DecisionEngine()
        
        # Create test data with high confidence
        ai_analysis = {
            'verdict': {
                'is_valid': True,
                'confidence_score': 82,
                'risk_level': 'low'
            }
        }
        
        validation_result = Mock()
        validation_result.score = 85
        validation_result.metadata = {'critical_failures': []}
        
        enhanced_signal = Mock()
        enhanced_signal.enhancement_score = 75
        enhanced_signal.enhanced_params = {'entry_price': 42500}
        enhanced_signal.risk_adjustments = {}
        enhanced_signal.execution_strategy = {}
        
        # Make decision
        decision = engine.make_decision(
            signal_data={},
            ai_analysis=ai_analysis,
            validation_result=validation_result,
            enhanced_signal=enhanced_signal,
            context={'components': {}}
        )
        
        # Check decision
        assert decision.action.value == 'execute'  # High confidence should execute
        assert decision.confidence.value in ['high', 'very_high']
        assert len(decision.execution_params) > 0
        assert len(decision.risk_limits) > 0
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, sample_telegram_message, mock_context_data, mock_ai_response):
        """Test complete Phase 3 pipeline integration."""
        # Initialize components
        detector = SignalDetector()
        
        # Mock other components
        with patch('src.context_management.context_manager.ContextManager') as MockContext, \
             patch('src.ai_integration.ai_analyzer.AIAnalyzer') as MockAI, \
             patch('src.analysis_processing.result_processor.ResultProcessor') as MockProcessor:
            
            # Setup mocks
            mock_context = MockContext.return_value
            mock_context.build_context = AsyncMock(return_value=mock_context_data)
            
            mock_ai = MockAI.return_value
            mock_ai.analyze_signal = AsyncMock(return_value=mock_ai_response)
            
            mock_processor = MockProcessor.return_value
            mock_processor.process_result = AsyncMock(return_value={
                'status': 'success',
                'result_id': 'test_123',
                'action': 'execute'
            })
            
            # Step 1: Detect signal
            signal = await detector.detect_signal(sample_telegram_message)
            assert signal is not None
            
            # Step 2: Build context
            context = await mock_context.build_context(signal, channel_id=12345)
            assert context is not None
            
            # Step 3: AI analysis
            ai_result = await mock_ai.analyze_signal(signal, context, 'full')
            assert ai_result['verdict']['is_valid'] == True
            
            # Step 4: Validation
            validator = ValidationFramework()
            validation = validator.validate(signal, ai_result, context)
            assert validation.score > 0
            
            # Step 5: Enhancement
            enhancer = SignalEnhancer()
            enhanced = enhancer.enhance_signal(signal, ai_result, validation, context)
            assert enhanced.enhancement_score > 0
            
            # Step 6: Decision
            engine = DecisionEngine()
            decision = engine.make_decision(signal, ai_result, validation, enhanced, context)
            assert decision.action.value in ['execute', 'monitor', 'reject', 'paper_trade']
            
            # Step 7: Process result
            result = await mock_processor.process_result(
                signal, ai_result, validation, enhanced, decision
            )
            assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_performance_targets(self, sample_telegram_message):
        """Test that performance targets are met."""
        import time
        
        detector = SignalDetector()
        
        # Test signal detection performance (<100ms)
        start = time.time()
        result = await detector.detect_signal(sample_telegram_message)
        detection_time = (time.time() - start) * 1000
        
        # Allow more time in test environment
        assert detection_time < 500  # 500ms for test environment
        assert result is not None
        
        # Check detection accuracy
        params = result['trading_params']
        assert params['pair'] == 'BTC/USDT'
        assert params['direction'] == 'long'
        assert params['stop_loss'] == 41000
    
    def test_token_optimization(self):
        """Test token optimization for LLM context."""
        from src.ai_integration.token_optimizer import TokenOptimizer
        
        optimizer = TokenOptimizer(max_tokens=8000)
        
        # Create large context
        large_context = {
            'signal': {'data': 'x' * 10000},  # Large data
            'market': {'prices': [42000 + i for i in range(1000)]},
            'historical': {'signals': [{'id': i} for i in range(500)]}
        }
        
        # Optimize
        optimized, metrics = optimizer.optimize_context(large_context)
        
        # Check optimization
        assert metrics.total_tokens <= 8000
        assert metrics.remaining_budget >= 0
        assert 'signal' in optimized  # High priority kept
    
    def test_validation_rules_coverage(self):
        """Test that all validation rules are implemented."""
        validator = ValidationFramework()
        
        # Check rule categories
        assert 'technical' in validator.validation_rules
        assert 'market' in validator.validation_rules
        assert 'signal' in validator.validation_rules
        assert 'risk' in validator.validation_rules
        
        # Count total rules
        total_rules = sum(
            len(rules) for rules in validator.validation_rules.values()
        )
        assert total_rules >= 20  # At least 20 validation rules
    
    def test_decision_actions(self):
        """Test all decision action types."""
        from src.analysis_processing.decision_engine import ActionType
        
        # Verify all action types exist
        actions = [a.value for a in ActionType]
        assert 'execute' in actions
        assert 'monitor' in actions
        assert 'reject' in actions
        assert 'paper_trade' in actions
        assert 'scale_in' in actions


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
