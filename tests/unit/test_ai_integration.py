"""Unit tests for AI integration modules."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.ai_integration.llm_client import LLMClient
from src.ai_integration.ai_analyzer import AIAnalyzer
from src.ai_integration.token_optimizer import TokenOptimizer


class TestLLMClient:
    """Test LLM client functionality."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client instance."""
        with patch('src.ai_integration.llm_client.get_settings') as mock_settings:
            mock_settings.return_value.llm_api_key = "test_key"
            mock_settings.return_value.llm_model = "gpt-4"
            mock_settings.return_value.llm_temperature = 0.3
            return LLMClient()
    
    @pytest.mark.asyncio
    async def test_analyze_signal(self, llm_client):
        """Test signal analysis."""
        llm_client.client = Mock()
        llm_client.client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content='{"confidence": 85, "recommendation": "BUY"}'))]
            )
        )
        
        result = await llm_client.analyze_signal({
            "pair": "BTC/USDT",
            "entry_price": 50000
        })
        
        assert result is not None
        assert "confidence" in result
        assert "recommendation" in result
    
    def test_request_batching(self, llm_client):
        """Test request batching logic."""
        llm_client.batch_requests = []
        
        # Add requests
        for i in range(3):
            llm_client.batch_requests.append({"id": i})
        
        assert len(llm_client.batch_requests) == 3
        
        # Clear batch
        llm_client.batch_requests.clear()
        assert len(llm_client.batch_requests) == 0


class TestAIAnalyzer:
    """Test AI analyzer functionality."""
    
    @pytest.fixture
    def ai_analyzer(self):
        """Create AI analyzer instance."""
        with patch('src.ai_integration.ai_analyzer.LLMClient'):
            return AIAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_with_context(self, ai_analyzer):
        """Test analysis with context."""
        ai_analyzer.llm_client = Mock()
        ai_analyzer.llm_client.analyze_signal = AsyncMock(
            return_value={"confidence": 90, "action": "EXECUTE"}
        )
        
        result = await ai_analyzer.analyze_with_context(
            signal={"pair": "ETH/USDT"},
            context={"market_trend": "bullish"}
        )
        
        assert result is not None
        assert "confidence" in result
        assert "action" in result
    
    def test_confidence_threshold(self, ai_analyzer):
        """Test confidence threshold filtering."""
        signals = [
            {"confidence": 95},
            {"confidence": 50},
            {"confidence": 80},
            {"confidence": 30}
        ]
        
        # Filter with 70% threshold
        filtered = [s for s in signals if s["confidence"] >= 70]
        assert len(filtered) == 2
        assert all(s["confidence"] >= 70 for s in filtered)


class TestTokenOptimizer:
    """Test token optimization."""
    
    @pytest.fixture
    def token_optimizer(self):
        """Create token optimizer instance."""
        return TokenOptimizer()
    
    def test_count_tokens(self, token_optimizer):
        """Test token counting."""
        text = "This is a test message for token counting."
        
        # Mock tiktoken encoding
        with patch('src.ai_integration.token_optimizer.tiktoken.get_encoding') as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
            mock_encoding.return_value = mock_encoder
            
            count = token_optimizer.count_tokens(text)
            assert count == 8
    
    def test_prune_context(self, token_optimizer):
        """Test context pruning."""
        context = {
            "essential": "keep this",
            "optional": "can be removed",
            "large_data": "x" * 10000
        }
        
        # Mock token counting
        with patch.object(token_optimizer, 'count_tokens') as mock_count:
            mock_count.side_effect = [5000, 3000, 1000]
            
            pruned = token_optimizer.prune_context(context, max_tokens=4000)
            
            # Should keep essential and optional, remove large_data
            assert "essential" in pruned
            assert "large_data" not in pruned or len(pruned["large_data"]) < 10000
    
    def test_token_budget_management(self, token_optimizer):
        """Test token budget management."""
        token_optimizer.token_budget = 8000
        token_optimizer.tokens_used = 0
        
        # Use some tokens
        token_optimizer.use_tokens(3000)
        assert token_optimizer.tokens_used == 3000
        assert token_optimizer.remaining_tokens() == 5000
        
        # Try to exceed budget
        with pytest.raises(ValueError):
            token_optimizer.use_tokens(6000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
