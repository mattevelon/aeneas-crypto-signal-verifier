"""DeepSeek V3.1 Configuration for AENEAS Project"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DeepSeekModel(Enum):
    """Available DeepSeek models via OpenRouter."""
    V3_1 = "deepseek/deepseek-v3.1"
    CHAT = "deepseek/deepseek-chat"
    CODER = "deepseek/deepseek-coder"

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek V3.1 model."""
    
    # Model selection
    primary_model: str = DeepSeekModel.V3_1.value
    fallback_model: str = DeepSeekModel.CHAT.value
    
    # Model parameters for crypto signal analysis
    temperature: float = 0.3  # Lower for more deterministic analysis
    max_tokens: int = 2000  # Sufficient for detailed analysis
    top_p: float = 0.9  # Balanced diversity
    frequency_penalty: float = 0.2  # Reduce repetition
    presence_penalty: float = 0.1  # Encourage covering all aspects
    
    # Response format
    response_format: str = "json"  # Structured output for parsing
    
    # Context window management
    max_context_tokens: int = 8000  # DeepSeek's context limit
    reserved_tokens: int = 2000  # Reserve for response
    
    # Performance settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Cost optimization
    cache_responses: bool = True
    cache_ttl: int = 3600  # 1 hour
    batch_requests: bool = True
    batch_size: int = 5
    
    # Signal analysis specific settings
    confidence_threshold: float = 70.0  # Minimum confidence for EXECUTE
    risk_score_max: float = 50.0  # Maximum acceptable risk score
    position_size_max: float = 0.1  # 10% max position size
    
    def get_api_params(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Get API parameters for DeepSeek request."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return {
            "model": self.primary_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": {"type": "json_object"} if self.response_format == "json" else None,
            "stream": False
        }
    
    def get_analysis_prompt_template(self) -> str:
        """Get the analysis prompt template for signal verification."""
        return """
Analyze this cryptocurrency trading signal and provide a detailed assessment:

Signal Details:
{signal_data}

Market Context:
{market_data}

Technical Indicators:
{technical_data}

Historical Performance:
{historical_data}

Provide analysis in the following JSON structure:
{{
  "signal_validity": {{
    "score": <0-100>,
    "confidence": "<HIGH|MEDIUM|LOW>",
    "recommendation": "<EXECUTE|MONITOR|REJECT>"
  }},
  "risk_assessment": {{
    "risk_score": <0-100>,
    "position_size": "<percentage>",
    "max_loss": "<USD amount>",
    "risk_reward_ratio": "<ratio>"
  }},
  "market_context": {{
    "trend": "<BULLISH|BEARISH|NEUTRAL>",
    "volatility": "<HIGH|MEDIUM|LOW>",
    "liquidity": "<SUFFICIENT|LIMITED|INSUFFICIENT>"
  }},
  "warnings": [],
  "justification": {{
    "primary_factors": [],
    "risk_factors": []
  }}
}}
"""

# Global instance
deepseek_config = DeepSeekConfig()
