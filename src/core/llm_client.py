"""
LLM client for OpenRouter/DeepSeek integration.
"""

from typing import List, Dict, Optional, Any
from openai import OpenAI
from src.config.settings import settings
import structlog

logger = structlog.get_logger()

# Check if LLM credentials are available
LLM_ENABLED = settings.has_llm_credentials


class LLMClient:
    """Client for interacting with LLM via OpenRouter."""
    
    def __init__(self):
        """Initialize the LLM client."""
        self.client = None
        self.model = None
        self.enabled = LLM_ENABLED
        
        if not self.enabled:
            logger.warning("LLM client disabled: missing API key")
            return
            
        # OpenRouter uses OpenAI-compatible API
        base_url = "https://openrouter.ai/api/v1"
        
        try:
            # For OpenRouter with DeepSeek
            if "openrouter" in settings.llm_provider.lower():
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=settings.llm_api_key,
                )
                # Use the model from settings or default to DeepSeek
                self.model = settings.llm_model if settings.llm_model else "deepseek/deepseek-chat-v3.1:free"
            else:
                # Standard OpenAI
                self.client = OpenAI(api_key=settings.llm_api_key)
                self.model = settings.llm_model
            
            self.temperature = settings.llm_temperature
            self.max_tokens = settings.llm_max_tokens
            logger.info(f"LLM client initialized with provider: {settings.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.enabled = False
        
    def analyze_signal(self, message_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a trading signal using LLM.
        
        Args:
            message_text: The signal message to analyze
            context: Additional context (market data, historical performance, etc.)
            
        Returns:
            Analysis results including validity, risk assessment, and recommendations
        """
        if not self.enabled or not self.client:
            logger.warning("LLM analysis skipped: client not available")
            return {
                "success": False,
                "error": "LLM service not configured",
                "analysis": None
            }
            
        try:
            system_prompt = """You are an expert cryptocurrency trading signal analyzer. 
            Analyze the provided trading signal and return a structured assessment including:
            1. Signal validity (is this a legitimate trading signal?)
            2. Risk level (high/medium/low)
            3. Recommended position size
            4. Key technical levels
            5. Potential issues or red flags
            """
            
            user_prompt = f"Analyze this trading signal:\n{message_text}"
            
            if context:
                user_prompt += f"\n\nAdditional context:\n{context}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add OpenRouter-specific headers if using OpenRouter
            extra_kwargs = {}
            if "openrouter" in settings.llm_provider.lower():
                extra_kwargs = {
                    "extra_headers": {
                        "HTTP-Referer": "https://crypto-signals-verifier.com",
                        "X-Title": "Crypto Signals Verifier"
                    }
                }
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra_kwargs
            )
            
            response = completion.choices[0].message.content
            
            # Parse and structure the response
            return {
                "success": True,
                "analysis": response,
                "model": self.model,
                "tokens_used": completion.usage.total_tokens if hasattr(completion, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def generate_tiered_explanation(
        self, 
        signal_analysis: Dict[str, Any], 
        level: str = "intermediate"
    ) -> str:
        """
        Generate explanation for different expertise levels.
        
        Args:
            signal_analysis: The analysis results
            level: One of "novice", "intermediate", "expert"
            
        Returns:
            Formatted explanation appropriate for the expertise level
        """
        if not self.enabled or not self.client:
            return "LLM service not configured - unable to generate explanation."
            
        try:
            level_prompts = {
                "novice": "Explain in simple terms for someone new to trading",
                "intermediate": "Provide a balanced technical and fundamental explanation",
                "expert": "Give detailed technical analysis with advanced metrics"
            }
            
            prompt = f"{level_prompts.get(level, level_prompts['intermediate'])}:\n{signal_analysis}"
            
            messages = [
                {"role": "system", "content": "You are a trading educator."},
                {"role": "user", "content": prompt}
            ]
            
            extra_kwargs = {}
            if "openrouter" in settings.llm_provider.lower():
                extra_kwargs = {
                    "extra_headers": {
                        "HTTP-Referer": "https://crypto-signals-verifier.com",
                        "X-Title": "Crypto Signals Verifier"
                    }
                }
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
                **extra_kwargs
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            return "Unable to generate explanation at this time."
