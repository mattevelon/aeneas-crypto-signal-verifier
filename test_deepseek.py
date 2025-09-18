#!/usr/bin/env python3
"""Test script for DeepSeek V3.1 integration via OpenRouter."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def test_deepseek_integration():
    """Test DeepSeek V3.1 model integration."""
    print("="*60)
    print("DeepSeek V3.1 Integration Test")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*60)
    
    # Check for OpenRouter API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("   Please set it in your .env file")
        return False
    else:
        print("✅ OpenRouter API key found")
    
    # Test prompt for signal analysis
    test_signal = {
        "pair": "BTC/USDT",
        "entry_price": 47500,
        "stop_loss": 46800,
        "take_profit_1": 48200,
        "take_profit_2": 49000,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create test prompt
    prompt = f"""
    Analyze this cryptocurrency trading signal:
    
    Signal: {json.dumps(test_signal, indent=2)}
    
    Current Market Price: 47450
    24h Volume: $28.5B
    RSI (14): 58
    MACD: Bullish crossover
    
    Provide analysis with:
    1. Confidence score (0-100)
    2. Recommendation (EXECUTE/MONITOR/REJECT)
    3. Risk assessment
    4. Key factors
    
    Format response as JSON.
    """
    
    print("\n📤 Sending test signal to DeepSeek V3.1...")
    print(f"   Signal: {test_signal['pair']} @ ${test_signal['entry_price']}")
    
    try:
        # Import after path setup
        from src.ai_integration.llm_client import LLMClient, LLMProvider
        
        # Create client
        client = LLMClient()
        
        # Prepare prompt
        llm_prompt = {
            "system": """You are AENEAS-AI, an advanced cryptocurrency trading signal analysis system.
Analyze signals with focus on risk assessment using Kelly Criterion and market manipulation detection.""",
            "user": prompt
        }
        
        # Make request using DeepSeek
        async with client:
            response = await client.analyze_signal(
                prompt=llm_prompt,
                provider=LLMProvider.OPENROUTER,
                model='deepseek/deepseek-chat-v3.1:free'  # DeepSeek V3.1 (free version)
            )
        
        print("\n✅ Response received from DeepSeek!")
        print("\n📊 Analysis Result:")
        print("-" * 40)
        
        # Try to parse as JSON
        try:
            result = json.loads(response.content)
            print(json.dumps(result, indent=2))
        except:
            # If not JSON, print raw
            print(response.content)
        
        print("-" * 40)
        print(f"\n📈 Model: {response.model}")
        print(f"⏱️  Latency: {response.latency_ms:.0f}ms")
        print(f"🔤 Tokens: {response.usage.get('total_tokens', 'N/A')}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure the application is properly installed")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your API configuration and network connection")
        return False

async def test_prompt_loading():
    """Test loading of DeepSeek system prompt."""
    print("\n" + "="*60)
    print("Testing DeepSeek System Prompt Loading")
    print("="*60)
    
    try:
        from src.ai_integration.prompt_engine import PromptEngine
        
        engine = PromptEngine()
        
        # Check if DeepSeek prompt is loaded
        if hasattr(engine, 'deepseek_system_prompt') and engine.deepseek_system_prompt:
            prompt_preview = engine.deepseek_system_prompt[:200] if len(engine.deepseek_system_prompt) > 200 else engine.deepseek_system_prompt
            print(f"✅ DeepSeek prompt loaded")
            print(f"   Preview: {prompt_preview}...")
        else:
            print("⚠️  DeepSeek prompt attribute not found or empty in PromptEngine")
            
    except Exception as e:
        print(f"❌ Error loading prompt: {e}")

async def main():
    """Run all tests."""
    print("\n🚀 Starting DeepSeek V3.1 Integration Tests\n")
    
    # Test prompt loading
    await test_prompt_loading()
    
    # Test API integration
    success = await test_deepseek_integration()
    
    print("\n" + "="*60)
    if success:
        print("✅ DeepSeek V3.1 integration test PASSED!")
        print("\nNext steps:")
        print("1. The system is configured to use DeepSeek V3.1")
        print("2. The model will be used automatically for signal analysis")
        print("3. Monitor the logs for performance metrics")
    else:
        print("❌ DeepSeek integration test FAILED")
        print("\nTroubleshooting:")
        print("1. Check OPENROUTER_API_KEY in .env")
        print("2. Verify internet connection")
        print("3. Check OpenRouter account status")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
