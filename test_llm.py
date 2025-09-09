"""
Test script for LLM client with OpenRouter/DeepSeek.
"""

import asyncio
import sys
sys.path.append('/Users/matt/Desktop/it/projects/AENEAS/aeneas_architecture of AI work')

from src.core.llm_client import LLMClient

def test_llm_connection():
    """Test the LLM client connection and basic functionality."""
    print("Testing LLM Client with OpenRouter/DeepSeek...")
    
    client = LLMClient()
    
    # Test signal analysis
    test_signal = """
    BTC/USDT
    Entry: $42,500
    Target 1: $43,200
    Target 2: $44,000
    Stop Loss: $41,800
    """
    
    print("\nAnalyzing test signal...")
    result = client.analyze_signal(test_signal)
    
    if result["success"]:
        print("✅ LLM Connection successful!")
        print(f"Model used: {result['model']}")
        print(f"Tokens used: {result['tokens_used']}")
        print("\nAnalysis:")
        print(result["analysis"][:500] + "..." if len(result["analysis"]) > 500 else result["analysis"])
    else:
        print(f"❌ LLM Connection failed: {result['error']}")
        
    return result["success"]

if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)
