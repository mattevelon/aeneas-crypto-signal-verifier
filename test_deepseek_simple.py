#!/usr/bin/env python3
"""Simple test to verify DeepSeek setup."""

import os
from pathlib import Path

print("DeepSeek V3.1 Setup Check")
print("=" * 40)

# 1. Check prompt file
prompt_file = Path("src/ai_integration/prompts/deepseek_v3_system.md")
if prompt_file.exists():
    print("✅ DeepSeek prompt file exists")
    print(f"   Size: {prompt_file.stat().st_size} bytes")
else:
    print("❌ DeepSeek prompt file missing")

# 2. Check prompt loading
try:
    from src.ai_integration.prompt_engine import PromptEngine
    engine = PromptEngine()
    if hasattr(engine, 'deepseek_system_prompt') and engine.deepseek_system_prompt:
        print("✅ Prompt engine initialized")
        print(f"   Prompt loaded: {len(engine.deepseek_system_prompt)} characters")
    else:
        print("❌ Prompt not loaded")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Check API key
if os.getenv('OPENROUTER_API_KEY'):
    key = os.getenv('OPENROUTER_API_KEY')
    print(f"✅ OPENROUTER_API_KEY is set ({len(key)} characters)")
else:
    print("❌ OPENROUTER_API_KEY not found")
    print("\n   To fix: Add to your .env file:")
    print("   OPENROUTER_API_KEY=your_key_here")

# 4. Check configuration
try:
    from src.ai_integration.deepseek_config import deepseek_config
    print(f"✅ DeepSeek config loaded")
    print(f"   Model: {deepseek_config.primary_model}")
    print(f"   Temperature: {deepseek_config.temperature}")
except Exception as e:
    print(f"❌ Config error: {e}")

print("\n" + "=" * 40)
print("Setup check complete!")
