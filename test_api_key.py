#!/usr/bin/env python3
"""Test OpenRouter API key validity."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    print("❌ No API key found")
    exit(1)

print(f"Testing API key: {api_key[:20]}...")

# Test 1: Get account info
url = "https://openrouter.ai/api/v1/auth/key"
headers = {"Authorization": f"Bearer {api_key}"}

print("\n1. Checking API key validity...")
response = requests.get(url, headers=headers)
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"   ✅ API key is valid")
    print(f"   Credits: ${data.get('data', {}).get('usage', 0):.4f} used")
    print(f"   Limit: ${data.get('data', {}).get('limit', 0):.2f}")
else:
    print(f"   ❌ Invalid API key: {response.text}")

# Test 2: Try a simple completion with a free model
print("\n2. Testing completion with free model...")
url = "https://openrouter.ai/api/v1/chat/completions"
data = {
    "model": "meta-llama/llama-3.2-3b-instruct:free",  # Try a different free model
    "messages": [{"role": "user", "content": "Say 'test'"}],
    "max_tokens": 10
}

response = requests.post(url, headers={
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}, json=data)

print(f"   Status: {response.status_code}")
if response.status_code == 200:
    print("   ✅ Completion works")
else:
    print(f"   ❌ Error: {response.text}")

print("\n" + "="*50)
print("Diagnostics:")
print(f"• API key format: {'✅ Valid' if api_key.startswith('sk-or-') else '❌ Invalid'}")
print(f"• API key length: {len(api_key)} chars")
print("• Instructions:")
print("  1. Go to https://openrouter.ai/settings/keys")
print("  2. Create a new API key if needed")
print("  3. Make sure your account is active")
print("  4. Add credits if using paid models")
