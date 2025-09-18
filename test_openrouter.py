#!/usr/bin/env python3
"""Direct test of OpenRouter API with DeepSeek."""

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
print(f"API Key found: {api_key[:20]}..." if api_key else "No API key")

if not api_key:
    print("Please set OPENROUTER_API_KEY in .env")
    exit(1)

# Test direct API call
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://aeneas-crypto.com",  # Optional
    "X-Title": "AENEAS Test"  # Optional
}

data = {
    "model": "deepseek/deepseek-chat-v3.1:free",  # Try the free v3.1 model
    "messages": [
        {
            "role": "user",
            "content": "Say 'Hello from DeepSeek' in JSON format"
        }
    ],
    "max_tokens": 100,
    "temperature": 0.3
}

print("\nTesting OpenRouter API...")
print(f"Model: {data['model']}")

try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']
            print(f"Response: {content}")
        print(f"Model used: {result.get('model', 'unknown')}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")

# List available models
print("\n" + "="*50)
print("Fetching available models...")

models_url = "https://openrouter.ai/api/v1/models"
try:
    response = requests.get(models_url, headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code == 200:
        models = response.json()
        deepseek_models = [m['id'] for m in models.get('data', []) if 'deepseek' in m['id'].lower()]
        if deepseek_models:
            print("Available DeepSeek models:")
            for model in deepseek_models[:5]:  # Show first 5
                print(f"  - {model}")
        else:
            print("No DeepSeek models found")
    else:
        print(f"Could not fetch models: {response.status_code}")
except Exception as e:
    print(f"Error fetching models: {e}")
