#!/bin/bash
# Quick OpenRouter API key verification

echo "Testing OpenRouter API key..."

# Read API key from .env
API_KEY=$(grep "OPENROUTER_API_KEY" .env | cut -d'=' -f2)

if [ -z "$API_KEY" ]; then
    echo "❌ No API key found in .env"
    exit 1
fi

echo "Key: ${API_KEY:0:20}..."

# Test the key
response=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $API_KEY" \
    https://openrouter.ai/api/v1/auth/key)

if [ "$response" = "200" ]; then
    echo "✅ API key is valid!"
    
    # Test DeepSeek model
    echo "Testing DeepSeek V3.1..."
    curl -s https://openrouter.ai/api/v1/chat/completions \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek/deepseek-chat-v3.1:free",
            "messages": [{"role": "user", "content": "Say test"}],
            "max_tokens": 10
        }' | python3 -m json.tool
else
    echo "❌ API key invalid (HTTP $response)"
    echo ""
    echo "To fix:"
    echo "1. Go to https://openrouter.ai/settings/keys"
    echo "2. Create a new API key"
    echo "3. Update OPENROUTER_API_KEY in .env"
fi
