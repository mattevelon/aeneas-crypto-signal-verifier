#!/bin/bash
# Setup script for DeepSeek V3.1 integration

echo "========================================="
echo "DeepSeek V3.1 Setup for AENEAS"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Check for OPENROUTER_API_KEY
if grep -q "^OPENROUTER_API_KEY=..*" .env; then
    echo "✅ OPENROUTER_API_KEY is set in .env"
else
    echo ""
    echo "⚠️  OPENROUTER_API_KEY is not set!"
    echo ""
    echo "To complete the setup:"
    echo "1. Get your API key from https://openrouter.ai/keys"
    echo "2. Add it to your .env file:"
    echo "   OPENROUTER_API_KEY=your_key_here"
    echo ""
fi

echo ""
echo "Testing prompt system..."
python3 -c "
from src.ai_integration.prompt_engine import PromptEngine
try:
    engine = PromptEngine()
    if hasattr(engine, 'deepseek_system_prompt'):
        print('✅ Prompt system initialized successfully')
    else:
        print('❌ Prompt system error')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "========================================="
echo "Setup check complete!"
echo "========================================="
