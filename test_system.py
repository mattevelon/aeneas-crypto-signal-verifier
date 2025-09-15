#!/usr/bin/env python3
"""
System test script to verify all components are working correctly.
"""

import asyncio
import sys
from typing import Dict, Any
import httpx

def print_status(component: str, status: bool, message: str = ""):
    """Print component status with color."""
    if status:
        print(f"✅ {component}: OK")
    else:
        print(f"❌ {component}: FAILED")
    if message:
        print(f"   {message}")

async def test_api_health():
    """Test API health endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/v1/health")
            if response.status_code == 200:
                data = response.json()
                return True, data
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

async def test_database():
    """Test database connection."""
    try:
        from src.core.database import engine
        from sqlalchemy import text
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return True, "Database connected"
    except Exception as e:
        return False, str(e)

async def test_redis():
    """Test Redis connection."""
    try:
        from src.core.redis_client import redis_client
        
        if redis_client:
            await redis_client.ping()
            return True, "Redis connected"
        return False, "Redis client not initialized"
    except Exception as e:
        return False, str(e)

async def test_config_validation():
    """Test configuration validation."""
    try:
        from src.config.settings import settings
        
        validations = []
        
        # Check critical settings
        if settings.database_url:
            validations.append("Database URL configured")
        
        # Check optional services
        if settings.has_telegram_credentials:
            validations.append("Telegram credentials available")
        else:
            validations.append("Telegram credentials missing (non-critical)")
            
        if settings.has_llm_credentials:
            validations.append("LLM credentials available")
        else:
            validations.append("LLM credentials missing (non-critical)")
            
        if settings.has_exchange_credentials:
            validations.append("Exchange credentials available")
        else:
            validations.append("Exchange credentials missing (non-critical)")
        
        return True, "; ".join(validations)
    except Exception as e:
        return False, str(e)

async def test_signal_detection():
    """Test signal detection logic."""
    try:
        from src.core.signal_detector import SignalDetector
        
        detector = SignalDetector()
        
        # Test with sample signal text
        test_text = """
        BTC/USDT
        LONG Entry: $65,000
        Stop Loss: $63,000
        Take Profit 1: $67,000
        Take Profit 2: $70,000
        Leverage: 5x
        """
        
        is_signal = await detector.detect(test_text)
        
        if is_signal:
            signal_data = await detector.extract(test_text)
            if signal_data:
                return True, f"Signal extracted: {signal_data.get('pair')} {signal_data.get('direction')}"
            return False, "Signal detected but extraction failed"
        return False, "Signal not detected in test text"
    except Exception as e:
        return False, str(e)

async def test_market_data():
    """Test market data integration."""
    try:
        from src.core.market_data import market_data
        
        # Test with BTC/USDT
        price = await market_data.get_price("BTC/USDT")
        
        if price:
            return True, f"BTC/USDT price: ${price:,.2f}"
        return False, "Unable to fetch market price"
    except Exception as e:
        return False, str(e)

async def test_docker_services():
    """Test Docker services availability."""
    services = {
        "PostgreSQL": ("localhost", 5432),
        "Redis": ("localhost", 6379),
        "Qdrant": ("localhost", 6333),
        "Kafka": ("localhost", 9092),
    }
    
    results = []
    for service, (host, port) in services.items():
        try:
            async with httpx.AsyncClient() as client:
                # Try to connect to the port
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    results.append(f"{service}: Running")
                else:
                    results.append(f"{service}: Not accessible")
        except Exception as e:
            results.append(f"{service}: Error - {e}")
    
    return True, "; ".join(results)

async def main():
    """Run all system tests."""
    print(f"\n{'='*60}")
    print(f"AENEAS System Test Suite")
    print(f"{'='*60}\n")
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("API Health", test_api_health),
        ("Database Connection", test_database),
        ("Redis Connection", test_redis),
        ("Signal Detection", test_signal_detection),
        ("Market Data", test_market_data),
        ("Docker Services", test_docker_services),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            success, message = await test_func()
            print_status(test_name, success, message)
            if not success:
                all_passed = False
        except Exception as e:
            print_status(test_name, False, str(e))
            all_passed = False
        print()  # Empty line between tests
    
    print(f"\n{'='*60}")
    if all_passed:
        print(f"All tests passed! System is operational.")
    else:
        print(f"Some tests failed. Check the errors above.")
        print(f"The system can still run with reduced functionality.")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
