#!/usr/bin/env python3
"""Test script for AENEAS project features."""

import asyncio
import json
from datetime import datetime

# Test 1: Signal Detection
async def test_signal_detection():
    print("\n=== TEST 1: Signal Detection System ===")
    try:
        from src.signal_detection.signal_detector import SignalDetector
        
        # Sample trading signal
        test_message = """
        🚀 SIGNAL ALERT 🚀
        
        Pair: BTC/USDT
        Direction: LONG
        
        Entry: $42,500 - $43,000
        
        Targets:
        TP1: $44,000 (2.3%)
        TP2: $45,500 (5.8%)
        TP3: $47,000 (9.4%)
        
        Stop Loss: $41,000
        
        Leverage: 5x
        Risk: Medium
        """
        
        detector = SignalDetector()
        result = await detector.detect_signal(test_message)
        
        if result:
            print("✅ Signal detected successfully!")
            print(f"  - Pair: {result['trading_params']['pair']}")
            print(f"  - Direction: {result['trading_params']['direction']}")
            print(f"  - Entry: ${result['trading_params']['entry_price']}")
            print(f"  - Stop Loss: ${result['trading_params']['stop_loss']}")
            print(f"  - Targets: {result['trading_params']['take_profits']}")
            print(f"  - Confidence: {result['classification']['confidence_score']:.1f}%")
        else:
            print("❌ No signal detected")
            
    except Exception as e:
        print(f"❌ Signal detection failed: {e}")
        return False
    return True

# Test 2: Database Connection
async def test_database():
    print("\n=== TEST 2: Database Connection ===")
    try:
        from src.core.database import get_async_session
        
        async with get_async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            print("✅ PostgreSQL connection successful")
            
            # Check tables
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5
            """))
            tables = result.fetchall()
            if tables:
                print(f"  - Found {len(tables)} tables:")
                for table in tables:
                    print(f"    • {table[0]}")
            else:
                print("  - No tables found (needs migration)")
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    return True

# Test 3: Redis Cache
async def test_redis():
    print("\n=== TEST 3: Redis Cache ===")
    try:
        from src.core.redis_client import get_redis, RedisCache
        
        cache = RedisCache("test")
        
        # Test set/get
        test_key = f"test_{datetime.now().timestamp()}"
        test_value = {"status": "working", "timestamp": datetime.now().isoformat()}
        
        await cache.set(test_key, test_value, ttl=60)
        retrieved = await cache.get(test_key)
        
        if retrieved and retrieved['status'] == 'working':
            print("✅ Redis cache working")
            print(f"  - Set/Get successful")
            print(f"  - Value: {retrieved}")
            
            # Cleanup
            await cache.delete(test_key)
            print(f"  - Cleanup successful")
        else:
            print("❌ Redis cache not working properly")
            
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False
    return True

# Test 4: Context Building
async def test_context_building():
    print("\n=== TEST 4: Context Manager ===")
    try:
        from src.context_management.context_manager import ContextManager
        
        manager = ContextManager()
        
        # Test with minimal signal data
        signal_data = {
            'trading_params': {
                'pair': 'BTC/USDT',
                'direction': 'long',
                'entry_price': 42500
            }
        }
        
        print("⏳ Building context (this may take a moment)...")
        context = await manager.build_context(signal_data, channel_id=12345)
        
        if context:
            print("✅ Context built successfully")
            print(f"  - Components: {list(context.get('components', {}).keys())}")
            if 'summary' in context:
                print(f"  - Summary length: {len(context['summary'])} chars")
        else:
            print("❌ Context building failed")
            
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        return False
    return True

# Test 5: Configuration
def test_configuration():
    print("\n=== TEST 5: Configuration ===")
    try:
        from src.config.settings import settings
        
        print("✅ Settings loaded successfully")
        print(f"  - Environment: {settings.app_env}")
        print(f"  - Port: {settings.app_port}")
        print(f"  - Redis URL: {settings.redis_url}")
        print(f"  - Telegram configured: {settings.has_telegram_credentials}")
        print(f"  - LLM configured: {settings.has_llm_credentials}")
        print(f"  - Exchange configured: {settings.has_exchange_credentials}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    return True

# Test 6: Phase 3 Modules
def test_phase3_imports():
    print("\n=== TEST 6: Phase 3 Module Imports ===")
    modules_to_test = [
        ("Signal Detection", "src.signal_detection.signal_detector", "SignalDetector"),
        ("Context Manager", "src.context_management.context_manager", "ContextManager"),
        ("AI Analyzer", "src.ai_integration.ai_analyzer", "AIAnalyzer"),
        ("Validation Framework", "src.analysis_processing.validation_framework", "ValidationFramework"),
        ("Decision Engine", "src.analysis_processing.decision_engine", "DecisionEngine"),
        ("Result Processor", "src.analysis_processing.result_processor", "ResultProcessor")
    ]
    
    all_successful = True
    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {name}: {class_name} imported successfully")
        except Exception as e:
            print(f"❌ {name}: Import failed - {e}")
            all_successful = False
    
    return all_successful

# Main test runner
async def main():
    print("=" * 60)
    print(" AENEAS PROJECT FEATURE TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_configuration()))
    results.append(("Phase 3 Imports", test_phase3_imports()))
    results.append(("Signal Detection", await test_signal_detection()))
    results.append(("Database", await test_database()))
    results.append(("Redis Cache", await test_redis()))
    # Context building might fail due to missing API keys
    # results.append(("Context Building", await test_context_building()))
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is operational.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
