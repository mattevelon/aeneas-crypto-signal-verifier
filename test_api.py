#!/usr/bin/env python3
"""Quick API test script for AENEAS."""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("1. Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    if response.status_code == 200:
        print(f"   ✅ Health check passed: {response.json()}")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
    return response.status_code == 200

def test_api_docs():
    """Test API documentation."""
    print("\n2. Testing API Documentation...")
    response = requests.get(f"{BASE_URL}/api/docs")
    if response.status_code == 200:
        print(f"   ✅ API docs available at {BASE_URL}/api/docs")
    else:
        print(f"   ❌ API docs failed: {response.status_code}")
    return response.status_code == 200

def test_openapi_schema():
    """Test OpenAPI schema."""
    print("\n3. Testing OpenAPI Schema...")
    response = requests.get(f"{BASE_URL}/api/openapi.json")
    if response.status_code == 200:
        schema = response.json()
        print(f"   ✅ OpenAPI schema loaded: {schema.get('info', {}).get('title', 'Unknown')}")
        print(f"      Version: {schema.get('info', {}).get('version', 'Unknown')}")
    else:
        print(f"   ❌ OpenAPI schema failed: {response.status_code}")
    return response.status_code == 200

def test_redis_connection():
    """Test Redis through health check details."""
    print("\n4. Testing Redis Connection...")
    response = requests.get(f"{BASE_URL}/api/v1/health/redis")
    if response.status_code == 200:
        print(f"   ✅ Redis connection successful")
    elif response.status_code == 404:
        print(f"   ⚠️  Redis health endpoint not found (may not be implemented)")
    else:
        print(f"   ❌ Redis check failed: {response.status_code}")
    return response.status_code in [200, 404]

def test_database_connection():
    """Test database through health check."""
    print("\n5. Testing Database Connection...")
    response = requests.get(f"{BASE_URL}/api/v1/health/db")
    if response.status_code == 200:
        print(f"   ✅ Database connection successful")
    elif response.status_code == 404:
        print(f"   ⚠️  Database health endpoint not found (may not be implemented)")
    else:
        print(f"   ❌ Database check failed: {response.status_code}")
    return response.status_code in [200, 404]

def test_websocket_info():
    """Test WebSocket endpoint info."""
    print("\n6. Testing WebSocket Info...")
    # Check if websocket endpoint is documented
    response = requests.get(f"{BASE_URL}/api/openapi.json")
    if response.status_code == 200:
        schema = response.json()
        ws_path = None
        for path in schema.get('paths', {}):
            if 'websocket' in path.lower() or 'ws' in path:
                ws_path = path
                break
        if ws_path:
            print(f"   ✅ WebSocket endpoint found: {ws_path}")
        else:
            print(f"   ⚠️  WebSocket endpoint not documented in OpenAPI")
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("AENEAS API Testing Suite")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Target: {BASE_URL}")
    print("="*60)
    
    tests = [
        test_health,
        test_api_docs,
        test_openapi_schema,
        test_redis_connection,
        test_database_connection,
        test_websocket_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test error: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
    elif passed > total * 0.7:
        print("⚠️  Most tests passed, some issues detected")
    else:
        print("❌ Multiple test failures detected")
    
    print("="*60)
    
    # Additional info
    print("\n📝 Quick Access URLs:")
    print(f"   • API Documentation: {BASE_URL}/api/docs")
    print(f"   • ReDoc: {BASE_URL}/api/redoc")
    print(f"   • Health Check: {BASE_URL}/api/v1/health")
    print(f"   • WebSocket: ws://localhost:8000/api/v1/ws")
    
if __name__ == "__main__":
    main()
