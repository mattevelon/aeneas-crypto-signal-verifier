#!/usr/bin/env python3
"""Debug script for signal detection."""

import asyncio
from src.signal_detection.signal_detector import SignalDetector
from src.signal_detection.pattern_engine import PatternRecognitionEngine

async def debug_signal_detection():
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
    
    # Test pattern engine first
    print("Testing Pattern Engine...")
    engine = PatternRecognitionEngine()
    patterns = engine.detect_signals(test_message)
    print(f"Detected patterns: {len(patterns)}")
    for p in patterns[:5]:
        print(f"  - {p}")
    
    if patterns:
        confidence = engine.calculate_confidence(patterns)
        print(f"Pattern confidence: {confidence}")
        components = engine.extract_key_components(patterns)
        print(f"Extracted components: {components}")
    
    # Test full detector
    print("\nTesting Full Signal Detector...")
    detector = SignalDetector()
    result = await detector.detect_signal(test_message)
    
    if result:
        print(f"✅ Signal detected!")
        print(f"  - ID: {result.get('signal_id')}")
        print(f"  - Params: {result.get('trading_params')}")
    else:
        print("❌ No signal detected")
        
        # Debug step by step
        from src.signal_detection.parameter_extractor import SignalParameterExtractor
        extractor = SignalParameterExtractor()
        params = extractor.extract_parameters(test_message, patterns if patterns else [])
        print(f"  - Parameters extracted: {params}")

if __name__ == "__main__":
    asyncio.run(debug_signal_detection())
