#!/usr/bin/env python3
"""Test the signal processing pipeline end-to-end."""

import asyncio
import json
from datetime import datetime
from src.signal_detection.signal_detector import SignalDetector
from src.context_management.context_manager import ContextManager
from src.ai_integration.ai_analyzer import AIAnalyzer
from src.analysis_processing.validation_framework import ValidationFramework
from src.analysis_processing.decision_engine import DecisionEngine
from src.analysis_processing.result_processor import ResultProcessor

async def test_signal_pipeline():
    """Test the complete signal processing pipeline."""
    
    # Sample trading signal message
    test_message = """
    🔥 SIGNAL ALERT 🔥
    
    Pair: BTC/USDT
    Direction: LONG
    
    Entry: $42,500 - $43,000
    
    Targets:
    TP1: $43,500
    TP2: $44,200
    TP3: $45,000
    
    Stop Loss: $41,800
    
    Risk: Medium
    Leverage: 5x
    
    Analysis: Strong support at $42,000, expecting bounce to resistance levels.
    Volume increasing, RSI oversold on 4H timeframe.
    """
    
    print("=" * 60)
    print("TESTING SIGNAL PROCESSING PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Signal Detection
        print("\n1. SIGNAL DETECTION")
        print("-" * 30)
        detector = SignalDetector()
        detected = await detector.detect_signal(test_message)
        
        if detected:
            print(f"✅ Signal detected with confidence: {detected.get('confidence_score', 0):.1f}%")
            print(f"   Pair: {detected.get('pair', 'N/A')}")
            print(f"   Direction: {detected.get('direction', 'N/A')}")
            print(f"   Entry: {detected.get('entry_price', 'N/A')}")
            print(f"   Stop Loss: {detected.get('stop_loss', 'N/A')}")
            print(f"   Take Profits: {detected.get('take_profits', [])}")
        else:
            print("❌ No signal detected")
            return
        
        # Step 2: Context Building
        print("\n2. CONTEXT BUILDING")
        print("-" * 30)
        context_mgr = ContextManager()
        context = await context_mgr.build_context(detected)
        print(f"✅ Context built with {len(context)} data points")
        print(f"   Historical signals: {context.get('historical_signals', 0)}")
        print(f"   Market data: {context.get('market_data', {}).get('current_price', 'N/A')}")
        
        # Step 3: Validation
        print("\n3. VALIDATION")
        print("-" * 30)
        validator = ValidationFramework()
        # Create mock AI analysis for testing
        ai_analysis = {
            'confidence': 75.0,
            'risk_score': 65.0,
            'recommendation': 'EXECUTE',
            'reasoning': 'Test analysis'
        }
        validation_result = validator.validate(
            signal_data=detected,
            ai_analysis=ai_analysis,
            context=context
        )
        print(f"✅ Validation score: {validation_result.score:.1f}%")
        print(f"   Status: {validation_result.status.value}")
        print(f"   Checks passed: {len(validation_result.checks_passed)}")
        print(f"   Checks failed: {len(validation_result.checks_failed)}")
        print(f"   Warnings: {len(validation_result.warnings)}")
        
        # Step 4: Decision Making
        print("\n4. DECISION ENGINE")
        print("-" * 30)
        decision_engine = DecisionEngine()
        # Create enhanced signal for decision making
        enhanced_signal = {
            **detected,
            'optimized_entry': detected.get('entry_price', 0),
            'optimized_stop_loss': detected.get('stop_loss', 0),
            'position_size': 0.01,  # 1% of portfolio
            'risk_adjusted': True,
            'enhancement_score': 75.0,  # Mock enhancement score
            'risk_adjustments': {
                'max_loss': 0.02,  # 2% max loss
                'position_size': 0.01,
                'adjusted_leverage': 1
            },
            'execution_strategy': {
                'type': 'limit',
                'urgency': 'medium'
            }
        }
        # Convert to object-like dict for attribute access
        from types import SimpleNamespace
        enhanced_signal = SimpleNamespace(**enhanced_signal)
        decision = decision_engine.make_decision(
            signal_data=detected,
            ai_analysis=ai_analysis,
            validation_result=validation_result,
            enhanced_signal=enhanced_signal,
            context=context
        )
        print(f"✅ Decision: {decision.action.value.upper()}")
        print(f"   Confidence: {decision.confidence}")
        print(f"   Reasoning: {decision.reasoning}")
        
        # Step 5: Result Processing
        print("\n5. RESULT PROCESSING")
        print("-" * 30)
        processor = ResultProcessor()
        final_result = await processor.process_result(
            signal_data=detected,
            ai_analysis=ai_analysis,
            validation_result=validation_result,
            enhanced_signal=enhanced_signal,
            decision=decision
        )
        
        print(f"✅ Result processed and saved")
        print(f"   ID: {final_result.get('id', 'N/A')}")
        print(f"   Status: {final_result.get('status', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("PIPELINE TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return final_result
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_signal_pipeline())
    
    if result:
        print("\n📊 Final Result Summary:")
        print(json.dumps(result, indent=2, default=str))
