#!/usr/bin/env python3
"""Fix for signal detection patterns."""

# Additional patterns to add to pattern_engine.py

ADDITIONAL_PATTERNS = '''
            # Entry patterns with dollar signs
            SignalPattern(
                name="entry_dollar_single",
                pattern=r"(?:entry|buy|long)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.9
            ),
            SignalPattern(
                name="entry_dollar_range",
                pattern=r"(?:entry|buy|long)[\s:]*\$?([\d,]+(?:\.\d+)?)\s*[-–]\s*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.95
            ),
            
            # Stop Loss with dollar signs
            SignalPattern(
                name="stop_loss_dollar",
                pattern=r"(?:stop\s*loss|sl|stop)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.95
            ),
            
            # Take Profit with dollar signs
            SignalPattern(
                name="take_profit_dollar",
                pattern=r"(?:tp\d?|target\s*\d?|take\s*profit\s*\d?)[\s:]*\$?([\d,]+(?:\.\d+)?)",
                confidence_weight=0.9
            ),
            
            # Direction patterns
            SignalPattern(
                name="direction_long",
                pattern=r"(?:direction|side|position)[\s:]*(?i)(long|buy)",
                confidence_weight=0.9,
                signal_type=SignalType.LONG
            ),
            SignalPattern(
                name="direction_short",
                pattern=r"(?:direction|side|position)[\s:]*(?i)(short|sell)",
                confidence_weight=0.9,
                signal_type=SignalType.SHORT
            ),
'''

print("Add these patterns to src/signal_detection/pattern_engine.py in the _initialize_patterns method")
print(ADDITIONAL_PATTERNS)
