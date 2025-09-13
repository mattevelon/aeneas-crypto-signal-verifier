"""
Signal detection and extraction system.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
import json

import structlog

from src.core.llm_client import LLMClient
from src.core.redis_client import signal_cache
from src.models import SignalDirection, RiskLevel

logger = structlog.get_logger()


class SignalDetector:
    """Detect and extract trading signals from text."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for signal detection."""
        return {
            # Cryptocurrency pairs
            "pair": re.compile(
                r'\b([A-Z]{2,10})[/\-_\s]?(USDT?|BTC|ETH|BNB|BUSD)\b',
                re.IGNORECASE
            ),
            
            # Entry price patterns
            "entry": re.compile(
                r'(?:entry|buy|long|short)[\s:@]*(?:at|price|zone)?[\s:]*'
                r'(\$?[\d,]+\.?\d*)',
                re.IGNORECASE
            ),
            
            # Stop loss patterns
            "stop_loss": re.compile(
                r'(?:stop[\s\-]?loss|sl|stop)[\s:@]*'
                r'(\$?[\d,]+\.?\d*)',
                re.IGNORECASE
            ),
            
            # Take profit patterns
            "take_profit": re.compile(
                r'(?:take[\s\-]?profit|tp|target)[\s:#]*(\d+)?[\s:@]*'
                r'(\$?[\d,]+\.?\d*)',
                re.IGNORECASE
            ),
            
            # Direction patterns
            "long": re.compile(r'\b(?:long|buy|bullish)\b', re.IGNORECASE),
            "short": re.compile(r'\b(?:short|sell|bearish)\b', re.IGNORECASE),
            
            # Leverage patterns
            "leverage": re.compile(
                r'(?:leverage|lev)[\s:]*(\d+)[xX]?',
                re.IGNORECASE
            ),
            
            # Risk patterns
            "risk": re.compile(
                r'(?:risk|position[\s\-]?size)[\s:]*(\d+(?:\.\d+)?)\s*%?',
                re.IGNORECASE
            )
        }
    
    async def detect(self, text: str) -> bool:
        """Quick detection if text contains a trading signal."""
        if not text:
            return False
        
        # Check for minimum required components
        has_pair = bool(self.patterns["pair"].search(text))
        has_entry = bool(self.patterns["entry"].search(text))
        has_direction = bool(
            self.patterns["long"].search(text) or 
            self.patterns["short"].search(text)
        )
        
        # Signal must have at least pair and entry/direction
        return has_pair and (has_entry or has_direction)
    
    async def extract(self, text: str, channel_id: int = None) -> Optional[Dict[str, Any]]:
        """Extract signal details from text."""
        if not await self.detect(text):
            return None
        
        try:
            # Extract basic components
            signal_data = {
                "pair": self._extract_pair(text),
                "direction": self._extract_direction(text),
                "entry_price": self._extract_entry_price(text),
                "stop_loss": self._extract_stop_loss(text),
                "take_profits": self._extract_take_profits(text),
                "leverage": self._extract_leverage(text),
                "risk_percentage": self._extract_risk(text)
            }
            
            # Validate required fields
            if not all([signal_data["pair"], signal_data["direction"], signal_data["entry_price"]]):
                return None
            
            # Use LLM for deeper analysis
            llm_analysis = await self._analyze_with_llm(text, signal_data)
            
            # Combine extracted and analyzed data
            signal_data.update({
                "confidence_score": llm_analysis.get("confidence_score", 50),
                "risk_level": self._determine_risk_level(signal_data, llm_analysis),
                "justification": llm_analysis.get("justification", {}),
                "metadata": {
                    "raw_text": text[:500],  # Store first 500 chars
                    "channel_id": channel_id,
                    "llm_analysis": llm_analysis
                }
            })
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error extracting signal: {e}")
            return None
    
    def _extract_pair(self, text: str) -> Optional[str]:
        """Extract cryptocurrency pair."""
        match = self.patterns["pair"].search(text)
        if match:
            base, quote = match.groups()
            return f"{base.upper()}/{quote.upper()}"
        return None
    
    def _extract_direction(self, text: str) -> Optional[SignalDirection]:
        """Extract signal direction."""
        if self.patterns["long"].search(text):
            return SignalDirection.LONG
        elif self.patterns["short"].search(text):
            return SignalDirection.SHORT
        return None
    
    def _extract_entry_price(self, text: str) -> Optional[float]:
        """Extract entry price."""
        match = self.patterns["entry"].search(text)
        if match:
            price_str = match.group(1).replace('$', '').replace(',', '')
            try:
                return float(price_str)
            except ValueError:
                pass
        return None
    
    def _extract_stop_loss(self, text: str) -> Optional[float]:
        """Extract stop loss price."""
        match = self.patterns["stop_loss"].search(text)
        if match:
            price_str = match.group(1).replace('$', '').replace(',', '')
            try:
                return float(price_str)
            except ValueError:
                pass
        return None
    
    def _extract_take_profits(self, text: str) -> List[float]:
        """Extract take profit levels."""
        take_profits = []
        for match in self.patterns["take_profit"].finditer(text):
            price_str = match.group(2).replace('$', '').replace(',', '')
            try:
                tp = float(price_str)
                if tp not in take_profits:
                    take_profits.append(tp)
            except ValueError:
                pass
        
        return sorted(take_profits)
    
    def _extract_leverage(self, text: str) -> Optional[int]:
        """Extract leverage."""
        match = self.patterns["leverage"].search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_risk(self, text: str) -> Optional[float]:
        """Extract risk percentage."""
        match = self.patterns["risk"].search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    async def _analyze_with_llm(
        self, 
        text: str, 
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM for deeper signal analysis."""
        try:
            # Check cache first
            cache_key = f"llm_analysis:{hash(text)}"
            cached = await signal_cache.get(cache_key)
            if cached:
                return cached
            
            # Prepare context for LLM
            context = {
                "extracted_data": extracted_data,
                "text_length": len(text),
                "has_stop_loss": bool(extracted_data.get("stop_loss")),
                "has_take_profits": bool(extracted_data.get("take_profits")),
                "leverage": extracted_data.get("leverage")
            }
            
            # Get LLM analysis
            result = await self.llm_client.analyze_signal(text, context)
            
            if result["success"]:
                # Parse LLM response
                analysis = self._parse_llm_response(result["analysis"])
                
                # Cache the result
                await signal_cache.set(cache_key, analysis, ttl=3600)
                
                return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
        
        # Return default analysis if LLM fails
        return {
            "confidence_score": 50,
            "risk_assessment": "medium",
            "justification": {
                "technical": "Unable to perform deep analysis",
                "risk": "Default risk assessment"
            }
        }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Try to extract JSON if present
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to text parsing
        analysis = {
            "confidence_score": 50,
            "risk_assessment": "medium",
            "justification": {}
        }
        
        # Extract confidence score
        confidence_match = re.search(r'confidence[:\s]+(\d+)', response, re.IGNORECASE)
        if confidence_match:
            analysis["confidence_score"] = min(100, int(confidence_match.group(1)))
        
        # Extract risk level
        if re.search(r'\bhigh\s+risk\b', response, re.IGNORECASE):
            analysis["risk_assessment"] = "high"
        elif re.search(r'\blow\s+risk\b', response, re.IGNORECASE):
            analysis["risk_assessment"] = "low"
        
        # Store full response as justification
        analysis["justification"] = {
            "full_analysis": response[:1000]  # First 1000 chars
        }
        
        return analysis
    
    def _determine_risk_level(
        self, 
        signal_data: Dict[str, Any], 
        llm_analysis: Dict[str, Any]
    ) -> RiskLevel:
        """Determine overall risk level."""
        risk_factors = []
        
        # Check stop loss distance
        if signal_data.get("stop_loss") and signal_data.get("entry_price"):
            sl_distance = abs(
                (signal_data["stop_loss"] - signal_data["entry_price"]) / 
                signal_data["entry_price"]
            ) * 100
            
            if sl_distance > 5:
                risk_factors.append("high")
            elif sl_distance > 2:
                risk_factors.append("medium")
            else:
                risk_factors.append("low")
        
        # Check leverage
        leverage = signal_data.get("leverage", 1)
        if leverage > 10:
            risk_factors.append("high")
        elif leverage > 5:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Consider LLM assessment
        llm_risk = llm_analysis.get("risk_assessment", "medium")
        risk_factors.append(llm_risk)
        
        # Determine overall risk
        high_count = risk_factors.count("high")
        low_count = risk_factors.count("low")
        
        if high_count >= 2:
            return RiskLevel.HIGH
        elif low_count >= 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM
