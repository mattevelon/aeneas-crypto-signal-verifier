"""
Signal validation and risk assessment system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

import structlog

from src.core.market_data import market_data
from src.core.llm_client import LLMClient
from src.models import SignalDirection, RiskLevel

logger = structlog.get_logger()


class SignalValidator:
    """Validate and assess trading signals."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.validation_rules = self._init_validation_rules()
    
    def _init_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules."""
        return {
            "max_stop_loss_percentage": 10.0,  # Max 10% stop loss
            "min_risk_reward_ratio": 1.5,      # Minimum 1:1.5 risk/reward
            "max_leverage": 20,                 # Maximum leverage
            "min_volume_usd": 100000,           # Minimum daily volume
            "max_price_deviation": 2.0,         # Max 2% from current price
            "min_confidence_score": 30.0        # Minimum confidence
        }
    
    async def validate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive signal validation."""
        validation_results = {
            "is_valid": True,
            "confidence_score": signal_data.get("confidence_score", 50),
            "risk_level": signal_data.get("risk_level", RiskLevel.MEDIUM),
            "validations": {},
            "warnings": [],
            "errors": []
        }
        
        # Run all validations in parallel
        validation_tasks = [
            self._validate_market_data(signal_data),
            self._validate_risk_parameters(signal_data),
            self._validate_technical_levels(signal_data),
            self._validate_liquidity(signal_data)
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process validation results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Validation error: {result}")
                validation_results["warnings"].append(str(result))
            else:
                validation_results["validations"].update(result)
                
                if not result.get("valid", True):
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(result.get("reason", "Unknown error"))
                elif result.get("warning"):
                    validation_results["warnings"].append(result.get("warning"))
        
        # Adjust confidence based on validations
        validation_results["confidence_score"] = self._adjust_confidence(
            signal_data.get("confidence_score", 50),
            validation_results
        )
        
        # Determine final risk level
        validation_results["risk_level"] = self._assess_risk_level(
            signal_data,
            validation_results
        )
        
        return validation_results
    
    async def _validate_market_data(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against current market data."""
        try:
            pair = signal_data.get("pair")
            entry_price = signal_data.get("entry_price")
            
            if not pair or not entry_price:
                return {
                    "market_validation": {
                        "valid": False,
                        "reason": "Missing pair or entry price"
                    }
                }
            
            # Validate price is within market range
            price_validation = await market_data.validate_price(
                pair,
                entry_price,
                tolerance=self.validation_rules["max_price_deviation"] / 100
            )
            
            return {
                "market_validation": {
                    "valid": price_validation["valid"],
                    "current_price": price_validation.get("current_price"),
                    "deviation": price_validation.get("deviation_percentage"),
                    "reason": price_validation.get("reason")
                }
            }
            
        except Exception as e:
            logger.error(f"Market validation error: {e}")
            return {
                "market_validation": {
                    "valid": True,  # Don't fail on market data errors
                    "warning": "Unable to validate market data"
                }
            }
    
    async def _validate_risk_parameters(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk management parameters."""
        entry_price = signal_data.get("entry_price", 0)
        stop_loss = signal_data.get("stop_loss", 0)
        take_profits = signal_data.get("take_profits", [])
        direction = signal_data.get("direction")
        leverage = signal_data.get("leverage", 1)
        
        if not entry_price or not stop_loss:
            return {
                "risk_validation": {
                    "valid": False,
                    "reason": "Missing entry price or stop loss"
                }
            }
        
        # Calculate stop loss percentage
        if direction == SignalDirection.LONG:
            sl_percentage = ((entry_price - stop_loss) / entry_price) * 100
        else:
            sl_percentage = ((stop_loss - entry_price) / entry_price) * 100
        
        # Check stop loss limits
        if sl_percentage > self.validation_rules["max_stop_loss_percentage"]:
            return {
                "risk_validation": {
                    "valid": False,
                    "reason": f"Stop loss too far: {sl_percentage:.2f}%",
                    "stop_loss_percentage": sl_percentage
                }
            }
        
        # Calculate risk/reward ratio
        if take_profits:
            first_tp = take_profits[0]
            if direction == SignalDirection.LONG:
                reward = first_tp - entry_price
                risk = entry_price - stop_loss
            else:
                reward = entry_price - first_tp
                risk = stop_loss - entry_price
            
            if risk > 0:
                risk_reward_ratio = reward / risk
                
                if risk_reward_ratio < self.validation_rules["min_risk_reward_ratio"]:
                    return {
                        "risk_validation": {
                            "valid": True,
                            "warning": f"Low risk/reward ratio: {risk_reward_ratio:.2f}",
                            "risk_reward_ratio": risk_reward_ratio,
                            "stop_loss_percentage": sl_percentage
                        }
                    }
        
        # Check leverage
        if leverage > self.validation_rules["max_leverage"]:
            return {
                "risk_validation": {
                    "valid": False,
                    "reason": f"Leverage too high: {leverage}x",
                    "leverage": leverage
                }
            }
        
        return {
            "risk_validation": {
                "valid": True,
                "stop_loss_percentage": sl_percentage,
                "leverage": leverage
            }
        }
    
    async def _validate_technical_levels(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate technical price levels."""
        entry_price = signal_data.get("entry_price", 0)
        stop_loss = signal_data.get("stop_loss", 0)
        take_profits = signal_data.get("take_profits", [])
        direction = signal_data.get("direction")
        
        issues = []
        
        # Validate price levels make sense for direction
        if direction == SignalDirection.LONG:
            if stop_loss >= entry_price:
                issues.append("Stop loss above entry for long position")
            
            for i, tp in enumerate(take_profits):
                if tp <= entry_price:
                    issues.append(f"Take profit {i+1} below entry for long position")
                if i > 0 and tp <= take_profits[i-1]:
                    issues.append(f"Take profit {i+1} not progressive")
        
        elif direction == SignalDirection.SHORT:
            if stop_loss <= entry_price:
                issues.append("Stop loss below entry for short position")
            
            for i, tp in enumerate(take_profits):
                if tp >= entry_price:
                    issues.append(f"Take profit {i+1} above entry for short position")
                if i > 0 and tp >= take_profits[i-1]:
                    issues.append(f"Take profit {i+1} not progressive")
        
        if issues:
            return {
                "technical_validation": {
                    "valid": False,
                    "reason": "; ".join(issues),
                    "issues": issues
                }
            }
        
        return {
            "technical_validation": {
                "valid": True,
                "levels_consistent": True
            }
        }
    
    async def _validate_liquidity(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market liquidity."""
        try:
            pair = signal_data.get("pair")
            
            if not pair:
                return {
                    "liquidity_validation": {
                        "valid": True,
                        "warning": "Unable to check liquidity"
                    }
                }
            
            # Get 24h volume
            stats = await market_data.get_24h_stats(pair)
            
            if stats:
                volume_usd = stats.get("quote_volume", 0)
                
                if volume_usd < self.validation_rules["min_volume_usd"]:
                    return {
                        "liquidity_validation": {
                            "valid": True,
                            "warning": f"Low liquidity: ${volume_usd:,.0f} daily volume",
                            "volume_usd": volume_usd
                        }
                    }
                
                return {
                    "liquidity_validation": {
                        "valid": True,
                        "volume_usd": volume_usd
                    }
                }
            
        except Exception as e:
            logger.error(f"Liquidity validation error: {e}")
        
        return {
            "liquidity_validation": {
                "valid": True,
                "warning": "Unable to validate liquidity"
            }
        }
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        validation_results: Dict[str, Any]
    ) -> float:
        """Adjust confidence score based on validations."""
        confidence = base_confidence
        
        # Reduce confidence for warnings
        confidence -= len(validation_results.get("warnings", [])) * 5
        
        # Reduce confidence for errors
        confidence -= len(validation_results.get("errors", [])) * 10
        
        # Boost confidence for good risk/reward
        risk_validation = validation_results.get("validations", {}).get("risk_validation", {})
        if risk_validation.get("risk_reward_ratio", 0) > 3:
            confidence += 10
        
        # Ensure within bounds
        return max(0, min(100, confidence))
    
    def _assess_risk_level(
        self, 
        signal_data: Dict[str, Any], 
        validation_results: Dict[str, Any]
    ) -> RiskLevel:
        """Assess overall risk level."""
        risk_factors = []
        
        # Check leverage
        leverage = signal_data.get("leverage", 1)
        if leverage > 10:
            risk_factors.append("high")
        elif leverage > 5:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Check stop loss distance
        risk_validation = validation_results.get("validations", {}).get("risk_validation", {})
        sl_percentage = risk_validation.get("stop_loss_percentage", 0)
        
        if sl_percentage > 5:
            risk_factors.append("high")
        elif sl_percentage > 2:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Check liquidity
        liquidity_validation = validation_results.get("validations", {}).get("liquidity_validation", {})
        volume_usd = liquidity_validation.get("volume_usd", 0)
        
        if volume_usd < 500000:
            risk_factors.append("high")
        elif volume_usd < 1000000:
            risk_factors.append("medium")
        else:
            risk_factors.append("low")
        
        # Count risk levels
        high_count = risk_factors.count("high")
        low_count = risk_factors.count("low")
        
        if high_count >= 2:
            return RiskLevel.HIGH
        elif low_count >= 2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM
