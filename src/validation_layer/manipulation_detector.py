"""
Manipulation Detection Module

Detects potential market manipulation patterns including pump & dump schemes,
wash trading, spoofing, and other fraudulent activities.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

from src.config.settings import get_settings
from src.core.redis_client import market_cache
from src.core.database import get_async_session
from sqlalchemy import select, func, and_

logger = logging.getLogger(__name__)
settings = get_settings()


class ManipulationType(Enum):
    """Types of market manipulation detected."""
    NONE = "none"
    PUMP_AND_DUMP = "pump_and_dump"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    FRONT_RUNNING = "front_running"
    LAYERING = "layering"
    UNUSUAL_ACTIVITY = "unusual_activity"


@dataclass
class ManipulationIndicator:
    """Individual manipulation indicator."""
    indicator_type: str
    severity: float  # 0-1 scale
    description: str
    evidence: Dict[str, Any]


@dataclass
class ManipulationDetectionResult:
    """Result of manipulation detection analysis."""
    manipulation_detected: bool
    manipulation_type: ManipulationType
    confidence_score: float  # 0-100
    severity_level: str  # low, medium, high, critical
    indicators: List[ManipulationIndicator]
    risk_score: float  # 0-100
    recommendations: List[str]
    alert_required: bool


class ManipulationDetector:
    """Detects various forms of market manipulation."""
    
    # Detection thresholds
    PUMP_VOLUME_MULTIPLIER = 3.0  # 3x average volume
    PUMP_PRICE_INCREASE = 0.10  # 10% price increase
    DUMP_PRICE_DECREASE = 0.15  # 15% price decrease
    WASH_TRADE_THRESHOLD = 0.70  # 70% similarity threshold
    SPOOF_CANCEL_RATE = 0.80  # 80% order cancellation rate
    UNUSUAL_ACTIVITY_ZSCORE = 3.0  # 3 standard deviations
    
    def __init__(self):
        """Initialize manipulation detector."""
        self.price_history = {}
        self.volume_history = {}
        self.order_book_snapshots = deque(maxlen=100)
        
    async def detect_manipulation(
        self,
        signal: Dict[str, Any],
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> ManipulationDetectionResult:
        """
        Perform comprehensive manipulation detection.
        
        Args:
            signal: Trading signal to analyze
            market_data: Current market data
            historical_data: Historical price and volume data
            
        Returns:
            ManipulationDetectionResult
        """
        indicators = []
        recommendations = []
        
        pair = signal.get('pair', '')
        
        # 1. Pump and Dump Detection
        pump_result = await self._detect_pump_and_dump(
            pair, market_data, historical_data
        )
        if pump_result:
            indicators.append(pump_result)
            
        # 2. Wash Trading Detection
        wash_result = await self._detect_wash_trading(
            market_data, historical_data
        )
        if wash_result:
            indicators.append(wash_result)
            
        # 3. Spoofing Detection
        spoof_result = await self._detect_spoofing(
            market_data.get('order_book_depth', {})
        )
        if spoof_result:
            indicators.append(spoof_result)
            
        # 4. Unusual Activity Detection
        unusual_result = await self._detect_unusual_activity(
            market_data, historical_data
        )
        if unusual_result:
            indicators.append(unusual_result)
            
        # Calculate overall risk score
        risk_score = self._calculate_manipulation_risk_score(indicators)
        
        # Determine manipulation type and severity
        manipulation_type, severity = self._classify_manipulation(indicators)
        
        # Generate recommendations
        if manipulation_type != ManipulationType.NONE:
            recommendations = self._generate_recommendations(
                manipulation_type, severity, indicators
            )
            
        # Determine if alert is required
        alert_required = (
            manipulation_type != ManipulationType.NONE and 
            severity in ['high', 'critical']
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(indicators)
        
        return ManipulationDetectionResult(
            manipulation_detected=manipulation_type != ManipulationType.NONE,
            manipulation_type=manipulation_type,
            confidence_score=confidence,
            severity_level=severity,
            indicators=indicators,
            risk_score=risk_score,
            recommendations=recommendations,
            alert_required=alert_required
        )
    
    async def _detect_pump_and_dump(
        self,
        pair: str,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ManipulationIndicator]:
        """
        Detect pump and dump patterns.
        
        Indicators:
        - Sudden volume spike (>3x average)
        - Rapid price increase (>10% in short period)
        - Social media coordination
        - Subsequent price collapse
        
        Args:
            pair: Trading pair
            market_data: Current market data
            historical_data: Historical data
            
        Returns:
            ManipulationIndicator if detected
        """
        if not historical_data:
            return None
            
        current_volume = market_data.get('volume_24h', 0)
        current_price = market_data.get('current_price', 0)
        price_change_24h = market_data.get('price_change_24h', 0)
        
        # Get historical averages
        avg_volume = historical_data.get('avg_volume_7d', current_volume)
        avg_volatility = historical_data.get('avg_volatility_7d', 0.05)
        
        evidence = {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'price_change_24h': price_change_24h,
            'avg_volatility': avg_volatility * 100
        }
        
        # Check for pump indicators
        volume_spike = current_volume > (avg_volume * self.PUMP_VOLUME_MULTIPLIER)
        price_pump = price_change_24h > (self.PUMP_PRICE_INCREASE * 100)
        abnormal_volatility = abs(price_change_24h / 100) > (avg_volatility * 2)
        
        pump_score = 0
        if volume_spike:
            pump_score += 0.4
        if price_pump:
            pump_score += 0.3
        if abnormal_volatility:
            pump_score += 0.3
            
        if pump_score >= 0.6:
            # Check for dump phase
            recent_high = historical_data.get('high_24h', current_price)
            if current_price < recent_high * (1 - self.DUMP_PRICE_DECREASE):
                pump_score = min(pump_score + 0.2, 1.0)
                evidence['dump_detected'] = True
                evidence['price_drop_from_high'] = (recent_high - current_price) / recent_high
                
            return ManipulationIndicator(
                indicator_type="pump_and_dump",
                severity=pump_score,
                description=f"Potential pump & dump: Volume {evidence['volume_ratio']:.1f}x average, "
                           f"price change {price_change_24h:.1f}%",
                evidence=evidence
            )
            
        return None
    
    async def _detect_wash_trading(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ManipulationIndicator]:
        """
        Detect wash trading patterns.
        
        Indicators:
        - Repetitive trade patterns
        - Similar order sizes
        - Minimal price impact despite high volume
        - Synchronized buy/sell orders
        
        Args:
            market_data: Current market data
            historical_data: Historical data
            
        Returns:
            ManipulationIndicator if detected
        """
        order_book = market_data.get('order_book_depth', {})
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return None
            
        evidence = {
            'order_book_depth': len(bids) + len(asks),
            'bid_ask_spread': (asks[0][0] - bids[0][0]) / bids[0][0] if bids and asks else 0
        }
        
        # Analyze order patterns
        wash_indicators = 0
        
        # Check for similar order sizes
        bid_sizes = [order[1] for order in bids[:10]]
        ask_sizes = [order[1] for order in asks[:10]]
        
        if bid_sizes and ask_sizes:
            size_similarity = self._calculate_order_similarity(bid_sizes, ask_sizes)
            evidence['size_similarity'] = size_similarity
            
            if size_similarity > self.WASH_TRADE_THRESHOLD:
                wash_indicators += 0.4
                
        # Check for minimal price impact
        volume = market_data.get('volume_24h', 0)
        price_change = abs(market_data.get('price_change_24h', 0))
        
        if volume > 0:
            # High volume with low price change is suspicious
            price_impact = price_change / (volume / 1000000)  # Normalize by millions
            evidence['price_impact'] = price_impact
            
            if price_impact < 0.01 and volume > 100000:  # Low impact with high volume
                wash_indicators += 0.3
                
        # Check for repetitive patterns in order placement
        if historical_data:
            pattern_score = await self._analyze_order_patterns(historical_data)
            evidence['pattern_score'] = pattern_score
            
            if pattern_score > 0.7:
                wash_indicators += 0.3
                
        if wash_indicators >= 0.6:
            return ManipulationIndicator(
                indicator_type="wash_trading",
                severity=min(wash_indicators, 1.0),
                description=f"Potential wash trading: Size similarity {size_similarity:.2f}, "
                           f"low price impact despite volume",
                evidence=evidence
            )
            
        return None
    
    async def _detect_spoofing(
        self,
        order_book: Dict[str, Any]
    ) -> Optional[ManipulationIndicator]:
        """
        Detect spoofing (placing and canceling large orders).
        
        Indicators:
        - Large orders far from market price
        - High cancellation rate
        - Order book imbalance
        
        Args:
            order_book: Current order book data
            
        Returns:
            ManipulationIndicator if detected
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return None
            
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        evidence = {
            'mid_price': mid_price,
            'best_bid': bids[0][0],
            'best_ask': asks[0][0]
        }
        
        spoof_score = 0
        
        # Check for large orders far from market
        for i, (price, size) in enumerate(bids[1:6]):  # Check orders 2-6
            distance_from_mid = abs(mid_price - price) / mid_price
            if size > bids[0][1] * 5 and distance_from_mid > 0.005:  # 5x larger and >0.5% away
                spoof_score += 0.2
                evidence[f'large_bid_{i}'] = {'price': price, 'size': size, 'distance': distance_from_mid}
                
        for i, (price, size) in enumerate(asks[1:6]):
            distance_from_mid = abs(price - mid_price) / mid_price
            if size > asks[0][1] * 5 and distance_from_mid > 0.005:
                spoof_score += 0.2
                evidence[f'large_ask_{i}'] = {'price': price, 'size': size, 'distance': distance_from_mid}
                
        # Check order book imbalance
        total_bid_volume = sum(order[1] for order in bids[:10])
        total_ask_volume = sum(order[1] for order in asks[:10])
        
        if total_bid_volume > 0 and total_ask_volume > 0:
            imbalance = abs(total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            evidence['order_book_imbalance'] = imbalance
            
            if imbalance > 0.7:  # 70% imbalance
                spoof_score += 0.3
                
        if spoof_score >= 0.5:
            return ManipulationIndicator(
                indicator_type="spoofing",
                severity=min(spoof_score, 1.0),
                description="Potential spoofing: Large orders away from market price with order book imbalance",
                evidence=evidence
            )
            
        return None
    
    async def _detect_unusual_activity(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ManipulationIndicator]:
        """
        Detect unusual trading activity using statistical analysis.
        
        Args:
            market_data: Current market data
            historical_data: Historical data
            
        Returns:
            ManipulationIndicator if detected
        """
        if not historical_data:
            return None
            
        evidence = {}
        unusual_score = 0
        
        # Volume analysis
        current_volume = market_data.get('volume_24h', 0)
        avg_volume = historical_data.get('avg_volume_30d', current_volume)
        std_volume = historical_data.get('std_volume_30d', avg_volume * 0.3)
        
        if std_volume > 0:
            volume_zscore = (current_volume - avg_volume) / std_volume
            evidence['volume_zscore'] = volume_zscore
            
            if abs(volume_zscore) > self.UNUSUAL_ACTIVITY_ZSCORE:
                unusual_score += 0.3
                
        # Price volatility analysis
        price_change = market_data.get('price_change_24h', 0)
        avg_volatility = historical_data.get('avg_volatility_30d', 5)
        std_volatility = historical_data.get('std_volatility_30d', 2)
        
        if std_volatility > 0:
            volatility_zscore = (abs(price_change) - avg_volatility) / std_volatility
            evidence['volatility_zscore'] = volatility_zscore
            
            if volatility_zscore > self.UNUSUAL_ACTIVITY_ZSCORE:
                unusual_score += 0.3
                
        # Trade count analysis
        trade_count = market_data.get('trade_count_24h', 0)
        avg_trades = historical_data.get('avg_trades_30d', trade_count)
        std_trades = historical_data.get('std_trades_30d', avg_trades * 0.3)
        
        if std_trades > 0:
            trades_zscore = (trade_count - avg_trades) / std_trades
            evidence['trades_zscore'] = trades_zscore
            
            if abs(trades_zscore) > self.UNUSUAL_ACTIVITY_ZSCORE:
                unusual_score += 0.2
                
        # Check for coordinated activity
        if 'social_mentions' in historical_data:
            social_spike = historical_data['social_mentions'] > historical_data.get('avg_social_mentions', 0) * 3
            if social_spike:
                unusual_score += 0.2
                evidence['social_spike'] = True
                
        if unusual_score >= 0.5:
            return ManipulationIndicator(
                indicator_type="unusual_activity",
                severity=min(unusual_score, 1.0),
                description=f"Unusual activity detected: Volume Z-score {evidence.get('volume_zscore', 0):.2f}, "
                           f"Volatility Z-score {evidence.get('volatility_zscore', 0):.2f}",
                evidence=evidence
            )
            
        return None
    
    def _calculate_order_similarity(
        self,
        bid_sizes: List[float],
        ask_sizes: List[float]
    ) -> float:
        """
        Calculate similarity between bid and ask order sizes.
        
        Args:
            bid_sizes: List of bid order sizes
            ask_sizes: List of ask order sizes
            
        Returns:
            Similarity score (0-1)
        """
        if not bid_sizes or not ask_sizes:
            return 0.0
            
        # Calculate coefficient of variation
        bid_cv = np.std(bid_sizes) / np.mean(bid_sizes) if np.mean(bid_sizes) > 0 else 1
        ask_cv = np.std(ask_sizes) / np.mean(ask_sizes) if np.mean(ask_sizes) > 0 else 1
        
        # Low variation suggests similar order sizes (potential wash trading)
        similarity = 1 - (bid_cv + ask_cv) / 2
        
        return max(0, min(similarity, 1))
    
    async def _analyze_order_patterns(
        self,
        historical_data: Dict[str, Any]
    ) -> float:
        """
        Analyze historical order patterns for repetitive behavior.
        
        Args:
            historical_data: Historical order data
            
        Returns:
            Pattern score (0-1)
        """
        # Simplified pattern analysis
        # In production, this would analyze actual order flow data
        orders = historical_data.get('recent_orders', [])
        
        if len(orders) < 10:
            return 0.0
            
        # Look for repetitive patterns in order sizes and timing
        sizes = [order.get('size', 0) for order in orders]
        times = [order.get('timestamp', 0) for order in orders]
        
        # Check for similar sizes
        size_std = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1
        
        # Check for regular timing intervals
        if len(times) > 1:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            interval_std = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
        else:
            interval_std = 1
            
        # Low variation in both suggests automated/coordinated trading
        pattern_score = 1 - (size_std + interval_std) / 2
        
        return max(0, min(pattern_score, 1))
    
    def _calculate_manipulation_risk_score(
        self,
        indicators: List[ManipulationIndicator]
    ) -> float:
        """
        Calculate overall manipulation risk score.
        
        Args:
            indicators: List of detected indicators
            
        Returns:
            Risk score (0-100)
        """
        if not indicators:
            return 0.0
            
        # Weight different manipulation types
        weights = {
            'pump_and_dump': 1.0,
            'wash_trading': 0.8,
            'spoofing': 0.7,
            'unusual_activity': 0.5
        }
        
        total_score = 0
        total_weight = 0
        
        for indicator in indicators:
            weight = weights.get(indicator.indicator_type, 0.5)
            total_score += indicator.severity * weight
            total_weight += weight
            
        if total_weight > 0:
            risk_score = (total_score / total_weight) * 100
        else:
            risk_score = 0
            
        return min(risk_score, 100)
    
    def _classify_manipulation(
        self,
        indicators: List[ManipulationIndicator]
    ) -> Tuple[ManipulationType, str]:
        """
        Classify the type and severity of manipulation.
        
        Args:
            indicators: List of detected indicators
            
        Returns:
            Tuple of (ManipulationType, severity_level)
        """
        if not indicators:
            return ManipulationType.NONE, "none"
            
        # Find dominant manipulation type
        type_scores = {}
        for indicator in indicators:
            type_name = indicator.indicator_type
            if type_name not in type_scores:
                type_scores[type_name] = 0
            type_scores[type_name] += indicator.severity
            
        # Get highest scoring type
        dominant_type = max(type_scores.items(), key=lambda x: x[1])
        
        # Map to ManipulationType
        type_mapping = {
            'pump_and_dump': ManipulationType.PUMP_AND_DUMP,
            'wash_trading': ManipulationType.WASH_TRADING,
            'spoofing': ManipulationType.SPOOFING,
            'unusual_activity': ManipulationType.UNUSUAL_ACTIVITY
        }
        
        manipulation_type = type_mapping.get(dominant_type[0], ManipulationType.UNUSUAL_ACTIVITY)
        
        # Determine severity
        max_severity = max(indicator.severity for indicator in indicators)
        if max_severity >= 0.8:
            severity = "critical"
        elif max_severity >= 0.6:
            severity = "high"
        elif max_severity >= 0.4:
            severity = "medium"
        else:
            severity = "low"
            
        return manipulation_type, severity
    
    def _generate_recommendations(
        self,
        manipulation_type: ManipulationType,
        severity: str,
        indicators: List[ManipulationIndicator]
    ) -> List[str]:
        """
        Generate recommendations based on detected manipulation.
        
        Args:
            manipulation_type: Type of manipulation detected
            severity: Severity level
            indicators: List of indicators
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if manipulation_type == ManipulationType.PUMP_AND_DUMP:
            recommendations.append("AVOID: High risk of pump & dump scheme")
            recommendations.append("Wait for price and volume to normalize")
            recommendations.append("Consider setting tight stop-losses if entering")
            
        elif manipulation_type == ManipulationType.WASH_TRADING:
            recommendations.append("CAUTION: Artificial volume detected")
            recommendations.append("True liquidity may be significantly lower")
            recommendations.append("Use limit orders to avoid slippage")
            
        elif manipulation_type == ManipulationType.SPOOFING:
            recommendations.append("WARNING: Order book manipulation detected")
            recommendations.append("Large orders may disappear before execution")
            recommendations.append("Consider using smaller position sizes")
            
        elif manipulation_type == ManipulationType.UNUSUAL_ACTIVITY:
            recommendations.append("MONITOR: Unusual market activity detected")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider reducing position size")
            
        # Add severity-based recommendations
        if severity in ["critical", "high"]:
            recommendations.append("Strongly recommend avoiding this trade")
            recommendations.append("Report suspicious activity to exchange")
        elif severity == "medium":
            recommendations.append("Proceed with extreme caution")
            recommendations.append("Consider paper trading only")
            
        return recommendations
    
    def _calculate_confidence_score(
        self,
        indicators: List[ManipulationIndicator]
    ) -> float:
        """
        Calculate confidence in manipulation detection.
        
        Args:
            indicators: List of detected indicators
            
        Returns:
            Confidence score (0-100)
        """
        if not indicators:
            return 0.0
            
        # More indicators increase confidence
        indicator_count_score = min(len(indicators) / 4, 1) * 40
        
        # Higher severity increases confidence
        max_severity = max(indicator.severity for indicator in indicators) if indicators else 0
        severity_score = max_severity * 40
        
        # Consistency across indicators increases confidence
        severities = [ind.severity for ind in indicators]
        if len(severities) > 1:
            consistency = 1 - np.std(severities) / np.mean(severities)
        else:
            consistency = 0.5
        consistency_score = consistency * 20
        
        confidence = indicator_count_score + severity_score + consistency_score
        
        return min(confidence, 100)
