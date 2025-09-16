"""Cross-Channel Validator for signal consensus and validation."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_async_session

logger = logging.getLogger(__name__)


@dataclass
class CrossChannelValidation:
    """Cross-channel validation results."""
    consensus_score: float
    similar_signals: List[Dict[str, Any]]
    channel_agreement: Dict[str, float]
    temporal_correlation: float
    conflict_indicators: List[str]
    validation_status: str  # validated, partial, conflict, insufficient


class CrossChannelValidator:
    """
    Validates signals across multiple channels for consensus and conflict detection.
    Implements temporal correlation analysis and reputation weighting.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize cross-channel validator.
        
        Args:
            similarity_threshold: Minimum similarity for signal matching
        """
        self.similarity_threshold = similarity_threshold
        self.time_window_minutes = 30  # Time window for related signals
        self.min_channels_for_consensus = 2
        logger.info("Initialized CrossChannelValidator")
    
    async def validate_signal(self,
                             pair: str,
                             direction: str,
                             entry_price: float,
                             channel_id: int,
                             timestamp: Optional[datetime] = None) -> CrossChannelValidation:
        """
        Validate a signal against other channels.
        
        Args:
            pair: Trading pair
            direction: Signal direction (long/short)
            entry_price: Entry price
            channel_id: Source channel ID
            timestamp: Signal timestamp
            
        Returns:
            Cross-channel validation results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Find similar signals from other channels
        similar_signals = await self._find_similar_signals(
            pair, direction, entry_price, channel_id, timestamp
        )
        
        # Calculate channel agreement
        channel_agreement = await self._calculate_channel_agreement(
            similar_signals, channel_id
        )
        
        # Analyze temporal correlation
        temporal_correlation = self._analyze_temporal_correlation(
            similar_signals, timestamp
        )
        
        # Detect conflicts
        conflicts = await self._detect_conflicts(
            pair, direction, entry_price, similar_signals
        )
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(
            similar_signals, channel_agreement, temporal_correlation, conflicts
        )
        
        # Determine validation status
        validation_status = self._determine_validation_status(
            consensus_score, len(similar_signals), conflicts
        )
        
        return CrossChannelValidation(
            consensus_score=consensus_score,
            similar_signals=similar_signals,
            channel_agreement=channel_agreement,
            temporal_correlation=temporal_correlation,
            conflict_indicators=conflicts,
            validation_status=validation_status
        )
    
    async def _find_similar_signals(self,
                                   pair: str,
                                   direction: str,
                                   entry_price: float,
                                   exclude_channel: int,
                                   timestamp: datetime) -> List[Dict[str, Any]]:
        """Find similar signals from other channels within time window."""
        window_start = timestamp - timedelta(minutes=self.time_window_minutes)
        window_end = timestamp + timedelta(minutes=self.time_window_minutes)
        
        async with get_async_session() as session:
            # Query for similar signals
            query = select('signals').where(
                and_(
                    'signals.pair' == pair,
                    'signals.direction' == direction,
                    'signals.source_channel_id' != exclude_channel,
                    'signals.created_at' >= window_start,
                    'signals.created_at' <= window_end
                )
            )
            
            result = await session.execute(query)
            signals = result.fetchall()
            
            # Calculate similarity and filter
            similar = []
            for signal in signals:
                similarity = self._calculate_similarity(
                    entry_price,
                    float(signal.entry_price),
                    signal.confidence_score
                )
                
                if similarity >= self.similarity_threshold:
                    # Get channel info
                    channel_info = await self._get_channel_info(
                        signal.source_channel_id, session
                    )
                    
                    similar.append({
                        'signal_id': signal.id,
                        'channel_id': signal.source_channel_id,
                        'channel_name': channel_info.get('name', 'Unknown'),
                        'channel_reputation': channel_info.get('reputation', 0.5),
                        'entry_price': float(signal.entry_price),
                        'confidence_score': signal.confidence_score,
                        'similarity': similarity,
                        'time_diff_minutes': abs((signal.created_at - timestamp).total_seconds() / 60),
                        'created_at': signal.created_at.isoformat()
                    })
            
            return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(self, price1: float, price2: float, confidence: float) -> float:
        """Calculate similarity between two signals."""
        # Price similarity (inverse of percentage difference)
        price_diff_pct = abs(price1 - price2) / price1 if price1 > 0 else 1
        price_similarity = max(0, 1 - price_diff_pct)
        
        # Weight by confidence
        weighted_similarity = price_similarity * (0.7 + 0.3 * (confidence / 100))
        
        return round(weighted_similarity, 4)
    
    async def _get_channel_info(self, channel_id: int, session: AsyncSession) -> Dict[str, Any]:
        """Get channel information and statistics."""
        query = select('channel_statistics').where(
            'channel_statistics.channel_id' == channel_id
        )
        
        result = await session.execute(query)
        stats = result.fetchone()
        
        if stats:
            return {
                'name': stats.channel_name,
                'reputation': stats.reputation_score or 0.5,
                'success_rate': (stats.successful_signals / stats.total_signals 
                               if stats.total_signals > 0 else 0.5)
            }
        
        return {'name': 'Unknown', 'reputation': 0.5, 'success_rate': 0.5}
    
    async def _calculate_channel_agreement(self,
                                          similar_signals: List[Dict[str, Any]],
                                          source_channel: int) -> Dict[str, float]:
        """Calculate agreement metrics across channels."""
        if not similar_signals:
            return {
                'agreement_rate': 0,
                'weighted_agreement': 0,
                'channel_count': 0,
                'avg_reputation': 0
            }
        
        # Group by channel
        channels = {}
        for signal in similar_signals:
            channel_id = signal['channel_id']
            if channel_id not in channels:
                channels[channel_id] = {
                    'signals': [],
                    'reputation': signal['channel_reputation']
                }
            channels[channel_id]['signals'].append(signal)
        
        # Calculate agreement metrics
        total_weight = 0
        weighted_agreement = 0
        
        for channel_id, data in channels.items():
            # Average similarity for this channel
            avg_similarity = np.mean([s['similarity'] for s in data['signals']])
            
            # Weight by reputation
            weight = data['reputation']
            weighted_agreement += avg_similarity * weight
            total_weight += weight
        
        return {
            'agreement_rate': len(channels) / max(len(channels) + 1, self.min_channels_for_consensus),
            'weighted_agreement': weighted_agreement / total_weight if total_weight > 0 else 0,
            'channel_count': len(channels),
            'avg_reputation': np.mean([d['reputation'] for d in channels.values()])
        }
    
    def _analyze_temporal_correlation(self,
                                     similar_signals: List[Dict[str, Any]],
                                     reference_time: datetime) -> float:
        """Analyze temporal correlation of signals."""
        if not similar_signals:
            return 0
        
        # Calculate time clustering score
        time_diffs = [s['time_diff_minutes'] for s in similar_signals]
        
        # Signals closer in time have higher correlation
        correlation_scores = []
        for diff in time_diffs:
            if diff <= 5:  # Within 5 minutes
                correlation_scores.append(1.0)
            elif diff <= 15:  # Within 15 minutes
                correlation_scores.append(0.7)
            elif diff <= 30:  # Within 30 minutes
                correlation_scores.append(0.4)
            else:
                correlation_scores.append(0.1)
        
        return round(np.mean(correlation_scores), 4)
    
    async def _detect_conflicts(self,
                               pair: str,
                               direction: str,
                               entry_price: float,
                               similar_signals: List[Dict[str, Any]]) -> List[str]:
        """Detect conflicting signals."""
        conflicts = []
        
        # Check for opposite direction signals
        async with get_async_session() as session:
            opposite_direction = 'short' if direction == 'long' else 'long'
            
            query = select(func.count('signals.id')).where(
                and_(
                    'signals.pair' == pair,
                    'signals.direction' == opposite_direction,
                    'signals.created_at' >= datetime.now() - timedelta(minutes=self.time_window_minutes)
                )
            )
            
            result = await session.execute(query)
            opposite_count = result.scalar()
            
            if opposite_count > 0:
                conflicts.append(f"opposite_signals:{opposite_count}")
        
        # Check for price divergence in similar signals
        if similar_signals:
            prices = [s['entry_price'] for s in similar_signals]
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            
            if price_std / price_mean > 0.02:  # More than 2% standard deviation
                conflicts.append("price_divergence")
        
        # Check for low reputation consensus
        if similar_signals:
            reputations = [s['channel_reputation'] for s in similar_signals]
            if np.mean(reputations) < 0.4:
                conflicts.append("low_reputation_consensus")
        
        return conflicts
    
    def _calculate_consensus_score(self,
                                  similar_signals: List[Dict[str, Any]],
                                  channel_agreement: Dict[str, float],
                                  temporal_correlation: float,
                                  conflicts: List[str]) -> float:
        """Calculate overall consensus score."""
        if not similar_signals:
            return 0
        
        # Base score components
        signal_count_score = min(len(similar_signals) / 5, 1.0) * 25  # Max 25 points
        agreement_score = channel_agreement['weighted_agreement'] * 25  # Max 25 points
        temporal_score = temporal_correlation * 25  # Max 25 points
        reputation_score = channel_agreement['avg_reputation'] * 25  # Max 25 points
        
        base_score = signal_count_score + agreement_score + temporal_score + reputation_score
        
        # Apply penalties for conflicts
        if 'opposite_signals' in str(conflicts):
            base_score *= 0.5
        if 'price_divergence' in conflicts:
            base_score *= 0.8
        if 'low_reputation_consensus' in conflicts:
            base_score *= 0.7
        
        return round(min(100, max(0, base_score)), 2)
    
    def _determine_validation_status(self,
                                    consensus_score: float,
                                    signal_count: int,
                                    conflicts: List[str]) -> str:
        """Determine validation status based on analysis."""
        if signal_count < self.min_channels_for_consensus:
            return 'insufficient'
        
        if conflicts and any('opposite' in c for c in conflicts):
            return 'conflict'
        
        if consensus_score >= 70:
            return 'validated'
        elif consensus_score >= 40:
            return 'partial'
        else:
            return 'conflict'
    
    async def get_channel_correlation_matrix(self, 
                                            pairs: List[str],
                                            time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Build correlation matrix between channels for given pairs.
        
        Args:
            pairs: List of trading pairs to analyze
            time_window_hours: Hours to look back
            
        Returns:
            Channel correlation matrix and statistics
        """
        window_start = datetime.now() - timedelta(hours=time_window_hours)
        
        async with get_async_session() as session:
            # Get all signals in time window for specified pairs
            query = select('signals').where(
                and_(
                    'signals.pair'.in_(pairs),
                    'signals.created_at' >= window_start
                )
            ).order_by('signals.created_at')
            
            result = await session.execute(query)
            signals = result.fetchall()
            
            # Group signals by channel
            channel_signals = {}
            for signal in signals:
                channel_id = signal.source_channel_id
                if channel_id not in channel_signals:
                    channel_signals[channel_id] = []
                channel_signals[channel_id].append({
                    'pair': signal.pair,
                    'direction': signal.direction,
                    'timestamp': signal.created_at,
                    'confidence': signal.confidence_score
                })
            
            # Calculate pairwise correlations
            channels = list(channel_signals.keys())
            correlation_matrix = {}
            
            for i, channel1 in enumerate(channels):
                correlation_matrix[channel1] = {}
                
                for j, channel2 in enumerate(channels):
                    if i == j:
                        correlation_matrix[channel1][channel2] = 1.0
                    else:
                        correlation = self._calculate_channel_correlation(
                            channel_signals[channel1],
                            channel_signals[channel2]
                        )
                        correlation_matrix[channel1][channel2] = correlation
            
            # Calculate statistics
            all_correlations = []
            for channel1 in channels:
                for channel2 in channels:
                    if channel1 != channel2:
                        all_correlations.append(correlation_matrix[channel1][channel2])
            
            return {
                'correlation_matrix': correlation_matrix,
                'channels': channels,
                'avg_correlation': np.mean(all_correlations) if all_correlations else 0,
                'max_correlation': max(all_correlations) if all_correlations else 0,
                'min_correlation': min(all_correlations) if all_correlations else 0
            }
    
    def _calculate_channel_correlation(self,
                                      signals1: List[Dict[str, Any]],
                                      signals2: List[Dict[str, Any]]) -> float:
        """Calculate correlation between two channels' signals."""
        if not signals1 or not signals2:
            return 0
        
        # Find matching signals (same pair, similar time)
        matches = 0
        total_comparisons = 0
        
        for s1 in signals1:
            for s2 in signals2:
                if s1['pair'] == s2['pair']:
                    time_diff = abs((s1['timestamp'] - s2['timestamp']).total_seconds() / 60)
                    
                    if time_diff <= self.time_window_minutes:
                        total_comparisons += 1
                        
                        # Check if signals agree
                        if s1['direction'] == s2['direction']:
                            # Weight by confidence
                            weight = (s1['confidence'] + s2['confidence']) / 200
                            matches += weight
        
        if total_comparisons > 0:
            return round(matches / total_comparisons, 4)
        
        return 0
