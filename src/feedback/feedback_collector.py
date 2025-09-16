"""
Feedback Collection System for user signal feedback.

This module handles collection, categorization, and analysis of user feedback
on trading signals to improve system performance.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import json

from pydantic import BaseModel, Field, validator
import numpy as np
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.kafka_client import KafkaClient
from src.models import Signal, UserFeedback, ChannelStatistics
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    ACCURACY = "accuracy"          # Signal accuracy feedback
    TIMING = "timing"              # Entry/exit timing feedback
    RISK_LEVEL = "risk_level"      # Risk assessment feedback
    PROFIT_LOSS = "profit_loss"    # P&L outcome feedback
    EXECUTION = "execution"         # Execution quality feedback
    ANALYSIS = "analysis"          # AI analysis quality
    GENERAL = "general"            # General comments


class FeedbackSentiment(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FeedbackRating(int, Enum):
    """Numeric rating scale."""
    VERY_POOR = 1
    POOR = 2
    AVERAGE = 3
    GOOD = 4
    EXCELLENT = 5


class FeedbackPriority(str, Enum):
    """Feedback priority for processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserFeedbackRequest(BaseModel):
    """User feedback submission model."""
    signal_id: UUID
    user_id: str
    feedback_type: FeedbackType
    rating: Optional[FeedbackRating] = None
    comment: Optional[str] = Field(None, max_length=1000)
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    actual_pnl: Optional[float] = None
    execution_issues: Optional[List[str]] = None
    improvement_suggestions: Optional[str] = None
    would_follow_again: Optional[bool] = None
    
    @validator('comment')
    def clean_comment(cls, v):
        """Clean and validate comment."""
        if v:
            # Remove excessive whitespace
            v = ' '.join(v.split())
            # Basic sanitization
            v = v.replace('<', '').replace('>', '')
        return v


class FeedbackAnalysis(BaseModel):
    """Feedback analysis results."""
    signal_id: UUID
    total_feedback: int
    average_rating: float
    sentiment_distribution: Dict[str, float]
    common_issues: List[str]
    improvement_areas: List[str]
    follow_again_rate: float
    accuracy_score: float
    timing_score: float
    risk_score: float


class FeedbackCollector:
    """Manages feedback collection and processing."""
    
    def __init__(self):
        self.redis = None
        self.kafka = None
        self.cache_ttl = 3600  # 1 hour
        self.sentiment_analyzer = SentimentAnalyzer()
        self.categorizer = FeedbackCategorizer()
        self.aggregator = FeedbackAggregator()
        
    async def initialize(self):
        """Initialize connections."""
        self.redis = await get_redis()
        self.kafka = KafkaClient()
        await self.kafka.initialize()
        logger.info("Feedback collector initialized")
        
    async def submit_feedback(
        self,
        feedback: UserFeedbackRequest
    ) -> Dict[str, Any]:
        """
        Submit user feedback for a signal.
        
        Args:
            feedback: User feedback request
            
        Returns:
            Submission confirmation and analysis
        """
        try:
            # Analyze sentiment
            sentiment = await self.sentiment_analyzer.analyze(feedback.comment)
            
            # Categorize feedback
            categories = await self.categorizer.categorize(feedback)
            
            # Determine priority
            priority = self._determine_priority(feedback, sentiment)
            
            # Store in database
            async with get_async_session() as session:
                db_feedback = UserFeedback(
                    signal_id=feedback.signal_id,
                    user_id=feedback.user_id,
                    feedback_type=feedback.feedback_type.value,
                    rating=feedback.rating.value if feedback.rating else None,
                    comment=feedback.comment,
                    sentiment=sentiment.value,
                    categories=categories,
                    priority=priority.value,
                    actual_entry=feedback.actual_entry,
                    actual_exit=feedback.actual_exit,
                    actual_pnl=feedback.actual_pnl,
                    execution_issues=feedback.execution_issues,
                    improvement_suggestions=feedback.improvement_suggestions,
                    would_follow_again=feedback.would_follow_again,
                    created_at=datetime.utcnow()
                )
                
                session.add(db_feedback)
                await session.commit()
                
                feedback_id = db_feedback.id
            
            # Publish to Kafka for real-time processing
            await self.kafka.publish(
                'feedback-events',
                {
                    'feedback_id': str(feedback_id),
                    'signal_id': str(feedback.signal_id),
                    'user_id': feedback.user_id,
                    'type': feedback.feedback_type.value,
                    'sentiment': sentiment.value,
                    'priority': priority.value,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Update signal statistics
            await self._update_signal_stats(feedback.signal_id, feedback)
            
            # Trigger improvement pipeline for critical feedback
            if priority == FeedbackPriority.CRITICAL:
                await self._trigger_improvement_pipeline(feedback, sentiment)
            
            # Cache feedback summary
            await self._cache_feedback_summary(feedback.signal_id)
            
            return {
                'feedback_id': str(feedback_id),
                'status': 'submitted',
                'sentiment': sentiment.value,
                'priority': priority.value,
                'categories': categories,
                'message': 'Thank you for your feedback!'
            }
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            raise
    
    async def get_signal_feedback(
        self,
        signal_id: UUID,
        limit: int = 50
    ) -> FeedbackAnalysis:
        """
        Get aggregated feedback for a signal.
        
        Args:
            signal_id: Signal identifier
            limit: Maximum feedback entries to analyze
            
        Returns:
            Feedback analysis results
        """
        try:
            # Check cache first
            cache_key = f"feedback_analysis:{signal_id}"
            cached = await self.redis.get(cache_key)
            if cached:
                return FeedbackAnalysis(**json.loads(cached))
            
            async with get_async_session() as session:
                # Fetch feedback
                result = await session.execute(
                    select(UserFeedback)
                    .where(UserFeedback.signal_id == signal_id)
                    .order_by(UserFeedback.created_at.desc())
                    .limit(limit)
                )
                feedback_entries = result.scalars().all()
                
                if not feedback_entries:
                    return FeedbackAnalysis(
                        signal_id=signal_id,
                        total_feedback=0,
                        average_rating=0.0,
                        sentiment_distribution={},
                        common_issues=[],
                        improvement_areas=[],
                        follow_again_rate=0.0,
                        accuracy_score=0.0,
                        timing_score=0.0,
                        risk_score=0.0
                    )
                
                # Aggregate feedback
                analysis = await self.aggregator.aggregate(feedback_entries)
                
                # Cache results
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(analysis.dict())
                )
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error getting signal feedback: {e}")
            raise
    
    async def get_feedback_report(
        self,
        start_date: datetime,
        end_date: datetime,
        channel_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feedback report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            channel_id: Optional channel filter
            
        Returns:
            Feedback report with insights
        """
        try:
            async with get_async_session() as session:
                # Build query
                query = select(UserFeedback).where(
                    and_(
                        UserFeedback.created_at >= start_date,
                        UserFeedback.created_at <= end_date
                    )
                )
                
                if channel_id:
                    # Join with signals to filter by channel
                    query = query.join(Signal).where(
                        Signal.source_channel_id == channel_id
                    )
                
                result = await session.execute(query)
                feedback_entries = result.scalars().all()
                
                # Generate report
                report = {
                    'period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'total_feedback': len(feedback_entries),
                    'unique_users': len(set(f.user_id for f in feedback_entries)),
                    'feedback_by_type': self._count_by_type(feedback_entries),
                    'sentiment_analysis': self._analyze_sentiment_distribution(feedback_entries),
                    'rating_distribution': self._rating_distribution(feedback_entries),
                    'common_issues': self._extract_common_issues(feedback_entries),
                    'improvement_trends': self._analyze_improvement_trends(feedback_entries),
                    'accuracy_metrics': self._calculate_accuracy_metrics(feedback_entries),
                    'user_satisfaction': self._calculate_satisfaction(feedback_entries),
                    'actionable_insights': self._generate_insights(feedback_entries)
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating feedback report: {e}")
            raise
    
    def _determine_priority(
        self,
        feedback: UserFeedbackRequest,
        sentiment: FeedbackSentiment
    ) -> FeedbackPriority:
        """Determine feedback priority based on content."""
        # Critical if significant loss or very negative
        if feedback.actual_pnl and feedback.actual_pnl < -1000:
            return FeedbackPriority.CRITICAL
        
        if feedback.rating == FeedbackRating.VERY_POOR:
            return FeedbackPriority.CRITICAL
        
        if sentiment == FeedbackSentiment.NEGATIVE and feedback.rating and feedback.rating <= 2:
            return FeedbackPriority.HIGH
        
        if feedback.execution_issues and len(feedback.execution_issues) > 2:
            return FeedbackPriority.HIGH
        
        if sentiment == FeedbackSentiment.POSITIVE and feedback.rating and feedback.rating >= 4:
            return FeedbackPriority.LOW
        
        return FeedbackPriority.MEDIUM
    
    async def _update_signal_stats(
        self,
        signal_id: UUID,
        feedback: UserFeedbackRequest
    ):
        """Update signal statistics based on feedback."""
        try:
            async with get_async_session() as session:
                # Update signal feedback count and average rating
                signal = await session.get(Signal, signal_id)
                if signal:
                    current_feedback_count = signal.feedback_count or 0
                    current_avg_rating = signal.average_rating or 0
                    
                    if feedback.rating:
                        # Update running average
                        new_count = current_feedback_count + 1
                        new_avg = ((current_avg_rating * current_feedback_count) + 
                                  feedback.rating.value) / new_count
                        
                        signal.feedback_count = new_count
                        signal.average_rating = new_avg
                    
                    if feedback.would_follow_again is not None:
                        current_follow_count = signal.follow_again_count or 0
                        if feedback.would_follow_again:
                            signal.follow_again_count = current_follow_count + 1
                    
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error updating signal stats: {e}")
    
    async def _trigger_improvement_pipeline(
        self,
        feedback: UserFeedbackRequest,
        sentiment: FeedbackSentiment
    ):
        """Trigger improvement pipeline for critical feedback."""
        try:
            # Publish to improvement topic
            await self.kafka.publish(
                'improvement-pipeline',
                {
                    'signal_id': str(feedback.signal_id),
                    'feedback_type': feedback.feedback_type.value,
                    'sentiment': sentiment.value,
                    'issues': feedback.execution_issues,
                    'suggestions': feedback.improvement_suggestions,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Triggered improvement pipeline for signal {feedback.signal_id}")
            
        except Exception as e:
            logger.error(f"Error triggering improvement pipeline: {e}")
    
    async def _cache_feedback_summary(self, signal_id: UUID):
        """Cache updated feedback summary."""
        try:
            # Invalidate existing cache
            cache_key = f"feedback_analysis:{signal_id}"
            await self.redis.delete(cache_key)
            
            # Pre-compute and cache new analysis
            analysis = await self.get_signal_feedback(signal_id)
            
        except Exception as e:
            logger.error(f"Error caching feedback summary: {e}")
    
    def _count_by_type(self, feedback_entries: List[UserFeedback]) -> Dict[str, int]:
        """Count feedback by type."""
        type_counts = {}
        for entry in feedback_entries:
            type_counts[entry.feedback_type] = type_counts.get(entry.feedback_type, 0) + 1
        return type_counts
    
    def _analyze_sentiment_distribution(
        self,
        feedback_entries: List[UserFeedback]
    ) -> Dict[str, float]:
        """Analyze sentiment distribution."""
        if not feedback_entries:
            return {}
        
        sentiment_counts = {}
        for entry in feedback_entries:
            sentiment_counts[entry.sentiment] = sentiment_counts.get(entry.sentiment, 0) + 1
        
        total = len(feedback_entries)
        return {k: v/total for k, v in sentiment_counts.items()}
    
    def _rating_distribution(self, feedback_entries: List[UserFeedback]) -> Dict[int, int]:
        """Get rating distribution."""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for entry in feedback_entries:
            if entry.rating:
                distribution[entry.rating] += 1
        return distribution
    
    def _extract_common_issues(self, feedback_entries: List[UserFeedback]) -> List[str]:
        """Extract common issues from feedback."""
        all_issues = []
        for entry in feedback_entries:
            if entry.execution_issues:
                all_issues.extend(entry.execution_issues)
        
        # Count occurrences and get top 10
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, _ in sorted_issues[:10]]
    
    def _analyze_improvement_trends(
        self,
        feedback_entries: List[UserFeedback]
    ) -> Dict[str, Any]:
        """Analyze improvement trends over time."""
        if not feedback_entries:
            return {}
        
        # Sort by date
        sorted_entries = sorted(feedback_entries, key=lambda x: x.created_at)
        
        # Calculate moving average of ratings
        ratings_over_time = []
        for entry in sorted_entries:
            if entry.rating:
                ratings_over_time.append({
                    'date': entry.created_at.isoformat(),
                    'rating': entry.rating
                })
        
        # Detect trend (improving, declining, stable)
        if len(ratings_over_time) >= 2:
            first_half_avg = np.mean([r['rating'] for r in ratings_over_time[:len(ratings_over_time)//2]])
            second_half_avg = np.mean([r['rating'] for r in ratings_over_time[len(ratings_over_time)//2:]])
            
            if second_half_avg > first_half_avg + 0.2:
                trend = "improving"
            elif second_half_avg < first_half_avg - 0.2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            'trend': trend,
            'ratings_timeline': ratings_over_time[-20:],  # Last 20 entries
            'current_avg': np.mean([r['rating'] for r in ratings_over_time[-10:]]) if len(ratings_over_time) >= 10 else None
        }
    
    def _calculate_accuracy_metrics(
        self,
        feedback_entries: List[UserFeedback]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics from feedback."""
        accuracy_feedback = [
            f for f in feedback_entries
            if f.feedback_type == FeedbackType.ACCURACY.value and f.rating
        ]
        
        if not accuracy_feedback:
            return {'accuracy_score': 0.0, 'sample_size': 0}
        
        avg_accuracy = np.mean([f.rating for f in accuracy_feedback])
        
        # Calculate hit rate if P&L data available
        with_pnl = [f for f in feedback_entries if f.actual_pnl is not None]
        hit_rate = len([f for f in with_pnl if f.actual_pnl > 0]) / len(with_pnl) if with_pnl else 0
        
        return {
            'accuracy_score': avg_accuracy / 5.0,  # Normalize to 0-1
            'hit_rate': hit_rate,
            'sample_size': len(accuracy_feedback)
        }
    
    def _calculate_satisfaction(
        self,
        feedback_entries: List[UserFeedback]
    ) -> Dict[str, float]:
        """Calculate user satisfaction metrics."""
        with_rating = [f for f in feedback_entries if f.rating]
        with_follow = [f for f in feedback_entries if f.would_follow_again is not None]
        
        avg_rating = np.mean([f.rating for f in with_rating]) if with_rating else 0
        follow_rate = len([f for f in with_follow if f.would_follow_again]) / len(with_follow) if with_follow else 0
        
        # Calculate Net Promoter Score (NPS)
        promoters = len([f for f in with_rating if f.rating >= 4])
        detractors = len([f for f in with_rating if f.rating <= 2])
        nps = ((promoters - detractors) / len(with_rating) * 100) if with_rating else 0
        
        return {
            'average_rating': avg_rating,
            'follow_again_rate': follow_rate,
            'net_promoter_score': nps,
            'satisfaction_index': (avg_rating / 5.0 * 0.5 + follow_rate * 0.5)  # Combined index
        }
    
    def _generate_insights(self, feedback_entries: List[UserFeedback]) -> List[str]:
        """Generate actionable insights from feedback."""
        insights = []
        
        # Check for consistent issues
        common_issues = self._extract_common_issues(feedback_entries)
        if common_issues:
            insights.append(f"Most common issue: {common_issues[0]} - Consider prioritizing fix")
        
        # Check satisfaction trend
        satisfaction = self._calculate_satisfaction(feedback_entries)
        if satisfaction['net_promoter_score'] < 0:
            insights.append("Negative NPS score - Urgent improvement needed")
        elif satisfaction['net_promoter_score'] > 50:
            insights.append("Excellent NPS score - Maintain current performance")
        
        # Check accuracy
        accuracy = self._calculate_accuracy_metrics(feedback_entries)
        if accuracy['accuracy_score'] < 0.6:
            insights.append("Low accuracy score - Review signal generation logic")
        
        # Check for timing issues
        timing_feedback = [
            f for f in feedback_entries
            if f.feedback_type == FeedbackType.TIMING.value and f.rating and f.rating <= 2
        ]
        if len(timing_feedback) / len(feedback_entries) > 0.2:
            insights.append("Significant timing issues reported - Consider entry/exit optimization")
        
        return insights


class SentimentAnalyzer:
    """Analyzes sentiment of feedback text."""
    
    async def analyze(self, text: Optional[str]) -> FeedbackSentiment:
        """
        Analyze sentiment of feedback text.
        
        Simple rule-based sentiment analysis.
        For production, integrate with NLP service.
        """
        if not text:
            return FeedbackSentiment.NEUTRAL
        
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'excellent', 'great', 'good', 'amazing', 'perfect',
            'accurate', 'profitable', 'successful', 'helpful', 'useful'
        ]
        
        # Negative indicators
        negative_words = [
            'terrible', 'horrible', 'bad', 'awful', 'poor',
            'inaccurate', 'loss', 'failed', 'useless', 'wrong'
        ]
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            return FeedbackSentiment.POSITIVE
        elif negative_score > positive_score:
            return FeedbackSentiment.NEGATIVE
        elif positive_score > 0 and negative_score > 0:
            return FeedbackSentiment.MIXED
        else:
            return FeedbackSentiment.NEUTRAL


class FeedbackCategorizer:
    """Categorizes feedback into actionable categories."""
    
    async def categorize(self, feedback: UserFeedbackRequest) -> List[str]:
        """
        Categorize feedback into actionable categories.
        
        Returns list of relevant categories.
        """
        categories = []
        
        # Add primary category
        categories.append(feedback.feedback_type.value)
        
        # Analyze comment for additional categories
        if feedback.comment:
            comment_lower = feedback.comment.lower()
            
            if any(word in comment_lower for word in ['entry', 'entrance', 'buy', 'purchase']):
                categories.append('entry_timing')
            
            if any(word in comment_lower for word in ['exit', 'sell', 'close', 'take profit']):
                categories.append('exit_timing')
            
            if any(word in comment_lower for word in ['risk', 'position', 'size', 'leverage']):
                categories.append('risk_management')
            
            if any(word in comment_lower for word in ['slippage', 'spread', 'execution', 'fill']):
                categories.append('execution_quality')
            
            if any(word in comment_lower for word in ['analysis', 'explanation', 'reasoning']):
                categories.append('analysis_quality')
        
        # Add performance category if P&L provided
        if feedback.actual_pnl is not None:
            if feedback.actual_pnl > 0:
                categories.append('profitable_trade')
            else:
                categories.append('losing_trade')
        
        return list(set(categories))  # Remove duplicates


class FeedbackAggregator:
    """Aggregates feedback for analysis."""
    
    async def aggregate(self, feedback_entries: List[UserFeedback]) -> FeedbackAnalysis:
        """
        Aggregate feedback entries into analysis.
        
        Args:
            feedback_entries: List of feedback entries
            
        Returns:
            Aggregated analysis
        """
        if not feedback_entries:
            raise ValueError("No feedback entries to aggregate")
        
        signal_id = feedback_entries[0].signal_id
        
        # Calculate average rating
        ratings = [f.rating for f in feedback_entries if f.rating]
        avg_rating = np.mean(ratings) if ratings else 0.0
        
        # Sentiment distribution
        sentiment_counts = {}
        for entry in feedback_entries:
            sentiment_counts[entry.sentiment] = sentiment_counts.get(entry.sentiment, 0) + 1
        
        total = len(feedback_entries)
        sentiment_dist = {k: v/total for k, v in sentiment_counts.items()}
        
        # Common issues
        all_issues = []
        for entry in feedback_entries:
            if entry.execution_issues:
                all_issues.extend(entry.execution_issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:5]
        
        # Improvement areas from suggestions
        improvement_areas = []
        for entry in feedback_entries:
            if entry.improvement_suggestions:
                # Simple extraction - in production, use NLP
                improvement_areas.append(entry.improvement_suggestions[:50])
        
        improvement_areas = list(set(improvement_areas))[:5]
        
        # Follow again rate
        follow_data = [f.would_follow_again for f in feedback_entries if f.would_follow_again is not None]
        follow_rate = len([f for f in follow_data if f]) / len(follow_data) if follow_data else 0.0
        
        # Category-specific scores
        accuracy_ratings = [
            f.rating for f in feedback_entries
            if f.feedback_type == FeedbackType.ACCURACY.value and f.rating
        ]
        accuracy_score = np.mean(accuracy_ratings) / 5.0 if accuracy_ratings else 0.0
        
        timing_ratings = [
            f.rating for f in feedback_entries
            if f.feedback_type == FeedbackType.TIMING.value and f.rating
        ]
        timing_score = np.mean(timing_ratings) / 5.0 if timing_ratings else 0.0
        
        risk_ratings = [
            f.rating for f in feedback_entries
            if f.feedback_type == FeedbackType.RISK_LEVEL.value and f.rating
        ]
        risk_score = np.mean(risk_ratings) / 5.0 if risk_ratings else 0.0
        
        return FeedbackAnalysis(
            signal_id=signal_id,
            total_feedback=total,
            average_rating=avg_rating,
            sentiment_distribution=sentiment_dist,
            common_issues=common_issues,
            improvement_areas=improvement_areas,
            follow_again_rate=follow_rate,
            accuracy_score=accuracy_score,
            timing_score=timing_score,
            risk_score=risk_score
        )
