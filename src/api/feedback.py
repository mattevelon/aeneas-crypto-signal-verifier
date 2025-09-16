"""
Feedback API endpoints.

Handles user feedback submission and retrieval.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from src.feedback.feedback_collector import (
    FeedbackCollector,
    UserFeedbackRequest,
    FeedbackAnalysis
)
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])
feedback_collector = FeedbackCollector()


@router.on_event("startup")
async def startup():
    """Initialize feedback collector on startup."""
    await feedback_collector.initialize()


@router.post("/submit")
async def submit_feedback(
    feedback: UserFeedbackRequest
) -> Dict[str, Any]:
    """
    Submit feedback for a signal.
    
    Args:
        feedback: User feedback request
        
    Returns:
        Submission confirmation
    """
    try:
        result = await feedback_collector.submit_feedback(feedback)
        return result
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal/{signal_id}")
async def get_signal_feedback(
    signal_id: UUID,
    limit: int = Query(50, ge=1, le=100)
) -> FeedbackAnalysis:
    """
    Get aggregated feedback for a signal.
    
    Args:
        signal_id: Signal identifier
        limit: Maximum feedback entries to analyze
        
    Returns:
        Feedback analysis
    """
    try:
        analysis = await feedback_collector.get_signal_feedback(signal_id, limit)
        return analysis
    except Exception as e:
        logger.error(f"Error getting signal feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_feedback_report(
    start_date: datetime = Query(..., description="Report start date"),
    end_date: datetime = Query(..., description="Report end date"),
    channel_id: Optional[int] = Query(None, description="Optional channel filter")
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
        if end_date < start_date:
            raise HTTPException(400, "End date must be after start date")
        
        if (end_date - start_date).days > 90:
            raise HTTPException(400, "Report period cannot exceed 90 days")
        
        report = await feedback_collector.get_feedback_report(
            start_date,
            end_date,
            channel_id
        )
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating feedback report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/recent")
async def get_recent_insights(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back")
) -> Dict[str, Any]:
    """
    Get recent feedback insights.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Recent insights and trends
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        report = await feedback_collector.get_feedback_report(
            start_date,
            end_date
        )
        
        # Extract key insights
        return {
            'period_hours': hours,
            'total_feedback': report['total_feedback'],
            'unique_users': report['unique_users'],
            'sentiment_summary': report['sentiment_analysis'],
            'satisfaction_metrics': report['user_satisfaction'],
            'top_issues': report['common_issues'][:3],
            'actionable_insights': report['actionable_insights'],
            'improvement_trend': report['improvement_trends']
        }
    except Exception as e:
        logger.error(f"Error getting recent insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
