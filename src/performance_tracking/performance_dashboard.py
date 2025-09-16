"""
Performance Dashboard Backend

Provides comprehensive performance metrics and analytics for the dashboard UI,
aggregating data from all performance tracking modules.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import uuid
import logging
import json
from collections import defaultdict

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.config.settings import get_settings

from .signal_tracker import SignalOutcomeTracker
from .pnl_calculator import PnLCalculator
from .slippage_analyzer import SlippageAnalyzer
from .execution_monitor import ExecutionMonitor

logger = logging.getLogger(__name__)
settings = get_settings()


class PerformanceDashboard:
    """
    Aggregates and serves performance metrics for the dashboard
    """
    
    def __init__(self):
        self.redis = get_redis()
        self.signal_tracker = SignalOutcomeTracker()
        self.pnl_calculator = PnLCalculator()
        self.slippage_analyzer = SlippageAnalyzer()
        self.execution_monitor = ExecutionMonitor()
        
        self.cache_ttl = 300  # 5 minutes cache
    
    async def get_dashboard_summary(self, 
                                   time_range: str = "24h") -> Dict[str, Any]:
        """
        Get comprehensive dashboard summary
        
        Args:
            time_range: Time range (24h, 7d, 30d, all)
            
        Returns:
            Dict with dashboard metrics
        """
        try:
            # Check cache first
            cache_key = f"dashboard_summary:{time_range}"
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = self._get_start_time(time_range, end_time)
            
            # Get signals in range
            signal_ids = await self._get_signals_in_range(start_time, end_time)
            
            # Aggregate metrics
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'time_range': time_range,
                'overview': await self._get_overview_metrics(signal_ids),
                'pnl_metrics': await self._get_pnl_summary(signal_ids),
                'execution_metrics': await self._get_execution_summary(start_time, end_time),
                'slippage_metrics': await self._get_slippage_summary(signal_ids),
                'performance_trends': await self._get_performance_trends(signal_ids, start_time, end_time),
                'top_performers': await self._get_top_performers(signal_ids),
                'risk_metrics': await self._get_risk_metrics(signal_ids),
                'alerts': await self._get_active_alerts()
            }
            
            # Cache the summary
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(summary, default=str)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {str(e)}")
            return {}
    
    async def get_signal_performance_details(self, signal_id: str) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a specific signal
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Dict with detailed performance data
        """
        try:
            # Get tracking status
            tracking_status = await self._get_tracking_status(signal_id)
            
            # Get P&L metrics
            pnl_metrics = await self.pnl_calculator.calculate_signal_pnl(signal_id)
            
            # Get slippage analysis
            slippage_metrics = await self.slippage_analyzer.analyze_signal_slippage(signal_id)
            
            # Get execution metrics
            execution_history = self.execution_monitor.execution_history.get(signal_id, [])
            
            # Get signal data
            async with get_async_session() as session:
                signal = await self._get_signal(session, signal_id)
                performance = await self._get_performance(session, signal_id)
            
            return {
                'signal_id': signal_id,
                'signal_data': {
                    'pair': signal.pair if signal else None,
                    'direction': signal.direction if signal else None,
                    'entry_price': float(signal.entry_price) if signal and signal.entry_price else None,
                    'stop_loss': float(signal.stop_loss) if signal and signal.stop_loss else None,
                    'status': signal.status if signal else None
                },
                'tracking': tracking_status,
                'pnl': {
                    'gross_pnl': str(pnl_metrics.gross_pnl) if pnl_metrics else "0",
                    'net_pnl': str(pnl_metrics.net_pnl) if pnl_metrics else "0",
                    'roi_percentage': str(pnl_metrics.roi_percentage) if pnl_metrics else "0",
                    'sharpe_ratio': pnl_metrics.sharpe_ratio if pnl_metrics else None,
                    'max_drawdown': str(pnl_metrics.max_drawdown) if pnl_metrics else "0"
                },
                'slippage': {
                    'slippage_percentage': str(slippage_metrics.slippage_percentage) if slippage_metrics else "0",
                    'severity': slippage_metrics.severity.value if slippage_metrics else None,
                    'cost_impact': str(slippage_metrics.cost_impact) if slippage_metrics else "0"
                },
                'execution': {
                    'total_orders': len(execution_history),
                    'average_latency': np.mean([e.latency_ms for e in execution_history 
                                               if e.latency_ms]) if execution_history else 0,
                    'execution_quality': self._get_execution_quality_summary(execution_history)
                },
                'performance': {
                    'actual_entry': float(performance.actual_entry) if performance and performance.actual_entry else None,
                    'actual_exit': float(performance.actual_exit) if performance and performance.actual_exit else None,
                    'pnl_percentage': float(performance.pnl_percentage) if performance and performance.pnl_percentage else None,
                    'duration_hours': performance.duration_hours if performance else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting signal performance details: {str(e)}")
            return {}
    
    async def get_portfolio_analytics(self, 
                                     portfolio_ids: List[str] = None,
                                     time_range: str = "30d") -> Dict[str, Any]:
        """
        Get portfolio-level analytics
        
        Args:
            portfolio_ids: List of portfolio identifiers
            time_range: Time range for analysis
            
        Returns:
            Dict with portfolio analytics
        """
        try:
            end_time = datetime.utcnow()
            start_time = self._get_start_time(time_range, end_time)
            
            if not portfolio_ids:
                # Get all signals as default portfolio
                portfolio_ids = await self._get_signals_in_range(start_time, end_time)
            
            # Calculate portfolio P&L
            portfolio_pnl = await self.pnl_calculator.calculate_portfolio_pnl(
                portfolio_ids, start_time, end_time
            )
            
            # Calculate portfolio slippage
            portfolio_slippage = await self.slippage_analyzer.analyze_portfolio_slippage(
                portfolio_ids, start_time, end_time
            )
            
            # Get execution report
            execution_report = await self.execution_monitor.get_execution_report(
                start_time, end_time
            )
            
            return {
                'time_range': time_range,
                'portfolio_size': len(portfolio_ids),
                'total_value': str(portfolio_pnl.get('total_net_pnl', 0)),
                'total_return': str(portfolio_pnl.get('total_net_pnl', 0)),
                'win_rate': portfolio_pnl.get('overall_win_rate', 0),
                'profit_factor': portfolio_pnl.get('overall_profit_factor', 0),
                'sharpe_ratio': portfolio_pnl.get('portfolio_sharpe', 0),
                'sortino_ratio': portfolio_pnl.get('portfolio_sortino', 0),
                'max_drawdown': str(portfolio_pnl.get('max_portfolio_drawdown', 0)),
                'average_slippage': str(portfolio_slippage.average_slippage_pct),
                'execution_rate': execution_report.execution_rate,
                'average_latency': execution_report.average_latency_ms,
                'best_performer': portfolio_pnl.get('best_trade', {}),
                'worst_performer': portfolio_pnl.get('worst_trade', {}),
                'recommendations': self._merge_recommendations(
                    portfolio_slippage.recommendations,
                    execution_report.recommendations
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {str(e)}")
            return {}
    
    # Private Methods
    async def _get_overview_metrics(self, signal_ids: List[str]) -> Dict[str, Any]:
        """Get overview metrics"""
        try:
            async with get_async_session() as session:
                # Count signals by status
                total_signals = len(signal_ids)
                
                # Get performance metrics
                result = await session.execute(
                    select(
                        func.count(SignalPerformance.id).label('total'),
                        func.sum(SignalPerformance.pnl_amount).label('total_pnl'),
                        func.avg(SignalPerformance.pnl_percentage).label('avg_pnl_pct')
                    ).where(
                        SignalPerformance.signal_id.in_([uuid.UUID(sid) for sid in signal_ids])
                    )
                )
                perf_data = result.first()
                
                return {
                    'total_signals': total_signals,
                    'active_signals': await self._count_active_signals(session, signal_ids),
                    'total_pnl': str(perf_data.total_pnl or 0),
                    'average_pnl_pct': float(perf_data.avg_pnl_pct or 0),
                    'signals_tracking': len(self.signal_tracker._tracking_tasks)
                }
                
        except Exception as e:
            logger.error(f"Error getting overview metrics: {str(e)}")
            return {}
    
    async def _get_pnl_summary(self, signal_ids: List[str]) -> Dict[str, Any]:
        """Get P&L summary"""
        try:
            if not signal_ids:
                return {}
            
            # Calculate portfolio P&L
            portfolio_pnl = await self.pnl_calculator.calculate_portfolio_pnl(signal_ids)
            
            return {
                'total_gross_pnl': str(portfolio_pnl.get('total_gross_pnl', 0)),
                'total_net_pnl': str(portfolio_pnl.get('total_net_pnl', 0)),
                'total_fees': str(portfolio_pnl.get('total_fees', 0)),
                'total_realized': str(portfolio_pnl.get('total_realized', 0)),
                'total_unrealized': str(portfolio_pnl.get('total_unrealized', 0)),
                'winning_trades': portfolio_pnl.get('winning_trades', 0),
                'losing_trades': portfolio_pnl.get('losing_trades', 0),
                'win_rate': portfolio_pnl.get('overall_win_rate', 0),
                'profit_factor': portfolio_pnl.get('overall_profit_factor', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting P&L summary: {str(e)}")
            return {}
    
    async def _get_execution_summary(self, start_time: datetime, 
                                    end_time: datetime) -> Dict[str, Any]:
        """Get execution summary"""
        try:
            report = await self.execution_monitor.get_execution_report(start_time, end_time)
            
            return {
                'total_orders': report.total_orders,
                'successful_executions': report.successful_executions,
                'failed_executions': report.failed_executions,
                'execution_rate': report.execution_rate,
                'average_latency_ms': report.average_latency_ms,
                'median_latency_ms': report.median_latency_ms,
                'p95_latency_ms': report.p95_latency_ms
            }
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {str(e)}")
            return {}
    
    async def _get_slippage_summary(self, signal_ids: List[str]) -> Dict[str, Any]:
        """Get slippage summary"""
        try:
            if not signal_ids:
                return {}
            
            analysis = await self.slippage_analyzer.analyze_portfolio_slippage(signal_ids)
            
            return {
                'total_slippage': str(analysis.total_slippage),
                'average_slippage_pct': str(analysis.average_slippage_pct),
                'median_slippage_pct': str(analysis.median_slippage_pct),
                'max_slippage': str(analysis.max_slippage),
                'entry_slippage': str(analysis.entry_slippage),
                'exit_slippage': str(analysis.exit_slippage),
                'patterns': analysis.patterns_identified[:3],  # Top 3 patterns
                'estimated_savings': str(analysis.estimated_savings)
            }
            
        except Exception as e:
            logger.error(f"Error getting slippage summary: {str(e)}")
            return {}
    
    async def _get_performance_trends(self, signal_ids: List[str],
                                     start_time: datetime,
                                     end_time: datetime) -> Dict[str, Any]:
        """Get performance trends over time"""
        try:
            # Group by day
            days = (end_time - start_time).days
            daily_metrics = defaultdict(lambda: {'pnl': 0, 'count': 0, 'win_rate': 0})
            
            async with get_async_session() as session:
                for signal_id in signal_ids:
                    perf = await self._get_performance(session, signal_id)
                    if perf and perf.closed_at:
                        day_key = perf.closed_at.date().isoformat()
                        daily_metrics[day_key]['pnl'] += float(perf.pnl_amount or 0)
                        daily_metrics[day_key]['count'] += 1
                        if perf.pnl_amount and perf.pnl_amount > 0:
                            daily_metrics[day_key]['win_rate'] += 1
            
            # Calculate win rates
            for day, metrics in daily_metrics.items():
                if metrics['count'] > 0:
                    metrics['win_rate'] = metrics['win_rate'] / metrics['count']
            
            return {
                'daily_pnl': dict(daily_metrics),
                'trend_direction': self._calculate_trend_direction(daily_metrics),
                'volatility': self._calculate_volatility(daily_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {str(e)}")
            return {}
    
    async def _get_top_performers(self, signal_ids: List[str]) -> List[Dict[str, Any]]:
        """Get top performing signals"""
        try:
            performers = []
            
            for signal_id in signal_ids[:100]:  # Limit to 100 for performance
                pnl_metrics = await self.pnl_calculator.calculate_signal_pnl(signal_id)
                if pnl_metrics:
                    performers.append({
                        'signal_id': signal_id,
                        'net_pnl': float(pnl_metrics.net_pnl),
                        'roi_percentage': float(pnl_metrics.roi_percentage)
                    })
            
            # Sort by net P&L
            performers.sort(key=lambda x: x['net_pnl'], reverse=True)
            
            return performers[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Error getting top performers: {str(e)}")
            return []
    
    async def _get_risk_metrics(self, signal_ids: List[str]) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            risk_metrics = {
                'var_95': 0,
                'cvar_95': 0,
                'max_position_size': 0,
                'average_position_size': 0,
                'risk_score': 0
            }
            
            if signal_ids:
                # Get risk-adjusted returns for first signal (as example)
                risk_data = await self.pnl_calculator.calculate_risk_adjusted_returns(signal_ids[0])
                
                risk_metrics.update({
                    'var_95': risk_data.get('var_95', 0),
                    'cvar_95': risk_data.get('cvar_95', 0),
                    'risk_score': self._calculate_risk_score(risk_data)
                })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        alerts = []
        
        # Check for high slippage
        recent_slippage = await self.redis.get("portfolio_slippage_analysis")
        if recent_slippage:
            data = json.loads(recent_slippage)
            if float(data.get('average_slippage_pct', 0)) > 2:
                alerts.append({
                    'type': 'warning',
                    'message': 'High slippage detected',
                    'severity': 'medium'
                })
        
        # Check execution quality
        real_time = await self.execution_monitor.get_real_time_metrics()
        if real_time.get('health_status') == 'poor':
            alerts.append({
                'type': 'error',
                'message': 'Poor execution quality',
                'severity': 'high'
            })
        
        # Check for stale tracking
        if len(self.signal_tracker._tracking_tasks) > 100:
            alerts.append({
                'type': 'info',
                'message': f'{len(self.signal_tracker._tracking_tasks)} signals being tracked',
                'severity': 'low'
            })
        
        return alerts
    
    # Helper Methods
    def _get_start_time(self, time_range: str, end_time: datetime) -> datetime:
        """Calculate start time based on range"""
        if time_range == "24h":
            return end_time - timedelta(days=1)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        else:  # all
            return end_time - timedelta(days=365)
    
    async def _get_signals_in_range(self, start_time: datetime, 
                                   end_time: datetime) -> List[str]:
        """Get signals within time range"""
        async with get_async_session() as session:
            result = await session.execute(
                select(Signal.id)
                .where(and_(
                    Signal.created_at >= start_time,
                    Signal.created_at <= end_time
                ))
            )
            return [str(row[0]) for row in result]
    
    async def _get_signal(self, session: AsyncSession, signal_id: str):
        """Get signal from database"""
        result = await session.execute(
            select(Signal).where(Signal.id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _get_performance(self, session: AsyncSession, signal_id: str):
        """Get performance record"""
        result = await session.execute(
            select(SignalPerformance).where(SignalPerformance.signal_id == uuid.UUID(signal_id))
        )
        return result.scalar_one_or_none()
    
    async def _count_active_signals(self, session: AsyncSession, signal_ids: List[str]) -> int:
        """Count active signals"""
        result = await session.execute(
            select(func.count(Signal.id))
            .where(and_(
                Signal.id.in_([uuid.UUID(sid) for sid in signal_ids]),
                Signal.status == 'active'
            ))
        )
        return result.scalar() or 0
    
    async def _get_tracking_status(self, signal_id: str) -> Dict[str, Any]:
        """Get tracking status for a signal"""
        metadata = await self.signal_tracker._load_metadata(signal_id)
        
        if metadata:
            return {
                'is_tracking': True,
                'start_time': metadata.start_time.isoformat(),
                'last_check': metadata.last_check.isoformat() if metadata.last_check else None,
                'entry_attempts': metadata.entry_attempts
            }
        
        return {'is_tracking': False}
    
    def _get_execution_quality_summary(self, execution_history: List) -> Dict[str, int]:
        """Summarize execution quality"""
        summary = defaultdict(int)
        for execution in execution_history:
            summary[execution.execution_quality.value] += 1
        return dict(summary)
    
    def _merge_recommendations(self, *recommendation_lists) -> List[str]:
        """Merge and deduplicate recommendations"""
        all_recommendations = []
        seen = set()
        
        for rec_list in recommendation_lists:
            for rec in rec_list:
                if rec not in seen:
                    all_recommendations.append(rec)
                    seen.add(rec)
        
        return all_recommendations[:10]  # Top 10 recommendations
    
    def _calculate_trend_direction(self, daily_metrics: Dict) -> str:
        """Calculate trend direction from daily metrics"""
        if not daily_metrics:
            return "neutral"
        
        values = [m['pnl'] for m in daily_metrics.values()]
        if len(values) < 2:
            return "neutral"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "upward"
        elif slope < -0.1:
            return "downward"
        else:
            return "neutral"
    
    def _calculate_volatility(self, daily_metrics: Dict) -> float:
        """Calculate volatility from daily metrics"""
        if not daily_metrics:
            return 0
        
        values = [m['pnl'] for m in daily_metrics.values()]
        if len(values) < 2:
            return 0
        
        return float(np.std(values))
    
    def _calculate_risk_score(self, risk_data: Dict) -> float:
        """Calculate overall risk score"""
        # Simple risk scoring based on metrics
        score = 0
        
        var_95 = risk_data.get('var_95', 0)
        if var_95 < -10:
            score += 3
        elif var_95 < -5:
            score += 2
        elif var_95 < -2:
            score += 1
        
        max_dd = risk_data.get('max_drawdown', 0)
        if max_dd > 20:
            score += 3
        elif max_dd > 10:
            score += 2
        elif max_dd > 5:
            score += 1
        
        return min(score / 6, 1.0)  # Normalize to 0-1
