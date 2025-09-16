"""
Model Performance Monitor

Monitors ML model performance in production, detects drift,
and triggers retraining when necessary.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error
)

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Signal, SignalPerformance
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.kafka_client import KafkaClient
from src.config.settings import get_settings
from .model_versioning import ModelVersionManager

logger = logging.getLogger(__name__)
settings = get_settings()


class DriftType(str, Enum):
    """Types of drift detected"""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    timestamp: datetime
    model_id: str
    predictions_count: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Drift detection results"""
    detected: bool
    drift_type: Optional[DriftType] = None
    severity: AlertSeverity = AlertSeverity.INFO
    drift_score: float = 0.0
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None


@dataclass
class MonitoringAlert:
    """Monitoring alert"""
    alert_id: str
    timestamp: datetime
    model_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class ModelPerformanceMonitor:
    """
    Monitors model performance in production and detects drift
    """
    
    def __init__(self, version_manager: Optional[ModelVersionManager] = None,
                 kafka_client: Optional[KafkaClient] = None):
        self.redis = get_redis()
        self.version_manager = version_manager or ModelVersionManager()
        self.kafka_client = kafka_client or KafkaClient()
        
        # Performance history
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        # Feature distributions for drift detection
        self.feature_distributions: Dict[str, Dict] = {}
        
        # Active alerts
        self.active_alerts: List[MonitoringAlert] = []
        
        # Monitoring configuration
        self.monitoring_config = {
            'performance_window_hours': 24,
            'drift_check_interval_minutes': 60,
            'min_samples_for_drift': 100,
            'performance_degradation_threshold': 0.1,  # 10% degradation
            'drift_significance_level': 0.05
        }
        
        # Monitoring task
        self._monitoring_task = None
    
    async def start_monitoring(self, model_id: str):
        """
        Start monitoring a model
        
        Args:
            model_id: Model to monitor
        """
        try:
            # Initialize performance history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            # Load baseline feature distribution
            await self._load_baseline_distribution(model_id)
            
            # Start monitoring task if not running
            if not self._monitoring_task:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Started monitoring model {model_id}")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    async def record_prediction(self, model_id: str, 
                               features: Dict[str, Any],
                               prediction: Any,
                               actual: Optional[Any] = None):
        """
        Record a model prediction for monitoring
        
        Args:
            model_id: Model identifier
            features: Input features
            prediction: Model prediction
            actual: Actual outcome (if available)
        """
        try:
            # Store prediction data
            prediction_data = {
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'features': features,
                'prediction': prediction,
                'actual': actual
            }
            
            # Cache in Redis (with TTL)
            await self.redis.lpush(
                f"predictions:{model_id}",
                json.dumps(prediction_data, default=str)
            )
            await self.redis.ltrim(f"predictions:{model_id}", 0, 10000)  # Keep last 10k
            await self.redis.expire(f"predictions:{model_id}", 86400)  # 24 hour TTL
            
            # Check for drift if enough samples
            await self._check_for_drift(model_id, features)
            
            # Update performance metrics if actual outcome available
            if actual is not None:
                await self._update_performance_metrics(model_id, prediction, actual)
                
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")
    
    async def get_performance_metrics(self, model_id: str,
                                     hours: int = 24) -> List[PerformanceMetrics]:
        """
        Get performance metrics for a model
        
        Args:
            model_id: Model identifier
            hours: Hours of history to retrieve
            
        Returns:
            List of performance metrics
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics by time
            if model_id in self.performance_history:
                return [
                    m for m in self.performance_history[model_id]
                    if m.timestamp >= cutoff_time
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return []
    
    async def detect_drift(self, model_id: str) -> DriftDetectionResult:
        """
        Detect drift in model performance or data
        
        Args:
            model_id: Model identifier
            
        Returns:
            DriftDetectionResult
        """
        try:
            # Get recent predictions
            predictions_data = await self._get_recent_predictions(model_id)
            
            if len(predictions_data) < self.monitoring_config['min_samples_for_drift']:
                return DriftDetectionResult(detected=False)
            
            # Extract features and predictions
            features = [p['features'] for p in predictions_data]
            predictions = [p['prediction'] for p in predictions_data]
            
            # Check for data drift
            data_drift = await self._detect_data_drift(model_id, features)
            if data_drift.detected:
                return data_drift
            
            # Check for prediction drift
            prediction_drift = await self._detect_prediction_drift(model_id, predictions)
            if prediction_drift.detected:
                return prediction_drift
            
            # Check for performance degradation
            performance_drift = await self._detect_performance_degradation(model_id)
            if performance_drift.detected:
                return performance_drift
            
            return DriftDetectionResult(detected=False)
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return DriftDetectionResult(detected=False)
    
    async def get_monitoring_dashboard(self, model_id: str) -> Dict[str, Any]:
        """
        Get monitoring dashboard data for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict with dashboard data
        """
        try:
            # Get recent performance metrics
            metrics = await self.get_performance_metrics(model_id, hours=24)
            
            # Get drift detection results
            drift_result = await self.detect_drift(model_id)
            
            # Get active alerts
            model_alerts = [a for a in self.active_alerts if a.model_id == model_id]
            
            # Calculate summary statistics
            summary = {
                'model_id': model_id,
                'monitoring_status': 'active',
                'last_update': datetime.utcnow().isoformat(),
                'performance': {
                    'current': self._calculate_current_performance(metrics),
                    'trend': self._calculate_performance_trend(metrics),
                    'history': [m.__dict__ for m in metrics[-20:]]  # Last 20 points
                },
                'drift': {
                    'detected': drift_result.detected,
                    'type': drift_result.drift_type.value if drift_result.drift_type else None,
                    'severity': drift_result.severity.value,
                    'score': drift_result.drift_score,
                    'details': drift_result.details
                },
                'alerts': {
                    'active_count': len(model_alerts),
                    'critical_count': sum(1 for a in model_alerts if a.severity == AlertSeverity.CRITICAL),
                    'recent_alerts': [self._serialize_alert(a) for a in model_alerts[-10:]]
                },
                'recommendations': await self._generate_recommendations(model_id, drift_result, metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard: {str(e)}")
            return {}
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Get all monitored models
                for model_id in list(self.performance_history.keys()):
                    # Check for drift
                    drift_result = await self.detect_drift(model_id)
                    
                    if drift_result.detected:
                        await self._handle_drift_detection(model_id, drift_result)
                    
                    # Check performance thresholds
                    await self._check_performance_thresholds(model_id)
                    
                    # Clean up old data
                    await self._cleanup_old_data(model_id)
                
                # Wait for next check
                await asyncio.sleep(
                    self.monitoring_config['drift_check_interval_minutes'] * 60
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _detect_data_drift(self, model_id: str, 
                                 features: List[Dict]) -> DriftDetectionResult:
        """Detect data drift using statistical tests"""
        try:
            if model_id not in self.feature_distributions:
                return DriftDetectionResult(detected=False)
            
            baseline = self.feature_distributions[model_id]
            
            # Convert features to DataFrame
            current_df = pd.DataFrame(features)
            
            drift_scores = {}
            p_values = {}
            
            for column in current_df.columns:
                if column in baseline:
                    # Kolmogorov-Smirnov test for continuous features
                    if current_df[column].dtype in [np.float64, np.int64]:
                        statistic, p_value = stats.ks_2samp(
                            baseline[column],
                            current_df[column].values
                        )
                        drift_scores[column] = statistic
                        p_values[column] = p_value
                    
                    # Chi-square test for categorical features
                    else:
                        # Implementation would go here
                        pass
            
            # Check if any feature has significant drift
            significant_drift = any(
                p < self.monitoring_config['drift_significance_level'] 
                for p in p_values.values()
            )
            
            if significant_drift:
                # Find most drifted features
                drifted_features = [
                    f for f, p in p_values.items() 
                    if p < self.monitoring_config['drift_significance_level']
                ]
                
                return DriftDetectionResult(
                    detected=True,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=AlertSeverity.WARNING,
                    drift_score=max(drift_scores.values()),
                    p_value=min(p_values.values()),
                    details={
                        'drifted_features': drifted_features,
                        'drift_scores': drift_scores,
                        'p_values': p_values
                    },
                    recommendation="Consider retraining model with recent data"
                )
            
            return DriftDetectionResult(detected=False)
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return DriftDetectionResult(detected=False)
    
    async def _detect_prediction_drift(self, model_id: str,
                                      predictions: List) -> DriftDetectionResult:
        """Detect drift in model predictions"""
        try:
            # Get baseline prediction distribution
            baseline_key = f"baseline_predictions:{model_id}"
            baseline_data = await self.redis.get(baseline_key)
            
            if not baseline_data:
                # Store current as baseline
                await self.redis.set(baseline_key, json.dumps(predictions))
                return DriftDetectionResult(detected=False)
            
            baseline = json.loads(baseline_data)
            
            # Compare distributions
            statistic, p_value = stats.ks_2samp(baseline, predictions)
            
            if p_value < self.monitoring_config['drift_significance_level']:
                return DriftDetectionResult(
                    detected=True,
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=AlertSeverity.WARNING,
                    drift_score=statistic,
                    p_value=p_value,
                    details={
                        'baseline_mean': np.mean(baseline),
                        'current_mean': np.mean(predictions),
                        'baseline_std': np.std(baseline),
                        'current_std': np.std(predictions)
                    },
                    recommendation="Model predictions have shifted significantly"
                )
            
            return DriftDetectionResult(detected=False)
            
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {str(e)}")
            return DriftDetectionResult(detected=False)
    
    async def _detect_performance_degradation(self, model_id: str) -> DriftDetectionResult:
        """Detect performance degradation"""
        try:
            metrics = await self.get_performance_metrics(model_id, hours=24)
            
            if len(metrics) < 2:
                return DriftDetectionResult(detected=False)
            
            # Get baseline performance (first half of window)
            mid_point = len(metrics) // 2
            baseline_metrics = metrics[:mid_point]
            current_metrics = metrics[mid_point:]
            
            # Calculate average performance
            baseline_avg = np.mean([m.accuracy or 0 for m in baseline_metrics])
            current_avg = np.mean([m.accuracy or 0 for m in current_metrics])
            
            # Check for degradation
            degradation = (baseline_avg - current_avg) / baseline_avg if baseline_avg > 0 else 0
            
            if degradation > self.monitoring_config['performance_degradation_threshold']:
                return DriftDetectionResult(
                    detected=True,
                    drift_type=DriftType.PERFORMANCE_DEGRADATION,
                    severity=AlertSeverity.CRITICAL,
                    drift_score=degradation,
                    details={
                        'baseline_performance': baseline_avg,
                        'current_performance': current_avg,
                        'degradation_percentage': degradation * 100
                    },
                    recommendation="Immediate action required - model performance has degraded significantly"
                )
            
            return DriftDetectionResult(detected=False)
            
        except Exception as e:
            logger.error(f"Error detecting performance degradation: {str(e)}")
            return DriftDetectionResult(detected=False)
    
    async def _handle_drift_detection(self, model_id: str, drift_result: DriftDetectionResult):
        """Handle detected drift"""
        try:
            # Create alert
            alert = MonitoringAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                model_id=model_id,
                alert_type=drift_result.drift_type.value if drift_result.drift_type else "unknown",
                severity=drift_result.severity,
                message=f"Drift detected: {drift_result.recommendation}",
                details=drift_result.details
            )
            
            self.active_alerts.append(alert)
            
            # Publish alert
            await self._publish_alert(alert)
            
            # Trigger retraining if critical
            if drift_result.severity == AlertSeverity.CRITICAL:
                await self._trigger_retraining(model_id)
                
        except Exception as e:
            logger.error(f"Error handling drift detection: {str(e)}")
    
    async def _update_performance_metrics(self, model_id: str, prediction: Any, actual: Any):
        """Update performance metrics with new data"""
        try:
            # Get existing predictions for batch calculation
            cache_key = f"performance_batch:{model_id}"
            batch_data = await self.redis.get(cache_key)
            
            if not batch_data:
                batch_data = {'predictions': [], 'actuals': []}
            else:
                batch_data = json.loads(batch_data)
            
            batch_data['predictions'].append(prediction)
            batch_data['actuals'].append(actual)
            
            # Calculate metrics if enough samples
            if len(batch_data['predictions']) >= 100:
                metrics = PerformanceMetrics(
                    timestamp=datetime.utcnow(),
                    model_id=model_id,
                    predictions_count=len(batch_data['predictions']),
                    accuracy=accuracy_score(batch_data['actuals'], batch_data['predictions']),
                    precision=precision_score(batch_data['actuals'], batch_data['predictions'], average='weighted'),
                    recall=recall_score(batch_data['actuals'], batch_data['predictions'], average='weighted'),
                    f1_score=f1_score(batch_data['actuals'], batch_data['predictions'], average='weighted')
                )
                
                # Store metrics
                if model_id not in self.performance_history:
                    self.performance_history[model_id] = []
                self.performance_history[model_id].append(metrics)
                
                # Reset batch
                batch_data = {'predictions': [], 'actuals': []}
            
            # Save batch
            await self.redis.setex(cache_key, 3600, json.dumps(batch_data, default=str))
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _check_performance_thresholds(self, model_id: str):
        """Check if performance metrics exceed thresholds"""
        try:
            metrics = await self.get_performance_metrics(model_id, hours=1)
            
            if not metrics:
                return
            
            latest = metrics[-1]
            
            # Check accuracy threshold
            if latest.accuracy and latest.accuracy < 0.7:  # Below 70%
                alert = MonitoringAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    model_id=model_id,
                    alert_type="low_accuracy",
                    severity=AlertSeverity.WARNING,
                    message=f"Model accuracy below threshold: {latest.accuracy:.2%}",
                    details={'accuracy': latest.accuracy}
                )
                self.active_alerts.append(alert)
                await self._publish_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {str(e)}")
    
    async def _load_baseline_distribution(self, model_id: str):
        """Load baseline feature distribution for drift detection"""
        try:
            # This would load from training data or initial production data
            # For now, using placeholder
            self.feature_distributions[model_id] = {}
            
        except Exception as e:
            logger.error(f"Error loading baseline distribution: {str(e)}")
    
    async def _get_recent_predictions(self, model_id: str, limit: int = 1000) -> List[Dict]:
        """Get recent predictions from cache"""
        try:
            data = await self.redis.lrange(f"predictions:{model_id}", 0, limit - 1)
            return [json.loads(d) for d in data]
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {str(e)}")
            return []
    
    async def _cleanup_old_data(self, model_id: str):
        """Clean up old monitoring data"""
        try:
            # Keep only last 7 days of performance history
            cutoff = datetime.utcnow() - timedelta(days=7)
            if model_id in self.performance_history:
                self.performance_history[model_id] = [
                    m for m in self.performance_history[model_id]
                    if m.timestamp >= cutoff
                ]
            
            # Clean up resolved alerts
            self.active_alerts = [
                a for a in self.active_alerts
                if not a.resolved or (a.resolved_at and a.resolved_at >= cutoff)
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def _trigger_retraining(self, model_id: str):
        """Trigger model retraining"""
        try:
            event = {
                'event_type': 'trigger_retraining',
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'drift_detected'
            }
            
            await self.kafka_client.send_message(
                'model-events',
                json.dumps(event)
            )
            
            logger.info(f"Triggered retraining for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {str(e)}")
    
    async def _publish_alert(self, alert: MonitoringAlert):
        """Publish monitoring alert"""
        try:
            await self.kafka_client.send_message(
                'monitoring-alerts',
                json.dumps(self._serialize_alert(alert))
            )
            
        except Exception as e:
            logger.error(f"Error publishing alert: {str(e)}")
    
    def _calculate_current_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate current performance summary"""
        if not metrics:
            return {}
        
        latest = metrics[-1]
        return {
            'accuracy': latest.accuracy or 0,
            'precision': latest.precision or 0,
            'recall': latest.recall or 0,
            'f1_score': latest.f1_score or 0
        }
    
    def _calculate_performance_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """Calculate performance trend"""
        if len(metrics) < 2:
            return "stable"
        
        # Compare first half vs second half
        mid = len(metrics) // 2
        first_half = [m.accuracy or 0 for m in metrics[:mid]]
        second_half = [m.accuracy or 0 for m in metrics[mid:]]
        
        if not first_half or not second_half:
            return "stable"
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        if avg_second > avg_first * 1.05:
            return "improving"
        elif avg_second < avg_first * 0.95:
            return "degrading"
        else:
            return "stable"
    
    async def _generate_recommendations(self, model_id: str, 
                                       drift_result: DriftDetectionResult,
                                       metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        if drift_result.detected:
            recommendations.append(drift_result.recommendation or "Monitor closely for continued drift")
        
        if metrics:
            trend = self._calculate_performance_trend(metrics)
            if trend == "degrading":
                recommendations.append("Performance is degrading - consider retraining")
            elif trend == "improving":
                recommendations.append("Performance is improving - current model is adapting well")
        
        if len(self.active_alerts) > 5:
            recommendations.append("Multiple alerts active - review model health")
        
        if not recommendations:
            recommendations.append("Model is performing within normal parameters")
        
        return recommendations
    
    def _serialize_alert(self, alert: MonitoringAlert) -> Dict[str, Any]:
        """Serialize alert for storage/transmission"""
        return {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'model_id': alert.model_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity.value,
            'message': alert.message,
            'details': alert.details,
            'resolved': alert.resolved,
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
        }
