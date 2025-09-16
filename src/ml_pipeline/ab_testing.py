"""
A/B Testing Framework

Manages A/B tests for model comparison in production environments.
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
from scipy import stats
import hashlib

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Signal
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.config.settings import get_settings
from .model_versioning import ModelVersionManager

logger = logging.getLogger(__name__)
settings = get_settings()


class TestStatus(str, Enum):
    """A/B test status"""
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TrafficSplitStrategy(str, Enum):
    """Traffic splitting strategies"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    USER_BASED = "user_based"
    TIME_BASED = "time_based"


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    name: str
    model_a_id: str  # Control
    model_b_id: str  # Variant
    traffic_split: float = 0.5  # Percentage to model B
    split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    min_sample_size: int = 100
    max_duration_days: int = 30
    success_metric: str = "accuracy"
    significance_level: float = 0.05
    enable_early_stopping: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B test results"""
    test_id: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    samples_a: int
    samples_b: int
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    statistical_results: Dict[str, Any]
    winner: Optional[str] = None
    confidence_level: Optional[float] = None
    improvement: Optional[float] = None
    notes: List[str] = field(default_factory=list)


class ABTestingFramework:
    """
    Manages A/B testing for model comparison
    """
    
    def __init__(self, version_manager: Optional[ModelVersionManager] = None):
        self.redis = get_redis()
        self.version_manager = version_manager or ModelVersionManager()
        
        # Active tests
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        
        # Test metrics storage
        self.test_metrics: Dict[str, Dict] = {}
        
        # Monitoring task
        self._monitoring_task = None
    
    async def create_test(self, config: ABTestConfig) -> str:
        """
        Create a new A/B test
        
        Args:
            config: Test configuration
            
        Returns:
            Test ID
        """
        try:
            # Validate models exist
            model_a = await self.version_manager.get_model_version(config.model_a_id)
            model_b = await self.version_manager.get_model_version(config.model_b_id)
            
            if not model_a or not model_b:
                raise ValueError("One or both models not found")
            
            # Initialize test result
            result = ABTestResult(
                test_id=config.test_id,
                status=TestStatus.PLANNING,
                start_time=datetime.utcnow(),
                end_time=None,
                samples_a=0,
                samples_b=0,
                metrics_a={},
                metrics_b={},
                statistical_results={}
            )
            
            # Store configuration and result
            self.active_tests[config.test_id] = config
            self.test_results[config.test_id] = result
            
            # Initialize metrics storage
            self.test_metrics[config.test_id] = {
                'model_a': [],
                'model_b': []
            }
            
            # Cache test configuration
            await self._cache_test_config(config)
            
            logger.info(f"Created A/B test {config.test_id}")
            return config.test_id
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {str(e)}")
            raise
    
    async def start_test(self, test_id: str) -> bool:
        """
        Start an A/B test
        
        Args:
            test_id: Test identifier
            
        Returns:
            bool: Success status
        """
        try:
            if test_id not in self.active_tests:
                logger.error(f"Test {test_id} not found")
                return False
            
            result = self.test_results[test_id]
            
            if result.status != TestStatus.PLANNING:
                logger.error(f"Test {test_id} cannot be started from status {result.status}")
                return False
            
            # Update status
            result.status = TestStatus.RUNNING
            result.start_time = datetime.utcnow()
            
            # Start monitoring task if not running
            if not self._monitoring_task:
                self._monitoring_task = asyncio.create_task(self._monitor_tests())
            
            # Cache updated status
            await self._cache_test_result(result)
            
            logger.info(f"Started A/B test {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting test: {str(e)}")
            return False
    
    async def route_request(self, test_id: str, request_id: str) -> str:
        """
        Route a request to model A or B based on split strategy
        
        Args:
            test_id: Test identifier
            request_id: Unique request identifier
            
        Returns:
            Model ID to use
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            config = self.active_tests[test_id]
            result = self.test_results[test_id]
            
            if result.status != TestStatus.RUNNING:
                # Default to model A if test not running
                return config.model_a_id
            
            # Determine routing based on strategy
            use_model_b = False
            
            if config.split_strategy == TrafficSplitStrategy.RANDOM:
                use_model_b = np.random.random() < config.traffic_split
                
            elif config.split_strategy == TrafficSplitStrategy.HASH_BASED:
                hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                use_model_b = (hash_val % 100) < (config.traffic_split * 100)
                
            elif config.split_strategy == TrafficSplitStrategy.USER_BASED:
                # Use request_id as user_id for consistent routing
                user_hash = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                use_model_b = (user_hash % 100) < (config.traffic_split * 100)
                
            elif config.split_strategy == TrafficSplitStrategy.TIME_BASED:
                # Alternate based on time windows
                minutes = datetime.utcnow().minute
                use_model_b = (minutes % 2 == 0) and (np.random.random() < config.traffic_split * 2)
            
            # Update sample counts
            if use_model_b:
                result.samples_b += 1
                return config.model_b_id
            else:
                result.samples_a += 1
                return config.model_a_id
                
        except Exception as e:
            logger.error(f"Error routing request: {str(e)}")
            # Default to model A on error
            if test_id in self.active_tests:
                return self.active_tests[test_id].model_a_id
            raise
    
    async def record_outcome(self, test_id: str, model_id: str, 
                            outcome: Dict[str, Any]) -> bool:
        """
        Record outcome for a test sample
        
        Args:
            test_id: Test identifier
            model_id: Model that produced the outcome
            outcome: Outcome metrics
            
        Returns:
            bool: Success status
        """
        try:
            if test_id not in self.active_tests:
                return False
            
            config = self.active_tests[test_id]
            
            # Determine which model
            if model_id == config.model_a_id:
                self.test_metrics[test_id]['model_a'].append(outcome)
            elif model_id == config.model_b_id:
                self.test_metrics[test_id]['model_b'].append(outcome)
            else:
                logger.warning(f"Unknown model {model_id} for test {test_id}")
                return False
            
            # Check if should analyze results
            await self._check_test_completion(test_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording outcome: {str(e)}")
            return False
    
    async def analyze_results(self, test_id: str) -> ABTestResult:
        """
        Analyze A/B test results
        
        Args:
            test_id: Test identifier
            
        Returns:
            ABTestResult with analysis
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            config = self.active_tests[test_id]
            result = self.test_results[test_id]
            
            # Get metrics
            metrics_a = self.test_metrics[test_id]['model_a']
            metrics_b = self.test_metrics[test_id]['model_b']
            
            if not metrics_a or not metrics_b:
                result.notes.append("Insufficient data for analysis")
                return result
            
            # Calculate aggregate metrics
            success_metric = config.success_metric
            
            values_a = [m.get(success_metric, 0) for m in metrics_a]
            values_b = [m.get(success_metric, 0) for m in metrics_b]
            
            result.metrics_a = {
                'mean': np.mean(values_a),
                'std': np.std(values_a),
                'median': np.median(values_a),
                'count': len(values_a)
            }
            
            result.metrics_b = {
                'mean': np.mean(values_b),
                'std': np.std(values_b),
                'median': np.median(values_b),
                'count': len(values_b)
            }
            
            # Perform statistical tests
            stat_results = await self._perform_statistical_tests(
                values_a, values_b, config.significance_level
            )
            result.statistical_results = stat_results
            
            # Determine winner
            if stat_results['p_value'] < config.significance_level:
                if result.metrics_b['mean'] > result.metrics_a['mean']:
                    result.winner = 'model_b'
                    result.improvement = ((result.metrics_b['mean'] - result.metrics_a['mean']) 
                                        / result.metrics_a['mean'] * 100)
                else:
                    result.winner = 'model_a'
                    result.improvement = ((result.metrics_a['mean'] - result.metrics_b['mean']) 
                                        / result.metrics_b['mean'] * 100)
                
                result.confidence_level = 1 - stat_results['p_value']
            else:
                result.notes.append("No statistically significant difference found")
            
            # Cache results
            await self._cache_test_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            raise
    
    async def stop_test(self, test_id: str, reason: Optional[str] = None) -> ABTestResult:
        """
        Stop an A/B test
        
        Args:
            test_id: Test identifier
            reason: Reason for stopping
            
        Returns:
            Final test results
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} not found")
            
            result = self.test_results[test_id]
            
            # Analyze final results
            result = await self.analyze_results(test_id)
            
            # Update status
            result.status = TestStatus.COMPLETED
            result.end_time = datetime.utcnow()
            
            if reason:
                result.notes.append(f"Test stopped: {reason}")
            
            # Cache final results
            await self._cache_test_result(result)
            
            # Clean up active test
            del self.active_tests[test_id]
            
            logger.info(f"Stopped A/B test {test_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error stopping test: {str(e)}")
            raise
    
    async def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """
        Get current test status
        
        Args:
            test_id: Test identifier
            
        Returns:
            Dict with test status
        """
        try:
            if test_id not in self.test_results:
                # Check cache
                cached = await self.redis.get(f"ab_test_result:{test_id}")
                if cached:
                    return json.loads(cached)
                return {}
            
            result = self.test_results[test_id]
            config = self.active_tests.get(test_id)
            
            status = {
                'test_id': test_id,
                'name': config.name if config else "Unknown",
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'samples_a': result.samples_a,
                'samples_b': result.samples_b,
                'current_metrics_a': result.metrics_a,
                'current_metrics_b': result.metrics_b,
                'winner': result.winner,
                'confidence_level': result.confidence_level,
                'improvement': result.improvement
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting test status: {str(e)}")
            return {}
    
    async def _monitor_tests(self):
        """Monitor running tests for completion conditions"""
        while True:
            try:
                for test_id in list(self.active_tests.keys()):
                    await self._check_test_completion(test_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in test monitoring: {str(e)}")
    
    async def _check_test_completion(self, test_id: str):
        """Check if test should be completed"""
        try:
            config = self.active_tests.get(test_id)
            result = self.test_results.get(test_id)
            
            if not config or not result:
                return
            
            if result.status != TestStatus.RUNNING:
                return
            
            # Check duration
            duration = datetime.utcnow() - result.start_time
            if duration.days >= config.max_duration_days:
                await self.stop_test(test_id, "Maximum duration reached")
                return
            
            # Check sample size
            if (result.samples_a >= config.min_sample_size and 
                result.samples_b >= config.min_sample_size):
                
                # Check for early stopping
                if config.enable_early_stopping:
                    temp_result = await self.analyze_results(test_id)
                    
                    # Stop if clear winner with high confidence
                    if temp_result.confidence_level and temp_result.confidence_level > 0.99:
                        await self.stop_test(test_id, "Early stopping - clear winner")
                        
        except Exception as e:
            logger.error(f"Error checking test completion: {str(e)}")
    
    async def _perform_statistical_tests(self, values_a: List[float], 
                                        values_b: List[float],
                                        significance_level: float) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        try:
            results = {}
            
            # T-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            results['t_statistic'] = float(t_stat)
            results['p_value'] = float(p_value)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(values_b) - np.mean(values_a)) / pooled_std
                results['effect_size'] = float(cohens_d)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            results['mann_whitney_u'] = float(u_stat)
            results['mann_whitney_p'] = float(u_p_value)
            
            # Confidence intervals
            ci_a = stats.t.interval(0.95, len(values_a)-1, 
                                   loc=np.mean(values_a), 
                                   scale=stats.sem(values_a))
            ci_b = stats.t.interval(0.95, len(values_b)-1,
                                   loc=np.mean(values_b),
                                   scale=stats.sem(values_b))
            
            results['ci_95_a'] = [float(ci_a[0]), float(ci_a[1])]
            results['ci_95_b'] = [float(ci_b[0]), float(ci_b[1])]
            
            # Statistical power
            if results.get('effect_size'):
                from statsmodels.stats.power import ttest_power
                power = ttest_power(results['effect_size'], len(values_a), 
                                   significance_level)
                results['statistical_power'] = float(power)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {str(e)}")
            return {}
    
    async def _cache_test_config(self, config: ABTestConfig):
        """Cache test configuration"""
        await self.redis.setex(
            f"ab_test_config:{config.test_id}",
            86400 * 30,  # 30 days
            json.dumps({
                'test_id': config.test_id,
                'name': config.name,
                'model_a_id': config.model_a_id,
                'model_b_id': config.model_b_id,
                'traffic_split': config.traffic_split,
                'split_strategy': config.split_strategy.value,
                'min_sample_size': config.min_sample_size,
                'max_duration_days': config.max_duration_days,
                'success_metric': config.success_metric,
                'significance_level': config.significance_level,
                'enable_early_stopping': config.enable_early_stopping,
                'metadata': config.metadata
            }, default=str)
        )
    
    async def _cache_test_result(self, result: ABTestResult):
        """Cache test result"""
        await self.redis.setex(
            f"ab_test_result:{result.test_id}",
            86400 * 30,  # 30 days
            json.dumps({
                'test_id': result.test_id,
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'samples_a': result.samples_a,
                'samples_b': result.samples_b,
                'metrics_a': result.metrics_a,
                'metrics_b': result.metrics_b,
                'statistical_results': result.statistical_results,
                'winner': result.winner,
                'confidence_level': result.confidence_level,
                'improvement': result.improvement,
                'notes': result.notes
            }, default=str)
        )
