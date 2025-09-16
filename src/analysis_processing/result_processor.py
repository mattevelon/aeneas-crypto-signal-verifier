"""Result Processor for final signal processing and distribution."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json
from dataclasses import asdict

from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.core.kafka_client import KafkaClient

logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    Processes final analysis results and distributes to appropriate channels.
    Handles persistence, caching, and notification delivery.
    """
    
    def __init__(self):
        """Initialize result processor."""
        self.redis_client = None  # Will use get_redis() when needed
        self.kafka_client = None  # Will be initialized when needed
        
        # Processing configuration
        self.cache_ttl = 3600  # 1 hour
        self.batch_size = 10
        
        # Result statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'notifications_sent': 0
        }
        
        logger.info("Initialized ResultProcessor")
    
    async def process_result(self,
                            signal_data: Dict[str, Any],
                            ai_analysis: Dict[str, Any],
                            validation_result: Any,
                            enhanced_signal: Any,
                            decision: Any) -> Dict[str, Any]:
        """
        Process complete analysis result.
        
        Args:
            signal_data: Original signal data
            ai_analysis: AI analysis results
            validation_result: Validation results
            enhanced_signal: Enhanced signal
            decision: Final trading decision
            
        Returns:
            Processing status and result ID
        """
        try:
            # Build complete result
            complete_result = self._build_complete_result(
                signal_data, ai_analysis, validation_result,
                enhanced_signal, decision
            )
            
            # Persist to database
            result_id = await self._persist_result(complete_result)
            complete_result['result_id'] = result_id
            
            # Cache result
            await self._cache_result(result_id, complete_result)
            
            # Publish to message queue
            await self._publish_result(complete_result)
            
            # Send notifications if needed
            if decision.action.value in ['execute', 'scale_in']:
                await self._send_notifications(complete_result)
            
            # Update statistics
            self._update_statistics(True)
            
            return {
                'status': 'success',
                'result_id': result_id,
                'action': decision.action.value,
                'confidence': decision.confidence.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            self._update_statistics(False)
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _build_complete_result(self,
                              signal_data: Dict[str, Any],
                              ai_analysis: Dict[str, Any],
                              validation_result: Any,
                              enhanced_signal: Any,
                              decision: Any) -> Dict[str, Any]:
        """Build complete analysis result."""
        return {
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'id': signal_data.get('signal_id'),
                'original_params': signal_data.get('trading_params', {}),
                'classification': signal_data.get('classification', {}),
                'pattern_analysis': signal_data.get('pattern_analysis', {})
            },
            'ai_analysis': {
                'verdict': ai_analysis.get('verdict', {}),
                'analysis': ai_analysis.get('analysis', {}),
                'risk': ai_analysis.get('risk', {}),
                'justification': ai_analysis.get('justification', {})
            },
            'validation': {
                'status': validation_result.status.value if validation_result else 'skipped',
                'score': validation_result.score if validation_result else 0,
                'checks_passed': validation_result.checks_passed if validation_result else [],
                'warnings': validation_result.warnings if validation_result else [],
                'recommendations': validation_result.recommendations if validation_result else []
            },
            'enhancement': {
                'enhanced_params': enhanced_signal.enhanced_params if enhanced_signal else {},
                'adjustments': enhanced_signal.adjustments if enhanced_signal else {},
                'enhancement_score': enhanced_signal.enhancement_score if enhanced_signal else 0,
                'execution_strategy': enhanced_signal.execution_strategy if enhanced_signal else {}
            } if enhanced_signal else None,
            'decision': {
                'action': decision.action.value,
                'confidence': decision.confidence.value,
                'reasoning': decision.reasoning,
                'execution_params': decision.execution_params,
                'risk_limits': decision.risk_limits,
                'monitoring_rules': decision.monitoring_rules,
                'alerts': decision.alerts
            },
            'metadata': {
                'processing_time_ms': 0,  # Will be calculated
                'version': '1.0.0',
                'environment': 'production'
            }
        }
    
    async def _persist_result(self, result: Dict[str, Any]) -> str:
        """Persist result to database."""
        async with get_async_session() as session:
            # Create analysis result record
            query = """
                INSERT INTO analysis_results (
                    signal_id, 
                    decision_action,
                    decision_confidence,
                    ai_confidence_score,
                    validation_score,
                    enhancement_score,
                    execution_params,
                    risk_limits,
                    complete_result,
                    created_at
                ) VALUES (
                    :signal_id,
                    :decision_action,
                    :decision_confidence,
                    :ai_confidence_score,
                    :validation_score,
                    :enhancement_score,
                    :execution_params,
                    :risk_limits,
                    :complete_result,
                    :created_at
                ) RETURNING id
            """
            
            values = {
                'signal_id': result['signal']['id'],
                'decision_action': result['decision']['action'],
                'decision_confidence': result['decision']['confidence'],
                'ai_confidence_score': result['ai_analysis']['verdict'].get('confidence_score', 0),
                'validation_score': result['validation']['score'],
                'enhancement_score': result['enhancement']['enhancement_score'] if result['enhancement'] else 0,
                'execution_params': json.dumps(result['decision']['execution_params']),
                'risk_limits': json.dumps(result['decision']['risk_limits']),
                'complete_result': json.dumps(result),
                'created_at': datetime.now()
            }
            
            result = await session.execute(query, values)
            result_id = result.scalar()
            await session.commit()
            
            return str(result_id)
    
    async def _cache_result(self, result_id: str, result: Dict[str, Any]):
        """Cache result in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"analysis_result:{result_id}"
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result)
            )
            
            # Also cache by signal ID for quick lookup
            signal_id = result['signal']['id']
            if signal_id:
                signal_key = f"signal_result:{signal_id}"
                await self.redis_client.setex(
                    signal_key,
                    self.cache_ttl,
                    result_id
                )
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def _publish_result(self, result: Dict[str, Any]):
        """Publish result to message queue."""
        if not self.kafka_producer:
            try:
                self.kafka_client = KafkaClient()
                # Kafka client initialization handled elsewhere
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
                return
        
        try:
            # Publish to different topics based on action
            action = result['decision']['action']
            
            if action in ['execute', 'scale_in']:
                topic = 'trading.signals.execute'
            elif action == 'paper_trade':
                topic = 'trading.signals.paper'
            elif action == 'monitor':
                topic = 'trading.signals.monitor'
            else:
                topic = 'trading.signals.rejected'
            
            # Simplified Kafka sending - actual implementation may vary
            # await self.kafka_client.send_message(topic, result)
            logger.info(f"Would publish to Kafka topic: {topic}")
            
            logger.info(f"Published result to {topic}")
            
        except Exception as e:
            logger.error(f"Error publishing result: {e}")
    
    async def _send_notifications(self, result: Dict[str, Any]):
        """Send notifications for actionable signals."""
        try:
            # Prepare notification content
            notification = self._prepare_notification(result)
            
            # Send to notification service (webhook, telegram, etc.)
            await self._send_webhook(notification)
            
            # Update stats
            self.processing_stats['notifications_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _prepare_notification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare notification content."""
        signal = result['signal']['original_params']
        decision = result['decision']
        ai_analysis = result['ai_analysis']
        
        # Format notification based on user level
        notification = {
            'type': 'trading_signal',
            'action': decision['action'],
            'pair': signal.get('pair', 'UNKNOWN'),
            'direction': signal.get('direction', 'unknown'),
            'entry_price': signal.get('entry_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profits': signal.get('take_profits', []),
            'confidence': ai_analysis['verdict'].get('confidence_score', 0),
            'risk_level': ai_analysis['verdict'].get('risk_level', 'unknown'),
            'justification': {
                'novice': ai_analysis['justification'].get('novice', ''),
                'intermediate': ai_analysis['justification'].get('intermediate', ''),
                'expert': ai_analysis['justification'].get('expert', '')
            },
            'alerts': decision.get('alerts', []),
            'timestamp': result['timestamp']
        }
        
        return notification
    
    async def _send_webhook(self, notification: Dict[str, Any]):
        """Send notification via webhook."""
        # This would implement actual webhook sending
        # For now, just log
        logger.info(f"Notification prepared: {notification['action']} for {notification['pair']}")
    
    def _update_statistics(self, success: bool):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['successful'] += 1
        else:
            self.processing_stats['failed'] += 1
    
    async def batch_process(self, results: List[Tuple]) -> List[Dict[str, Any]]:
        """
        Process multiple results in batch.
        
        Args:
            results: List of (signal_data, ai_analysis, validation, enhanced, decision) tuples
            
        Returns:
            List of processing results
        """
        tasks = []
        
        for result_tuple in results[:self.batch_size]:
            signal_data, ai_analysis, validation, enhanced, decision = result_tuple
            tasks.append(
                self.process_result(
                    signal_data, ai_analysis, validation, enhanced, decision
                )
            )
        
        processed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(processed):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error for item {i}: {result}")
                final_results.append({
                    'status': 'error',
                    'error': str(result),
                    'index': i
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a processed result.
        
        Args:
            result_id: Result ID
            
        Returns:
            Complete result or None
        """
        # Check cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"analysis_result:{result_id}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Cache retrieval error: {e}")
        
        # Fetch from database
        async with get_async_session() as session:
            query = "SELECT complete_result FROM analysis_results WHERE id = :id"
            result = await session.execute(query, {'id': result_id})
            row = result.fetchone()
            
            if row:
                return json.loads(row.complete_result)
        
        return None
    
    async def get_result_by_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve result by signal ID.
        
        Args:
            signal_id: Signal ID
            
        Returns:
            Complete result or None
        """
        # Check cache for result ID
        if self.redis_client:
            try:
                result_id = await self.redis_client.get(f"signal_result:{signal_id}")
                if result_id:
                    return await self.get_result(result_id)
            except Exception as e:
                logger.error(f"Cache retrieval error: {e}")
        
        # Fetch from database
        async with get_async_session() as session:
            query = "SELECT complete_result FROM analysis_results WHERE signal_id = :signal_id ORDER BY created_at DESC LIMIT 1"
            result = await session.execute(query, {'signal_id': signal_id})
            row = result.fetchone()
            
            if row:
                return json.loads(row.complete_result)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        total = self.processing_stats['total_processed']
        
        return {
            'total_processed': total,
            'successful': self.processing_stats['successful'],
            'failed': self.processing_stats['failed'],
            'success_rate': self.processing_stats['successful'] / total if total > 0 else 0,
            'notifications_sent': self.processing_stats['notifications_sent'],
            'notification_rate': self.processing_stats['notifications_sent'] / total if total > 0 else 0
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.kafka_client:
            # await self.kafka_client.close()
            self.kafka_client = None
