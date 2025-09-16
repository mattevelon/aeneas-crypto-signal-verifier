"""
Kafka client for event streaming.
"""

from typing import Optional, Dict, Any, List
import json
import asyncio

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import structlog

from src.config.settings import settings

logger = structlog.get_logger()


class KafkaClient:
    """Wrapper class for Kafka operations."""
    
    def __init__(self):
        self.producer = None
        self.consumer = None
        
    async def send_message(self, topic: str, message: Any):
        """Send a message to a Kafka topic."""
        # Placeholder implementation
        logger.info(f"Would send message to topic {topic}")
        
    async def close(self):
        """Close Kafka connections."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()

# Global Kafka clients
kafka_producer: Optional[AIOKafkaProducer] = None
kafka_consumers: Dict[str, AIOKafkaConsumer] = {}


async def init_kafka():
    """Initialize Kafka producer."""
    global kafka_producer
    
    try:
        kafka_producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='gzip',
            max_batch_size=16384,
            linger_ms=10,
            retry_backoff_ms=100,
            request_timeout_ms=30000,
        )
        
        await kafka_producer.start()
        logger.info("Kafka producer initialized")
        
        # Create topics if they don't exist
        await create_topics()
        
    except Exception as e:
        logger.warning(f"Kafka initialization failed (non-critical): {e}")
        logger.info("Application will continue without Kafka support")
        # Don't raise - allow app to continue without Kafka


async def close_kafka():
    """Close Kafka connections."""
    global kafka_producer, kafka_consumers
    
    if kafka_producer:
        await kafka_producer.stop()
        kafka_producer = None
    
    for consumer in kafka_consumers.values():
        await consumer.stop()
    
    kafka_consumers.clear()
    logger.info("Kafka connections closed")


async def create_topics():
    """Create Kafka topics if they don't exist."""
    # In production, topics should be created via Kafka admin tools
    # This is a placeholder for development
    topics = [
        settings.kafka_topic_signals,
        settings.kafka_topic_validation,
        f"{settings.kafka_topic_signals}-alerts"
    ]
    logger.info(f"Kafka topics ready: {topics}")


def get_producer() -> Optional[AIOKafkaProducer]:
    """Get Kafka producer instance."""
    return kafka_producer


async def send_event(
    topic: str,
    event: Dict[str, Any],
    key: Optional[str] = None
) -> bool:
    """Send event to Kafka topic."""
    try:
        producer = get_producer()
        if not producer:
            logger.debug(f"Kafka not available, skipping event to {topic}")
            return False
            
        await producer.send(
            topic=topic,
            value=event,
            key=key
        )
        logger.debug(f"Event sent to {topic}", event_type=event.get('type'))
        return True
        
    except KafkaError as e:
        logger.error(f"Failed to send event to {topic}", error=str(e))
        return False


async def create_consumer(
    topics: List[str],
    group_id: str,
    auto_offset_reset: str = 'latest'
) -> AIOKafkaConsumer:
    """Create Kafka consumer."""
    try:
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=100,
        )
        
        await consumer.start()
        kafka_consumers[group_id] = consumer
        logger.info(f"Kafka consumer created", group_id=group_id, topics=topics)
        
        return consumer
        
    except Exception as e:
        logger.error("Failed to create Kafka consumer", error=str(e))
        raise


class KafkaEventPublisher:
    """Helper class for publishing events."""
    
    @staticmethod
    async def publish_signal(signal_data: Dict[str, Any]) -> bool:
        """Publish signal event."""
        event = {
            "type": "signal.created",
            "data": signal_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        return await send_event(settings.kafka_topic_signals, event)
    
    @staticmethod
    async def publish_validation(validation_data: Dict[str, Any]) -> bool:
        """Publish validation event."""
        event = {
            "type": "signal.validated",
            "data": validation_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        return await send_event(settings.kafka_topic_validation, event)
    
    @staticmethod
    async def publish_alert(alert_data: Dict[str, Any]) -> bool:
        """Publish alert event."""
        event = {
            "type": "alert",
            "data": alert_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        return await send_event(f"{settings.kafka_topic_signals}-alerts", event)
