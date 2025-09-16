#!/usr/bin/env python3
"""
Initialize Kafka topics for the crypto signals system.
"""

import asyncio
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings


def create_topics():
    """Create Kafka topics."""
    admin_client = KafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id='topic_creator'
    )
    
    topics = [
        NewTopic(
            name=settings.kafka_topic_signals,
            num_partitions=3,
            replication_factor=1,
            topic_configs={
                'retention.ms': '604800000',  # 7 days
                'compression.type': 'gzip',
                'max.message.bytes': '1048576'  # 1MB
            }
        ),
        NewTopic(
            name=settings.kafka_topic_validation,
            num_partitions=3,
            replication_factor=1,
            topic_configs={
                'retention.ms': '604800000',  # 7 days
                'compression.type': 'gzip'
            }
        ),
        NewTopic(
            name=f"{settings.kafka_topic_signals}-alerts",
            num_partitions=1,
            replication_factor=1,
            topic_configs={
                'retention.ms': '86400000',  # 1 day
                'compression.type': 'gzip'
            }
        ),
        NewTopic(
            name=f"{settings.kafka_topic_signals}-dlq",  # Dead letter queue
            num_partitions=1,
            replication_factor=1,
            topic_configs={
                'retention.ms': '2592000000',  # 30 days
                'compression.type': 'gzip'
            }
        ),
        NewTopic(
            name="resource-alerts",
            num_partitions=2,
            replication_factor=1,
            topic_configs={
                'retention.ms': '172800000',  # 2 days
                'compression.type': 'gzip'
            }
        ),
        NewTopic(
            name="feedback-events",
            num_partitions=2,
            replication_factor=1,
            topic_configs={
                'retention.ms': '604800000',  # 7 days
                'compression.type': 'gzip'
            }
        ),
        NewTopic(
            name="improvement-pipeline",
            num_partitions=1,
            replication_factor=1,
            topic_configs={
                'retention.ms': '2592000000',  # 30 days
                'compression.type': 'gzip'
            }
        )
    ]
    
    for topic in topics:
        try:
            admin_client.create_topics([topic])
            print(f"✅ Created topic: {topic.name}")
        except TopicAlreadyExistsError:
            print(f"ℹ️  Topic already exists: {topic.name}")
        except Exception as e:
            print(f"❌ Failed to create topic {topic.name}: {e}")
    
    # List all topics
    print("\n📋 All topics:")
    for topic in admin_client.list_topics():
        if not topic.startswith('__'):  # Skip internal topics
            print(f"  - {topic}")
    
    admin_client.close()


if __name__ == "__main__":
    print("🚀 Initializing Kafka topics...")
    create_topics()
    print("✅ Kafka topics initialization complete!")
