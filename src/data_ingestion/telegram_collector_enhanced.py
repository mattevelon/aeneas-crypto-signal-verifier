"""
Enhanced Telegram data collector with connection pooling and advanced features.
Phase 2 Implementation: Complete Data Collection Pipeline
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque
from enum import Enum
import time
import os
from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
from telethon.errors import SessionPasswordNeededError, FloodWaitError, ChannelPrivateError
from telethon.sessions import StringSession
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.core.database import get_db_context
from src.core.kafka_client import KafkaEventPublisher
from src.core.redis_client import signal_cache
from src.models import TelegramMessage
from src.core.signal_detector import SignalDetector

logger = structlog.get_logger()

# Check if Telegram credentials are available
TELEGRAM_ENABLED = settings.has_telegram_credentials


class MessagePriority(Enum):
    """Message priority levels."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class ConnectionPool:
    """Manages multiple Telegram client connections."""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.connections: List[TelegramClient] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.in_use: Set[TelegramClient] = set()
        self.health_check_interval = 60  # seconds
        
    async def initialize(self, api_id: int, api_hash: str, session_name: str):
        """Initialize connection pool with multiple clients."""
        for i in range(self.size):
            session = StringSession() if i > 0 else session_name
            client = TelegramClient(
                f"{session_name}_{i}" if i > 0 else session_name,
                api_id,
                api_hash
            )
            self.connections.append(client)
            await self.available.put(client)
        
        # Start health check task
        asyncio.create_task(self._health_check_loop())
    
    async def acquire(self) -> TelegramClient:
        """Acquire a connection from the pool."""
        client = await self.available.get()
        self.in_use.add(client)
        
        # Ensure connection is alive
        if not client.is_connected():
            await client.connect()
        
        return client
    
    async def release(self, client: TelegramClient):
        """Release a connection back to the pool."""
        if client in self.in_use:
            self.in_use.remove(client)
            await self.available.put(client)
    
    async def _health_check_loop(self):
        """Periodically check connection health."""
        while True:
            await asyncio.sleep(self.health_check_interval)
            for client in self.connections:
                if client not in self.in_use:
                    try:
                        if not client.is_connected():
                            await client.connect()
                    except Exception as e:
                        logger.error(f"Health check failed for connection: {e}")
    
    async def close_all(self):
        """Close all connections in the pool."""
        for client in self.connections:
            if client.is_connected():
                await client.disconnect()


class MessageQueue:
    """Priority message queue with dead letter support."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.high_priority = asyncio.Queue()
        self.medium_priority = asyncio.Queue()
        self.low_priority = asyncio.Queue()
        self.dead_letter = deque(maxlen=1000)
        self.processed_hashes = set()
        self.retry_counts = {}
        
    async def put(self, message: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM):
        """Add message to appropriate priority queue."""
        # Deduplication
        msg_hash = self._hash_message(message)
        if msg_hash in self.processed_hashes:
            logger.debug(f"Duplicate message detected: {msg_hash}")
            return
        
        self.processed_hashes.add(msg_hash)
        
        # Add to appropriate queue
        if priority == MessagePriority.HIGH:
            await self.high_priority.put(message)
        elif priority == MessagePriority.MEDIUM:
            await self.medium_priority.put(message)
        else:
            await self.low_priority.put(message)
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get next message from priority queues."""
        # Check high priority first
        if not self.high_priority.empty():
            return await self.high_priority.get()
        
        # Then medium priority
        if not self.medium_priority.empty():
            return await self.medium_priority.get()
        
        # Finally low priority
        if not self.low_priority.empty():
            return await self.low_priority.get()
        
        return None
    
    def add_to_dead_letter(self, message: Dict[str, Any], error: str):
        """Add failed message to dead letter queue."""
        message['error'] = error
        message['failed_at'] = datetime.utcnow().isoformat()
        self.dead_letter.append(message)
        logger.error(f"Message added to dead letter queue: {error}")
    
    def _hash_message(self, message: Dict[str, Any]) -> str:
        """Generate hash for message deduplication."""
        key = f"{message.get('channel_id')}_{message.get('message_id')}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def should_retry(self, message: Dict[str, Any], max_retries: int = 3) -> bool:
        """Check if message should be retried."""
        msg_hash = self._hash_message(message)
        retry_count = self.retry_counts.get(msg_hash, 0)
        
        if retry_count < max_retries:
            self.retry_counts[msg_hash] = retry_count + 1
            return True
        return False


class ChannelManager:
    """Manages channel subscriptions and health monitoring."""
    
    def __init__(self):
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.health_stats: Dict[str, Dict[str, Any]] = {}
        
    async def add_channel(self, client: TelegramClient, channel_id: str) -> bool:
        """Add and validate a channel."""
        if channel_id in self.blacklist:
            logger.warning(f"Channel {channel_id} is blacklisted")
            return False
        
        try:
            entity = await client.get_entity(channel_id)
            
            self.channels[channel_id] = {
                'entity': entity,
                'name': getattr(entity, 'title', channel_id),
                'subscribed_at': datetime.utcnow(),
                'message_count': 0,
                'last_message': None,
                'is_active': True
            }
            
            self.health_stats[channel_id] = {
                'last_check': datetime.utcnow(),
                'consecutive_failures': 0,
                'total_messages': 0,
                'signals_detected': 0,
                'accuracy_rate': 0.0
            }
            
            logger.info(f"Successfully subscribed to channel: {channel_id}")
            return True
            
        except ChannelPrivateError:
            logger.error(f"Cannot access private channel: {channel_id}")
            self.blacklist.add(channel_id)
            return False
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel_id}: {e}")
            return False
    
    def update_channel_stats(self, channel_id: str, is_signal: bool = False):
        """Update channel statistics."""
        if channel_id in self.channels:
            self.channels[channel_id]['message_count'] += 1
            self.channels[channel_id]['last_message'] = datetime.utcnow()
            
            if channel_id in self.health_stats:
                self.health_stats[channel_id]['total_messages'] += 1
                if is_signal:
                    self.health_stats[channel_id]['signals_detected'] += 1
                
                # Calculate accuracy rate
                total = self.health_stats[channel_id]['total_messages']
                signals = self.health_stats[channel_id]['signals_detected']
                if total > 0:
                    self.health_stats[channel_id]['accuracy_rate'] = signals / total
    
    async def check_channel_health(self, client: TelegramClient, channel_id: str) -> bool:
        """Check if channel is still accessible and active."""
        try:
            await client.get_entity(channel_id)
            
            if channel_id in self.health_stats:
                self.health_stats[channel_id]['last_check'] = datetime.utcnow()
                self.health_stats[channel_id]['consecutive_failures'] = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for channel {channel_id}: {e}")
            
            if channel_id in self.health_stats:
                self.health_stats[channel_id]['consecutive_failures'] += 1
                
                # Auto-blacklist after 5 consecutive failures
                if self.health_stats[channel_id]['consecutive_failures'] >= 5:
                    self.blacklist.add(channel_id)
                    self.channels[channel_id]['is_active'] = False
                    logger.warning(f"Channel {channel_id} blacklisted after repeated failures")
            
            return False
    
    def get_active_channels(self) -> List[str]:
        """Get list of active channels."""
        return [
            channel_id for channel_id, info in self.channels.items()
            if info['is_active'] and channel_id not in self.blacklist
        ]
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """Get comprehensive channel statistics."""
        return {
            'total_channels': len(self.channels),
            'active_channels': len(self.get_active_channels()),
            'blacklisted': len(self.blacklist),
            'channel_details': self.health_stats
        }


class EnhancedTelegramCollector:
    """Enhanced Telegram collector with connection pooling and advanced features."""
    
    def __init__(self):
        self.connection_pool: Optional[ConnectionPool] = None
        self.message_queue = MessageQueue(max_size=10000)
        self.channel_manager = ChannelManager()
        self.signal_detector = SignalDetector()
        self.is_running = False
        self.enabled = TELEGRAM_ENABLED
        self.batch_size = 10
        self.batch_timeout = 1.0  # seconds
        self.rate_limit_delay = 1.0  # Base delay for exponential backoff
        self.max_rate_limit_delay = 32.0  # Maximum delay
        
        # Media download settings
        self.media_download_path = Path("data/telegram_media")
        self.media_download_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'signals_detected': 0,
            'media_downloaded': 0,
            'errors': 0,
            'rate_limits': 0,
            'start_time': None
        }
        
        if self.enabled:
            self.connection_pool = ConnectionPool(size=10)
        else:
            logger.warning("Telegram collector disabled: missing credentials")
    
    async def start(self):
        """Start the Telegram collector with connection pool."""
        if not self.enabled:
            logger.info("Telegram collector skipped: no credentials configured")
            return
            
        try:
            # Initialize connection pool
            await self.connection_pool.initialize(
                settings.telegram_api_id,
                settings.telegram_api_hash,
                settings.telegram_session_name
            )
            
            # Get primary client for initial setup
            primary_client = await self.connection_pool.acquire()
            
            try:
                # Start primary client with phone authentication
                await primary_client.start(phone=settings.telegram_phone_number)
                logger.info("Primary Telegram client started")
                
                # Subscribe to channels
                for channel in settings.telegram_channels_list:
                    await self.channel_manager.add_channel(primary_client, channel)
                
                # Register event handlers on all connections
                for client in self.connection_pool.connections:
                    client.add_event_handler(
                        self.handle_new_message,
                        events.NewMessage(chats=self.channel_manager.get_active_channels())
                    )
                
            finally:
                await self.connection_pool.release(primary_client)
            
            self.is_running = True
            self.stats['start_time'] = datetime.utcnow()
            
            # Start background tasks
            asyncio.create_task(self.process_messages())
            asyncio.create_task(self.monitor_channels())
            asyncio.create_task(self.process_dead_letters())
            
            logger.info(f"Telegram collector started with {self.connection_pool.size} connections")
            
        except SessionPasswordNeededError:
            logger.error("2FA is enabled. Please provide the password.")
            raise
        except Exception as e:
            logger.error(f"Failed to start Telegram collector: {e}")
            raise
    
    async def stop(self):
        """Stop the Telegram collector."""
        self.is_running = False
        if self.connection_pool and self.enabled:
            await self.connection_pool.close_all()
            logger.info("Telegram collector stopped")
            
            # Log final statistics
            logger.info(f"Final statistics: {self.stats}")
            logger.info(f"Channel statistics: {self.channel_manager.get_channel_stats()}")
    
    async def handle_new_message(self, event):
        """Handle incoming Telegram messages with priority detection."""
        try:
            message: Message = event.message
            
            # Extract comprehensive message data
            message_data = {
                "channel_id": event.chat_id,
                "message_id": message.id,
                "content": message.text or "",
                "author": message.sender_id if message.sender_id else "Unknown",
                "timestamp": message.date,
                "has_media": bool(message.media),
                "media_urls": [],
                "reply_to": message.reply_to_msg_id if message.reply_to else None,
                "forwards": message.forwards if hasattr(message, 'forwards') else 0,
                "views": message.views if hasattr(message, 'views') else 0,
                "reactions": getattr(message, 'reactions', None)
            }
            
            # Handle media
            if message.media:
                if isinstance(message.media, (MessageMediaPhoto, MessageMediaDocument)):
                    message_data["media_urls"].append({
                        "type": "photo" if isinstance(message.media, MessageMediaPhoto) else "document",
                        "id": message.media.id if hasattr(message.media, 'id') else None,
                        "size": getattr(message.media, 'size', None)
                    })
            
            # Determine priority
            priority = self._determine_priority(message_data)
            
            # Add to processing queue with priority
            await self.message_queue.put(message_data, priority)
            
            # Update channel statistics
            self.channel_manager.update_channel_stats(str(event.chat_id))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.stats['errors'] += 1
    
    def _determine_priority(self, message_data: Dict[str, Any]) -> MessagePriority:
        """Determine message priority based on content and metadata."""
        content_lower = message_data['content'].lower()
        
        # High priority keywords
        high_priority_keywords = [
            'urgent', 'alert', 'buy now', 'sell now', 'breaking', 
            'immediate', 'entry', 'stop loss', 'take profit'
        ]
        if any(keyword in content_lower for keyword in high_priority_keywords):
            return MessagePriority.HIGH
        
        # Medium priority for messages with media or high engagement
        if (message_data['has_media'] or 
            message_data.get('views', 0) > 1000 or
            message_data.get('forwards', 0) > 10):
            return MessagePriority.MEDIUM
        
        return MessagePriority.LOW
    
    async def process_messages(self):
        """Process messages from priority queues with batching."""
        while self.is_running:
            try:
                batch = []
                start_time = asyncio.get_event_loop().time()
                
                while len(batch) < self.batch_size:
                    remaining_time = self.batch_timeout - (asyncio.get_event_loop().time() - start_time)
                    if remaining_time <= 0:
                        break
                    
                    message = await self.message_queue.get()
                    if message:
                        batch.append(message)
                    else:
                        await asyncio.sleep(0.1)
                
                if batch:
                    # Acquire connection from pool for processing
                    client = await self.connection_pool.acquire()
                    try:
                        await self.process_batch(batch, client)
                    finally:
                        await self.connection_pool.release(client)
                
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def process_batch(self, messages: List[Dict[str, Any]], client: TelegramClient):
        """Process a batch of messages with retry logic."""
        async with get_db_context() as db:
            for message_data in messages:
                try:
                    # Check if message is a signal
                    is_signal = await self.signal_detector.detect(message_data["content"])
                    message_data["is_signal"] = is_signal
                    
                    # Process media if present
                    if message_data["has_media"] and message_data["media_urls"]:
                        for media_info in message_data["media_urls"]:
                            media_path = await self._download_media(client, message_data, media_info)
                            if media_path:
                                media_info['local_path'] = media_path
                                self.stats['media_downloaded'] += 1
                    
                    # Save to database
                    telegram_message = TelegramMessage(
                        channel_id=message_data["channel_id"],
                        message_id=message_data["message_id"],
                        content=message_data["content"],
                        author=str(message_data["author"]),
                        timestamp=message_data["timestamp"],
                        has_media=message_data["has_media"],
                        media_urls=message_data["media_urls"],
                        processed=True,
                        is_signal=is_signal
                    )
                    db.add(telegram_message)
                    
                    # Cache signal for quick access
                    if is_signal:
                        cache_key = f"signal:{message_data['channel_id']}:{message_data['message_id']}"
                        await signal_cache.set_signal(cache_key, message_data)
                    
                    # Publish to Kafka if it's a signal
                    if is_signal:
                        await KafkaEventPublisher.publish_signal({
                            "channel_id": message_data["channel_id"],
                            "message_id": message_data["message_id"],
                            "content": message_data["content"],
                            "timestamp": message_data["timestamp"].isoformat(),
                            "media_urls": message_data.get("media_urls", []),
                            "metadata": {
                                "views": message_data.get("views", 0),
                                "forwards": message_data.get("forwards", 0)
                            }
                        })
                        
                        self.stats['signals_detected'] += 1
                        self.channel_manager.update_channel_stats(
                            str(message_data['channel_id']), 
                            is_signal=True
                        )
                        logger.info(f"Signal detected from channel {message_data['channel_id']}")
                    
                    self.stats['messages_processed'] += 1
                    
                except FloodWaitError as e:
                    self.stats['rate_limits'] += 1
                    await self.handle_rate_limit(e.seconds)
                    
                    # Retry the message if possible
                    if self.message_queue.should_retry(message_data):
                        await self.message_queue.put(message_data, MessagePriority.HIGH)
                    else:
                        self.message_queue.add_to_dead_letter(message_data, str(e))
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.stats['errors'] += 1
                    
                    # Add to dead letter queue if max retries exceeded
                    if not self.message_queue.should_retry(message_data):
                        self.message_queue.add_to_dead_letter(message_data, str(e))
            
            await db.commit()
    
    async def _download_media(self, client: TelegramClient, message_data: Dict, media_info: Dict) -> Optional[str]:
        """Download media from message."""
        try:
            channel_id = message_data['channel_id']
            message_id = message_data['message_id']
            
            # Get the message object
            message = await client.get_messages(channel_id, ids=message_id)
            
            if message and message.media:
                # Create unique filename
                timestamp = message_data['timestamp'].strftime('%Y%m%d_%H%M%S')
                filename = f"{channel_id}_{message_id}_{timestamp}"
                
                # Download media
                path = await client.download_media(
                    message.media,
                    file=self.media_download_path / filename
                )
                
                logger.info(f"Downloaded media: {path}")
                return str(path)
                
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
        
        return None
    
    async def handle_rate_limit(self, wait_time: int):
        """Handle Telegram rate limiting with exponential backoff."""
        actual_wait = min(wait_time, self.max_rate_limit_delay)
        logger.warning(f"Rate limited. Waiting {actual_wait} seconds...")
        await asyncio.sleep(actual_wait)
    
    async def monitor_channels(self):
        """Monitor channel health and auto-reconnect."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                client = await self.connection_pool.acquire()
                try:
                    for channel_id in self.channel_manager.get_active_channels():
                        await self.channel_manager.check_channel_health(client, channel_id)
                finally:
                    await self.connection_pool.release(client)
                    
            except Exception as e:
                logger.error(f"Channel monitoring error: {e}")
    
    async def process_dead_letters(self):
        """Process messages from dead letter queue."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Process every minute
                
                if self.message_queue.dead_letter:
                    dead_letter_batch = []
                    
                    # Get up to 10 messages from dead letter queue
                    for _ in range(min(10, len(self.message_queue.dead_letter))):
                        if self.message_queue.dead_letter:
                            dead_letter_batch.append(self.message_queue.dead_letter.popleft())
                    
                    if dead_letter_batch:
                        logger.info(f"Processing {len(dead_letter_batch)} dead letter messages")
                        
                        # Log dead letters for analysis
                        for msg in dead_letter_batch:
                            logger.error(f"Dead letter message: {msg}")
                        
            except Exception as e:
                logger.error(f"Dead letter processing error: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            'collector_stats': {
                **self.stats,
                'uptime_seconds': uptime,
                'messages_per_second': self.stats['messages_processed'] / uptime if uptime else 0
            },
            'channel_stats': self.channel_manager.get_channel_stats(),
            'queue_stats': {
                'high_priority': self.message_queue.high_priority.qsize(),
                'medium_priority': self.message_queue.medium_priority.qsize(),
                'low_priority': self.message_queue.low_priority.qsize(),
                'dead_letter_count': len(self.message_queue.dead_letter)
            }
        }


# Global collector instance
enhanced_collector = EnhancedTelegramCollector()


async def start_enhanced_telegram_collector():
    """Start the enhanced Telegram collector."""
    if TELEGRAM_ENABLED:
        await enhanced_collector.start()
    else:
        logger.info("Enhanced Telegram collector not started: credentials not configured")


async def stop_enhanced_telegram_collector():
    """Stop the enhanced Telegram collector."""
    if TELEGRAM_ENABLED:
        await enhanced_collector.stop()


async def get_collector_statistics():
    """Get collector statistics."""
    return await enhanced_collector.get_statistics()
