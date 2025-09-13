"""
Telegram data collector for signal extraction.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from telethon import TelegramClient, events
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
from telethon.errors import SessionPasswordNeededError, FloodWaitError
import structlog

from src.config.settings import settings
from src.core.database import get_db_context
from src.core.kafka_client import KafkaEventPublisher
from src.core.redis_client import signal_cache
from src.models import TelegramMessage
from src.core.signal_detector import SignalDetector

logger = structlog.get_logger()


class TelegramCollector:
    """Telegram message collector and processor."""
    
    def __init__(self):
        self.client = TelegramClient(
            settings.telegram_session_name,
            settings.telegram_api_id,
            settings.telegram_api_hash
        )
        self.channels = settings.telegram_channels_list
        self.signal_detector = SignalDetector()
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.is_running = False
    
    async def start(self):
        """Start the Telegram client and connect."""
        try:
            await self.client.start(phone=settings.telegram_phone_number)
            logger.info("Telegram client started")
            
            # Subscribe to channels
            for channel in self.channels:
                try:
                    await self.client.get_entity(channel)
                    logger.info(f"Subscribed to channel: {channel}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {channel}: {e}")
            
            # Register event handlers
            self.client.add_event_handler(
                self.handle_new_message,
                events.NewMessage(chats=self.channels)
            )
            
            self.is_running = True
            
            # Start message processor
            asyncio.create_task(self.process_messages())
            
        except SessionPasswordNeededError:
            logger.error("2FA is enabled. Please provide the password.")
            raise
        except Exception as e:
            logger.error(f"Failed to start Telegram client: {e}")
            raise
    
    async def stop(self):
        """Stop the Telegram client."""
        self.is_running = False
        await self.client.disconnect()
        logger.info("Telegram client stopped")
    
    async def handle_new_message(self, event):
        """Handle incoming Telegram messages."""
        try:
            message: Message = event.message
            
            # Extract message data
            message_data = {
                "channel_id": event.chat_id,
                "message_id": message.id,
                "content": message.text or "",
                "author": message.sender_id if message.sender_id else "Unknown",
                "timestamp": message.date,
                "has_media": bool(message.media),
                "media_urls": []
            }
            
            # Handle media
            if message.media:
                if isinstance(message.media, (MessageMediaPhoto, MessageMediaDocument)):
                    # Download and process media later
                    message_data["media_urls"].append({
                        "type": "photo" if isinstance(message.media, MessageMediaPhoto) else "document",
                        "id": message.media.id if hasattr(message.media, 'id') else None
                    })
            
            # Add to processing queue
            await self.message_queue.put(message_data)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def process_messages(self):
        """Process messages from the queue."""
        while self.is_running:
            try:
                # Batch processing
                batch = []
                batch_timeout = 1.0  # seconds
                start_time = asyncio.get_event_loop().time()
                
                while len(batch) < 10:
                    try:
                        remaining_time = batch_timeout - (asyncio.get_event_loop().time() - start_time)
                        if remaining_time <= 0:
                            break
                        
                        message = await asyncio.wait_for(
                            self.message_queue.get(),
                            timeout=remaining_time
                        )
                        batch.append(message)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self.process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                await asyncio.sleep(1)
    
    async def process_batch(self, messages: List[Dict[str, Any]]):
        """Process a batch of messages."""
        async with get_db_context() as db:
            for message_data in messages:
                try:
                    # Check if message is a signal
                    is_signal = await self.signal_detector.detect(message_data["content"])
                    message_data["is_signal"] = is_signal
                    
                    # Save to database
                    telegram_message = TelegramMessage(
                        channel_id=message_data["channel_id"],
                        message_id=message_data["message_id"],
                        content=message_data["content"],
                        author=str(message_data["author"]),
                        timestamp=message_data["timestamp"],
                        has_media=message_data["has_media"],
                        media_urls=message_data["media_urls"],
                        processed=False,
                        is_signal=is_signal
                    )
                    db.add(telegram_message)
                    
                    # If it's a signal, publish to Kafka
                    if is_signal:
                        await KafkaEventPublisher.publish_signal({
                            "channel_id": message_data["channel_id"],
                            "message_id": message_data["message_id"],
                            "content": message_data["content"],
                            "timestamp": message_data["timestamp"].isoformat()
                        })
                        
                        logger.info(f"Signal detected from channel {message_data['channel_id']}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
            
            await db.commit()
    
    async def handle_rate_limit(self, wait_time: int):
        """Handle Telegram rate limiting."""
        logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
        await asyncio.sleep(wait_time)
    
    async def reconnect(self):
        """Reconnect to Telegram if disconnected."""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                if not self.client.is_connected():
                    await self.client.connect()
                    logger.info("Reconnected to Telegram")
                    return
            except FloodWaitError as e:
                await self.handle_rate_limit(e.seconds)
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))
        
        raise Exception("Failed to reconnect after maximum retries")


# Global collector instance
telegram_collector = TelegramCollector()


async def start_telegram_collector():
    """Start the Telegram collector."""
    await telegram_collector.start()


async def stop_telegram_collector():
    """Stop the Telegram collector."""
    await telegram_collector.stop()
