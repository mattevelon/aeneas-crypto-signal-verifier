"""
API endpoints for data collector management and statistics.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import structlog

from src.config.settings import settings
from src.data_ingestion.telegram_collector_enhanced import (
    enhanced_collector,
    get_collector_statistics
)
from src.data_ingestion.image_processor import image_processor

logger = structlog.get_logger()

router = APIRouter(prefix="/collector", tags=["collector"])


class CollectorStatus(BaseModel):
    """Collector status response model."""
    is_running: bool
    enabled: bool
    start_time: datetime = None
    uptime_seconds: float = None
    connection_pool_size: int
    active_channels: int
    messages_processed: int
    signals_detected: int
    errors: int


class ChannelStats(BaseModel):
    """Channel statistics model."""
    channel_id: str
    channel_name: str
    is_active: bool
    message_count: int
    signals_detected: int
    accuracy_rate: float
    last_message: datetime = None
    subscribed_at: datetime


class CollectorConfig(BaseModel):
    """Collector configuration model."""
    batch_size: int = 10
    batch_timeout: float = 1.0
    max_rate_limit_delay: float = 32.0
    connection_pool_size: int = 10
    message_queue_size: int = 10000


@router.get("/status", response_model=CollectorStatus)
async def get_collector_status() -> CollectorStatus:
    """Get current collector status."""
    try:
        if not settings.has_telegram_credentials:
            return CollectorStatus(
                is_running=False,
                enabled=False,
                connection_pool_size=0,
                active_channels=0,
                messages_processed=0,
                signals_detected=0,
                errors=0
            )
        
        stats = await get_collector_statistics()
        collector_stats = stats.get('collector_stats', {})
        channel_stats = stats.get('channel_stats', {})
        
        return CollectorStatus(
            is_running=enhanced_collector.is_running,
            enabled=enhanced_collector.enabled,
            start_time=collector_stats.get('start_time'),
            uptime_seconds=collector_stats.get('uptime_seconds'),
            connection_pool_size=enhanced_collector.connection_pool.size if enhanced_collector.connection_pool else 0,
            active_channels=channel_stats.get('active_channels', 0),
            messages_processed=collector_stats.get('messages_processed', 0),
            signals_detected=collector_stats.get('signals_detected', 0),
            errors=collector_stats.get('errors', 0)
        )
        
    except Exception as e:
        logger.error(f"Error getting collector status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics() -> Dict[str, Any]:
    """Get comprehensive collector statistics."""
    try:
        if not settings.has_telegram_credentials:
            return {
                "error": "Telegram credentials not configured",
                "enabled": False
            }
        
        stats = await get_collector_statistics()
        
        # Add additional computed metrics
        if stats['collector_stats'].get('uptime_seconds'):
            uptime = stats['collector_stats']['uptime_seconds']
            stats['collector_stats']['messages_per_minute'] = (
                stats['collector_stats']['messages_processed'] / (uptime / 60)
                if uptime > 0 else 0
            )
            stats['collector_stats']['signals_per_hour'] = (
                stats['collector_stats']['signals_detected'] / (uptime / 3600)
                if uptime > 0 else 0
            )
            stats['collector_stats']['error_rate'] = (
                stats['collector_stats']['errors'] / stats['collector_stats']['messages_processed']
                if stats['collector_stats']['messages_processed'] > 0 else 0
            )
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting collector statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channels", response_model=List[ChannelStats])
async def get_channel_statistics() -> List[ChannelStats]:
    """Get per-channel statistics."""
    try:
        if not settings.has_telegram_credentials:
            return []
        
        stats = await get_collector_statistics()
        channel_stats = stats.get('channel_stats', {})
        channel_details = channel_stats.get('channel_details', {})
        
        result = []
        for channel_id, details in channel_details.items():
            # Get channel info from channel manager
            channel_info = enhanced_collector.channel_manager.channels.get(channel_id, {})
            
            result.append(ChannelStats(
                channel_id=channel_id,
                channel_name=channel_info.get('name', channel_id),
                is_active=channel_info.get('is_active', False),
                message_count=channel_info.get('message_count', 0),
                signals_detected=details.get('signals_detected', 0),
                accuracy_rate=details.get('accuracy_rate', 0.0),
                last_message=channel_info.get('last_message'),
                subscribed_at=channel_info.get('subscribed_at', datetime.utcnow())
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting channel statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue-status")
async def get_queue_status() -> Dict[str, Any]:
    """Get message queue status."""
    try:
        if not settings.has_telegram_credentials:
            return {
                "error": "Telegram credentials not configured",
                "enabled": False
            }
        
        stats = await get_collector_statistics()
        queue_stats = stats.get('queue_stats', {})
        
        # Add queue health metrics
        total_queued = (
            queue_stats.get('high_priority', 0) +
            queue_stats.get('medium_priority', 0) +
            queue_stats.get('low_priority', 0)
        )
        
        queue_health = "healthy"
        if total_queued > 5000:
            queue_health = "overloaded"
        elif total_queued > 1000:
            queue_health = "busy"
        
        return {
            **queue_stats,
            'total_queued': total_queued,
            'queue_health': queue_health,
            'max_queue_size': enhanced_collector.message_queue.max_size
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dead-letters")
async def get_dead_letters(limit: int = Query(default=10, le=100)) -> List[Dict[str, Any]]:
    """Get recent dead letter messages."""
    try:
        if not settings.has_telegram_credentials:
            return []
        
        dead_letters = list(enhanced_collector.message_queue.dead_letter)
        
        # Return most recent messages up to limit
        return dead_letters[-limit:] if dead_letters else []
        
    except Exception as e:
        logger.error(f"Error getting dead letters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/channels/{channel_id}/subscribe")
async def subscribe_to_channel(channel_id: str) -> Dict[str, Any]:
    """Subscribe to a new Telegram channel."""
    try:
        if not settings.has_telegram_credentials:
            raise HTTPException(status_code=400, detail="Telegram credentials not configured")
        
        if not enhanced_collector.is_running:
            raise HTTPException(status_code=400, detail="Collector is not running")
        
        # Acquire a connection from pool
        client = await enhanced_collector.connection_pool.acquire()
        try:
            success = await enhanced_collector.channel_manager.add_channel(client, channel_id)
            
            if success:
                # Update event handlers for new channel
                for conn in enhanced_collector.connection_pool.connections:
                    conn.add_event_handler(
                        enhanced_collector.handle_new_message,
                        events.NewMessage(chats=enhanced_collector.channel_manager.get_active_channels())
                    )
                
                return {
                    "success": True,
                    "message": f"Successfully subscribed to channel {channel_id}",
                    "channel_id": channel_id
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to subscribe to channel {channel_id}",
                    "channel_id": channel_id
                }
                
        finally:
            await enhanced_collector.connection_pool.release(client)
            
    except Exception as e:
        logger.error(f"Error subscribing to channel {channel_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/channels/{channel_id}/unsubscribe")
async def unsubscribe_from_channel(channel_id: str) -> Dict[str, Any]:
    """Unsubscribe from a Telegram channel."""
    try:
        if not settings.has_telegram_credentials:
            raise HTTPException(status_code=400, detail="Telegram credentials not configured")
        
        if channel_id in enhanced_collector.channel_manager.channels:
            # Mark channel as inactive
            enhanced_collector.channel_manager.channels[channel_id]['is_active'] = False
            
            # Add to blacklist
            enhanced_collector.channel_manager.blacklist.add(channel_id)
            
            return {
                "success": True,
                "message": f"Successfully unsubscribed from channel {channel_id}",
                "channel_id": channel_id
            }
        else:
            return {
                "success": False,
                "message": f"Channel {channel_id} not found",
                "channel_id": channel_id
            }
            
    except Exception as e:
        logger.error(f"Error unsubscribing from channel {channel_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
async def restart_collector() -> Dict[str, Any]:
    """Restart the collector."""
    try:
        if not settings.has_telegram_credentials:
            raise HTTPException(status_code=400, detail="Telegram credentials not configured")
        
        # Stop if running
        if enhanced_collector.is_running:
            await enhanced_collector.stop()
            await asyncio.sleep(2)  # Wait for clean shutdown
        
        # Start again
        await enhanced_collector.start()
        
        return {
            "success": True,
            "message": "Collector restarted successfully",
            "is_running": enhanced_collector.is_running
        }
        
    except Exception as e:
        logger.error(f"Error restarting collector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=CollectorConfig)
async def update_collector_config(config: CollectorConfig) -> CollectorConfig:
    """Update collector configuration."""
    try:
        if not settings.has_telegram_credentials:
            raise HTTPException(status_code=400, detail="Telegram credentials not configured")
        
        # Update configuration
        enhanced_collector.batch_size = config.batch_size
        enhanced_collector.batch_timeout = config.batch_timeout
        enhanced_collector.max_rate_limit_delay = config.max_rate_limit_delay
        
        # Note: connection_pool_size and message_queue_size require restart to take effect
        
        logger.info(f"Collector configuration updated: {config}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error updating collector config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image-processing/stats")
async def get_image_processing_stats() -> Dict[str, Any]:
    """Get image processing statistics."""
    try:
        # Get cached processing results count
        cache_dir = image_processor.cache_dir
        processed_count = len(list(cache_dir.glob("*.json")))
        
        return {
            "processed_images": processed_count,
            "cache_directory": str(cache_dir),
            "ocr_engines_available": {
                "google_vision": image_processor.ocr_processor.google_vision_client is not None,
                "easyocr": image_processor.ocr_processor.easyocr_reader is not None,
                "tesseract": image_processor.ocr_processor.tesseract_available
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting image processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
