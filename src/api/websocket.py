"""
WebSocket endpoints for real-time updates.
"""

from typing import Dict, Set, Any
import json
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from src.core.kafka_client import create_consumer
from src.config.settings import settings

router = APIRouter()
logger = structlog.get_logger()

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "signals": set(),
            "alerts": set(),
            "performance": set()
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].add(websocket)
        logger.info(f"WebSocket connected to {channel}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        self.active_connections[channel].discard(websocket)
        logger.info(f"WebSocket disconnected from {channel}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, channel: str):
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")

manager = ConnectionManager()


@router.websocket("/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates."""
    await manager.connect(websocket, "signals")
    
    try:
        # Create Kafka consumer for this connection
        consumer = await create_consumer(
            [settings.kafka_topic_signals],
            f"ws-signals-{id(websocket)}",
            auto_offset_reset='latest'
        )
        
        # Listen for messages
        async for msg in consumer:
            await manager.send_personal_message(
                json.dumps({
                    "type": "signal",
                    "data": msg.value
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "signals")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, "signals")
    finally:
        if 'consumer' in locals():
            await consumer.stop()


@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts."""
    await manager.connect(websocket, "alerts")
    
    try:
        # Create Kafka consumer for alerts
        consumer = await create_consumer(
            [f"{settings.kafka_topic_signals}-alerts"],
            f"ws-alerts-{id(websocket)}",
            auto_offset_reset='latest'
        )
        
        # Listen for messages
        async for msg in consumer:
            await manager.send_personal_message(
                json.dumps({
                    "type": "alert",
                    "data": msg.value
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "alerts")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, "alerts")
    finally:
        if 'consumer' in locals():
            await consumer.stop()


@router.websocket("/performance")
async def websocket_performance(websocket: WebSocket):
    """WebSocket endpoint for real-time performance updates."""
    await manager.connect(websocket, "performance")
    
    try:
        while True:
            # Send heartbeat every 30 seconds
            await asyncio.sleep(30)
            await manager.send_personal_message(
                json.dumps({
                    "type": "heartbeat",
                    "timestamp": asyncio.get_event_loop().time()
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "performance")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, "performance")
