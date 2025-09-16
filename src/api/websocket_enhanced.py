"""
Enhanced WebSocket endpoints with subscription management and connection recovery.
"""

import json
import asyncio
import time
from typing import Dict, Set, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from uuid import uuid4
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from pydantic import BaseModel, Field
import structlog

from src.core.kafka_client import create_consumer
from src.core.redis_client import get_redis
from src.config.settings import get_settings
from src.api.auth import get_current_user

router = APIRouter()
logger = structlog.get_logger()
settings = get_settings()


class SubscriptionType(str, Enum):
    """Types of subscriptions available."""
    SIGNALS = "signals"
    ALERTS = "alerts"
    PERFORMANCE = "performance"
    CHANNELS = "channels"
    STATISTICS = "statistics"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Any
    metadata: Optional[Dict[str, Any]] = None


class SubscriptionRequest(BaseModel):
    """Subscription request format."""
    type: SubscriptionType
    filters: Optional[Dict[str, Any]] = None
    priority_filter: Optional[List[MessagePriority]] = None


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""
    connection_id: str
    user_id: Optional[str] = None
    connected_at: datetime
    last_activity: datetime
    subscriptions: List[SubscriptionType]
    message_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0


class EnhancedConnectionManager:
    """Enhanced WebSocket connection manager with subscription and recovery support."""
    
    def __init__(self):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[SubscriptionType, Set[str]] = defaultdict(set)
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.redis = get_redis()
        self.recovery_window = timedelta(minutes=5)
        
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Accept WebSocket connection and initialize connection state.
        
        Returns:
            Connection ID for tracking
        """
        await websocket.accept()
        
        # Generate or recover connection ID
        if not connection_id:
            connection_id = str(uuid4())
        
        # Check for existing connection recovery
        recovered = await self._recover_connection(connection_id)
        
        # Initialize connection info
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "subscriptions": set(),
            "filters": {},
            "message_count": recovered.get("message_count", 0) if recovered else 0,
            "error_count": 0,
            "reconnect_count": recovered.get("reconnect_count", 0) + 1 if recovered else 0
        }
        
        # Initialize message queue
        self.message_queues[connection_id] = asyncio.Queue(maxsize=1000)
        
        # Start heartbeat
        self.heartbeat_tasks[connection_id] = asyncio.create_task(
            self._heartbeat_loop(connection_id)
        )
        
        # Store connection info in Redis for recovery
        await self._store_connection_info(connection_id)
        
        logger.info(
            "WebSocket connected",
            connection_id=connection_id,
            user_id=user_id,
            reconnect_count=self.connections[connection_id]["reconnect_count"]
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect and cleanup connection."""
        if connection_id not in self.connections:
            return
        
        # Cancel heartbeat
        if connection_id in self.heartbeat_tasks:
            self.heartbeat_tasks[connection_id].cancel()
            del self.heartbeat_tasks[connection_id]
        
        # Remove from subscriptions
        for sub_type in self.connections[connection_id]["subscriptions"]:
            self.subscriptions[sub_type].discard(connection_id)
        
        # Store final state for recovery
        await self._store_connection_info(connection_id, final=True)
        
        # Cleanup
        del self.connections[connection_id]
        if connection_id in self.message_queues:
            del self.message_queues[connection_id]
        
        logger.info("WebSocket disconnected", connection_id=connection_id)
    
    async def subscribe(
        self,
        connection_id: str,
        subscription: SubscriptionRequest
    ) -> bool:
        """Subscribe connection to specific event types."""
        if connection_id not in self.connections:
            return False
        
        # Add subscription
        self.connections[connection_id]["subscriptions"].add(subscription.type)
        self.subscriptions[subscription.type].add(connection_id)
        
        # Store filters
        if subscription.filters:
            if "filters" not in self.connections[connection_id]:
                self.connections[connection_id]["filters"] = {}
            self.connections[connection_id]["filters"][subscription.type] = subscription.filters
        
        # Store priority filter
        if subscription.priority_filter:
            self.connections[connection_id]["priority_filter"] = subscription.priority_filter
        
        await self._store_connection_info(connection_id)
        
        logger.info(
            "Subscription added",
            connection_id=connection_id,
            subscription_type=subscription.type
        )
        
        return True
    
    async def unsubscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType
    ) -> bool:
        """Unsubscribe connection from specific event type."""
        if connection_id not in self.connections:
            return False
        
        self.connections[connection_id]["subscriptions"].discard(subscription_type)
        self.subscriptions[subscription_type].discard(connection_id)
        
        if subscription_type in self.connections[connection_id].get("filters", {}):
            del self.connections[connection_id]["filters"][subscription_type]
        
        await self._store_connection_info(connection_id)
        
        return True
    
    async def send_message(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return False
        
        try:
            # Check priority filter
            priority_filter = self.connections[connection_id].get("priority_filter")
            if priority_filter and message.priority not in priority_filter:
                return False
            
            # Queue message
            await self.message_queues[connection_id].put(message)
            
            # Update stats
            self.connections[connection_id]["message_count"] += 1
            self.connections[connection_id]["last_activity"] = datetime.utcnow()
            
            return True
            
        except asyncio.QueueFull:
            logger.warning(
                "Message queue full",
                connection_id=connection_id,
                message_type=message.type
            )
            return False
        except Exception as e:
            logger.error(
                "Error sending message",
                connection_id=connection_id,
                error=str(e)
            )
            self.connections[connection_id]["error_count"] += 1
            return False
    
    async def broadcast(
        self,
        subscription_type: SubscriptionType,
        message: WebSocketMessage,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Broadcast message to all subscribed connections."""
        for connection_id in self.subscriptions[subscription_type]:
            # Check connection filters
            if filters and not self._match_filters(connection_id, subscription_type, filters):
                continue
            
            await self.send_message(connection_id, message)
    
    async def process_messages(self, connection_id: str):
        """Process queued messages for a connection."""
        if connection_id not in self.connections:
            return
        
        websocket = self.connections[connection_id]["websocket"]
        queue = self.message_queues[connection_id]
        
        try:
            while True:
                # Get message from queue
                message = await queue.get()
                
                # Send to WebSocket
                await websocket.send_text(message.json())
                
                # Update activity
                self.connections[connection_id]["last_activity"] = datetime.utcnow()
                
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
        except Exception as e:
            logger.error(
                "Error processing messages",
                connection_id=connection_id,
                error=str(e)
            )
            self.connections[connection_id]["error_count"] += 1
            if self.connections[connection_id]["error_count"] > 10:
                await self.disconnect(connection_id)
    
    async def _heartbeat_loop(self, connection_id: str):
        """Send periodic heartbeat messages."""
        while connection_id in self.connections:
            try:
                await asyncio.sleep(30)  # 30 second heartbeat
                
                heartbeat = WebSocketMessage(
                    type="heartbeat",
                    priority=MessagePriority.LOW,
                    data={
                        "connection_id": connection_id,
                        "server_time": datetime.utcnow().isoformat()
                    }
                )
                
                await self.send_message(connection_id, heartbeat)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Heartbeat error",
                    connection_id=connection_id,
                    error=str(e)
                )
    
    async def _store_connection_info(self, connection_id: str, final: bool = False):
        """Store connection info in Redis for recovery."""
        if connection_id not in self.connections:
            return
        
        info = {
            "user_id": self.connections[connection_id]["user_id"],
            "connected_at": self.connections[connection_id]["connected_at"].isoformat(),
            "last_activity": self.connections[connection_id]["last_activity"].isoformat(),
            "subscriptions": list(self.connections[connection_id]["subscriptions"]),
            "filters": self.connections[connection_id].get("filters", {}),
            "message_count": self.connections[connection_id]["message_count"],
            "reconnect_count": self.connections[connection_id]["reconnect_count"]
        }
        
        if final:
            info["disconnected_at"] = datetime.utcnow().isoformat()
        
        await self.redis.setex(
            f"ws:connection:{connection_id}",
            int(self.recovery_window.total_seconds()),
            json.dumps(info)
        )
    
    async def _recover_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Recover connection state from Redis."""
        data = await self.redis.get(f"ws:connection:{connection_id}")
        if data:
            return json.loads(data)
        return None
    
    def _match_filters(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if message matches connection filters."""
        conn_filters = self.connections[connection_id].get("filters", {}).get(subscription_type, {})
        
        for key, value in filters.items():
            if key in conn_filters and conn_filters[key] != value:
                return False
        
        return True
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a connection."""
        if connection_id not in self.connections:
            return None
        
        conn = self.connections[connection_id]
        return ConnectionInfo(
            connection_id=connection_id,
            user_id=conn["user_id"],
            connected_at=conn["connected_at"],
            last_activity=conn["last_activity"],
            subscriptions=list(conn["subscriptions"]),
            message_count=conn["message_count"],
            error_count=conn["error_count"],
            reconnect_count=conn["reconnect_count"]
        )
    
    def get_all_connections(self) -> List[ConnectionInfo]:
        """Get information about all active connections."""
        return [
            self.get_connection_info(conn_id)
            for conn_id in self.connections
        ]


# Global connection manager instance
manager = EnhancedConnectionManager()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint with authentication and subscription management.
    """
    # Optional authentication
    user_id = None
    if token:
        try:
            # Validate token and get user
            # For now, just use token as user_id
            user_id = token
        except Exception:
            await websocket.close(code=1008, reason="Invalid authentication")
            return
    
    # Connect
    connection_id = await manager.connect(websocket, connection_id, user_id)
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connected",
        "connection_id": connection_id,
        "timestamp": datetime.utcnow().isoformat()
    }))
    
    # Create tasks for handling
    receive_task = asyncio.create_task(handle_receive(websocket, connection_id))
    send_task = asyncio.create_task(manager.process_messages(connection_id))
    
    try:
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
    except Exception as e:
        logger.error(
            "WebSocket error",
            connection_id=connection_id,
            error=str(e)
        )
    finally:
        await manager.disconnect(connection_id)


async def handle_receive(websocket: WebSocket, connection_id: str):
    """Handle incoming messages from WebSocket."""
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle message types
            if message["type"] == "subscribe":
                subscription = SubscriptionRequest(**message.get("data", {}))
                success = await manager.subscribe(connection_id, subscription)
                
                await websocket.send_text(json.dumps({
                    "type": "subscription_result",
                    "success": success,
                    "subscription": subscription.dict()
                }))
                
            elif message["type"] == "unsubscribe":
                subscription_type = SubscriptionType(message["data"]["type"])
                success = await manager.unsubscribe(connection_id, subscription_type)
                
                await websocket.send_text(json.dumps({
                    "type": "unsubscribe_result",
                    "success": success,
                    "subscription_type": subscription_type
                }))
                
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(
            "Error handling received message",
            connection_id=connection_id,
            error=str(e)
        )


@router.get("/connections")
async def get_connections():
    """Get information about all active WebSocket connections."""
    connections = manager.get_all_connections()
    return {
        "total": len(connections),
        "connections": [conn.dict() for conn in connections]
    }


@router.get("/connections/{connection_id}")
async def get_connection(connection_id: str):
    """Get information about a specific WebSocket connection."""
    info = manager.get_connection_info(connection_id)
    if not info:
        return {"error": "Connection not found"}
    return info.dict()


# Example usage for broadcasting signals
async def broadcast_signal_update(signal_data: Dict[str, Any]):
    """Broadcast signal update to all subscribed connections."""
    message = WebSocketMessage(
        type="signal_update",
        priority=MessagePriority.HIGH,
        data=signal_data
    )
    
    await manager.broadcast(SubscriptionType.SIGNALS, message)


async def broadcast_alert(alert_data: Dict[str, Any], priority: MessagePriority = MessagePriority.URGENT):
    """Broadcast alert to all subscribed connections."""
    message = WebSocketMessage(
        type="alert",
        priority=priority,
        data=alert_data
    )
    
    await manager.broadcast(SubscriptionType.ALERTS, message)
