"""
Async database operations with bulk insert optimization.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, update, delete
from sqlalchemy.dialects.postgresql import insert
import structlog

from src.core.database import get_db_context
from src.models import Signal, TelegramMessage, ChannelStatistics

logger = structlog.get_logger()


class AsyncDatabaseOperations:
    """Optimized async database operations."""
    
    def __init__(self):
        self.batch_size = 1000
        self.compression_enabled = True
    
    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions."""
        async with get_db_context() as db:
            try:
                yield db
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
    
    async def bulk_insert_messages(
        self, 
        messages: List[Dict[str, Any]], 
        compress: bool = True
    ) -> int:
        """Bulk insert messages with optimization."""
        if not messages:
            return 0
        
        inserted = 0
        
        async with self.transaction() as db:
            try:
                # Process in batches
                for i in range(0, len(messages), self.batch_size):
                    batch = messages[i:i + self.batch_size]
                    
                    # Prepare data for bulk insert
                    values = []
                    for msg in batch:
                        # Compress large content if enabled
                        content = msg.get("content", "")
                        if compress and len(content) > 1000:
                            content = self._compress_text(content)
                        
                        values.append({
                            "channel_id": msg["channel_id"],
                            "message_id": msg["message_id"],
                            "content": content,
                            "author": msg.get("author"),
                            "timestamp": msg["timestamp"],
                            "has_media": msg.get("has_media", False),
                            "media_urls": msg.get("media_urls"),
                            "processed": msg.get("processed", False),
                            "is_signal": msg.get("is_signal", False),
                            "created_at": datetime.utcnow()
                        })
                    
                    # Use PostgreSQL's ON CONFLICT for upsert
                    stmt = insert(TelegramMessage).values(values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["channel_id", "message_id"],
                        set_={
                            "content": stmt.excluded.content,
                            "processed": stmt.excluded.processed,
                            "is_signal": stmt.excluded.is_signal
                        }
                    )
                    
                    result = await db.execute(stmt)
                    inserted += result.rowcount
                    
                    logger.info(f"Bulk inserted {result.rowcount} messages")
                
                return inserted
                
            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                raise
    
    async def bulk_update_signals(
        self,
        signal_updates: List[Dict[str, Any]]
    ) -> int:
        """Bulk update signals with transaction management."""
        if not signal_updates:
            return 0
        
        updated = 0
        
        async with self.transaction() as db:
            try:
                for update_data in signal_updates:
                    signal_id = update_data.pop("id")
                    update_data["updated_at"] = datetime.utcnow()
                    
                    stmt = (
                        update(Signal)
                        .where(Signal.id == signal_id)
                        .values(**update_data)
                    )
                    
                    result = await db.execute(stmt)
                    updated += result.rowcount
                
                logger.info(f"Bulk updated {updated} signals")
                return updated
                
            except Exception as e:
                logger.error(f"Bulk update failed: {e}")
                raise
    
    async def archive_old_messages(
        self,
        days_old: int = 30,
        archive_table: str = "telegram_messages_archive"
    ) -> int:
        """Archive old messages to separate table."""
        async with self.transaction() as db:
            try:
                # Create archive table if not exists
                await db.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {archive_table} 
                    (LIKE telegram_messages INCLUDING ALL)
                """))
                
                # Move old messages
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                result = await db.execute(text(f"""
                    WITH moved AS (
                        DELETE FROM telegram_messages
                        WHERE created_at < :cutoff
                        RETURNING *
                    )
                    INSERT INTO {archive_table}
                    SELECT * FROM moved
                """), {"cutoff": cutoff_date})
                
                archived = result.rowcount
                logger.info(f"Archived {archived} messages older than {days_old} days")
                
                return archived
                
            except Exception as e:
                logger.error(f"Archive operation failed: {e}")
                raise
    
    async def optimize_tables(self):
        """Run VACUUM and ANALYZE on tables."""
        async with get_db_context() as db:
            try:
                # Note: VACUUM cannot run in a transaction
                await db.execute(text("COMMIT"))
                await db.execute(text("VACUUM ANALYZE signals"))
                await db.execute(text("VACUUM ANALYZE telegram_messages"))
                await db.execute(text("VACUUM ANALYZE channel_statistics"))
                
                logger.info("Database tables optimized")
                
            except Exception as e:
                logger.error(f"Table optimization failed: {e}")
                raise
    
    async def get_channel_statistics(
        self,
        channel_id: int,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get channel statistics with caching."""
        async with get_db_context() as db:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get message count
                msg_count = await db.execute(
                    select(func.count(TelegramMessage.id))
                    .where(TelegramMessage.channel_id == channel_id)
                    .where(TelegramMessage.created_at >= cutoff_date)
                )
                
                # Get signal count
                signal_count = await db.execute(
                    select(func.count(Signal.id))
                    .where(Signal.source_channel_id == channel_id)
                    .where(Signal.created_at >= cutoff_date)
                )
                
                # Get accuracy metrics
                accuracy = await db.execute(
                    select(ChannelStatistics)
                    .where(ChannelStatistics.channel_id == channel_id)
                    .order_by(ChannelStatistics.created_at.desc())
                    .limit(1)
                )
                
                return {
                    "channel_id": channel_id,
                    "message_count": msg_count.scalar() or 0,
                    "signal_count": signal_count.scalar() or 0,
                    "accuracy": accuracy.scalar_one_or_none(),
                    "period_days": days
                }
                
            except Exception as e:
                logger.error(f"Failed to get channel statistics: {e}")
                return {}
    
    def _compress_text(self, text: str) -> str:
        """Compress large text content."""
        import zlib
        import base64
        
        if len(text) > 1000:
            compressed = zlib.compress(text.encode('utf-8'), level=6)
            return f"COMPRESSED:{base64.b64encode(compressed).decode('utf-8')}"
        return text
    
    def _decompress_text(self, text: str) -> str:
        """Decompress text content."""
        import zlib
        import base64
        
        if text.startswith("COMPRESSED:"):
            compressed_data = text[11:]
            decompressed = zlib.decompress(base64.b64decode(compressed_data))
            return decompressed.decode('utf-8')
        return text


# Global instance
db_ops = AsyncDatabaseOperations()


# Convenience functions
async def bulk_insert_messages(messages: List[Dict[str, Any]]) -> int:
    """Bulk insert messages."""
    return await db_ops.bulk_insert_messages(messages)


async def bulk_update_signals(updates: List[Dict[str, Any]]) -> int:
    """Bulk update signals."""
    return await db_ops.bulk_update_signals(updates)


async def archive_old_data(days: int = 30) -> int:
    """Archive old data."""
    return await db_ops.archive_old_messages(days)


async def optimize_database():
    """Optimize database tables."""
    await db_ops.optimize_tables()


from datetime import timedelta
from sqlalchemy import func
