"""
Log retention and rotation policies for the AENEAS system.
"""

import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import asyncio
import logging.handlers

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = structlog.get_logger()


class LogRetentionManager:
    """Manages log retention, rotation, and archival."""
    
    def __init__(
        self,
        log_dir: str = "/var/log/aeneas",
        retention_days: int = 30,
        archive_days: int = 90,
        max_size_mb: int = 100,
        compress_after_days: int = 7
    ):
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.archive_days = archive_days
        self.max_size_mb = max_size_mb
        self.compress_after_days = compress_after_days
        self.scheduler = AsyncIOScheduler()
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "archive").mkdir(exist_ok=True)
        (self.log_dir / "compressed").mkdir(exist_ok=True)
    
    async def start(self):
        """Start the retention scheduler."""
        # Daily log rotation at 2 AM
        self.scheduler.add_job(
            self.rotate_logs,
            CronTrigger(hour=2, minute=0),
            id="log_rotation",
            name="Daily log rotation"
        )
        
        # Weekly compression at 3 AM on Sunday
        self.scheduler.add_job(
            self.compress_old_logs,
            CronTrigger(day_of_week=0, hour=3, minute=0),
            id="log_compression",
            name="Weekly log compression"
        )
        
        # Monthly cleanup at 4 AM on the 1st
        self.scheduler.add_job(
            self.cleanup_old_logs,
            CronTrigger(day=1, hour=4, minute=0),
            id="log_cleanup",
            name="Monthly log cleanup"
        )
        
        self.scheduler.start()
        logger.info("Log retention manager started", 
                   retention_days=self.retention_days,
                   archive_days=self.archive_days)
    
    async def stop(self):
        """Stop the retention scheduler."""
        self.scheduler.shutdown()
        logger.info("Log retention manager stopped")
    
    async def rotate_logs(self):
        """Rotate log files based on size and age."""
        try:
            for log_file in self.log_dir.glob("*.log"):
                # Check file size
                size_mb = log_file.stat().st_size / (1024 * 1024)
                
                if size_mb > self.max_size_mb:
                    # Rotate based on size
                    await self._rotate_file(log_file, reason="size")
                
                # Check file age
                age_days = (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days
                
                if age_days >= 1:
                    # Rotate daily logs
                    await self._rotate_file(log_file, reason="daily")
            
            logger.info("Log rotation completed")
            
        except Exception as e:
            logger.error("Log rotation failed", error=str(e), exc_info=e)
    
    async def _rotate_file(self, file_path: Path, reason: str = "unknown"):
        """Rotate a single log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{file_path.stem}_{timestamp}_{reason}{file_path.suffix}"
        new_path = self.log_dir / "archive" / new_name
        
        try:
            shutil.move(str(file_path), str(new_path))
            logger.info("Log file rotated", 
                       original=str(file_path),
                       new_path=str(new_path),
                       reason=reason)
        except Exception as e:
            logger.error("Failed to rotate log file",
                        file=str(file_path),
                        error=str(e))
    
    async def compress_old_logs(self):
        """Compress logs older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.compress_after_days)
            
            for log_file in (self.log_dir / "archive").glob("*.log"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    compressed_path = self.log_dir / "compressed" / f"{log_file.name}.gz"
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original after compression
                    log_file.unlink()
                    
                    logger.info("Log file compressed",
                              original=str(log_file),
                              compressed=str(compressed_path))
            
            logger.info("Log compression completed")
            
        except Exception as e:
            logger.error("Log compression failed", error=str(e), exc_info=e)
    
    async def cleanup_old_logs(self):
        """Remove logs older than retention period."""
        try:
            # Cleanup archived logs
            archive_cutoff = datetime.now() - timedelta(days=self.retention_days)
            
            for log_file in (self.log_dir / "archive").glob("*"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_time < archive_cutoff:
                    log_file.unlink()
                    logger.info("Archived log deleted", file=str(log_file))
            
            # Cleanup compressed logs
            compress_cutoff = datetime.now() - timedelta(days=self.archive_days)
            
            for log_file in (self.log_dir / "compressed").glob("*.gz"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_time < compress_cutoff:
                    log_file.unlink()
                    logger.info("Compressed log deleted", file=str(log_file))
            
            logger.info("Log cleanup completed")
            
        except Exception as e:
            logger.error("Log cleanup failed", error=str(e), exc_info=e)
    
    def get_log_statistics(self) -> dict:
        """Get statistics about log files."""
        stats = {
            "active_logs": 0,
            "archived_logs": 0,
            "compressed_logs": 0,
            "total_size_mb": 0,
            "oldest_log": None,
            "newest_log": None
        }
        
        try:
            # Active logs
            active_logs = list(self.log_dir.glob("*.log"))
            stats["active_logs"] = len(active_logs)
            
            # Archived logs
            archived_logs = list((self.log_dir / "archive").glob("*"))
            stats["archived_logs"] = len(archived_logs)
            
            # Compressed logs
            compressed_logs = list((self.log_dir / "compressed").glob("*.gz"))
            stats["compressed_logs"] = len(compressed_logs)
            
            # Calculate total size
            all_logs = active_logs + archived_logs + compressed_logs
            total_size = sum(f.stat().st_size for f in all_logs if f.exists())
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Find oldest and newest
            if all_logs:
                sorted_logs = sorted(all_logs, key=lambda f: f.stat().st_mtime)
                stats["oldest_log"] = {
                    "name": sorted_logs[0].name,
                    "date": datetime.fromtimestamp(sorted_logs[0].stat().st_mtime).isoformat()
                }
                stats["newest_log"] = {
                    "name": sorted_logs[-1].name,
                    "date": datetime.fromtimestamp(sorted_logs[-1].stat().st_mtime).isoformat()
                }
        
        except Exception as e:
            logger.error("Failed to get log statistics", error=str(e))
        
        return stats


class TimedRotatingLogger:
    """Custom timed rotating file handler with compression."""
    
    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 30,
        compress: bool = True
    ):
        self.base_filename = filename
        self.when = when
        self.interval = interval
        self.backup_count = backup_count
        self.compress = compress
        
        # Create the handler
        self.handler = logging.handlers.TimedRotatingFileHandler(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backup_count
        )
        
        if compress:
            self.handler.rotator = self._compress_rotator
            self.handler.namer = self._compress_namer
    
    def _compress_rotator(self, source: str, dest: str):
        """Compress log file during rotation."""
        with open(source, 'rb') as f_in:
            with gzip.open(f"{dest}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)
    
    def _compress_namer(self, name: str) -> str:
        """Name compressed log files."""
        return f"{name}.gz"
    
    def get_handler(self) -> logging.Handler:
        """Get the configured handler."""
        return self.handler


# Global retention manager instance
retention_manager = None


def init_retention_manager(
    log_dir: str = None,
    retention_days: int = 30,
    archive_days: int = 90
) -> LogRetentionManager:
    """Initialize the global retention manager."""
    global retention_manager
    
    if log_dir is None:
        from src.config.settings import get_settings
        settings = get_settings()
        log_dir = settings.log_directory or "/var/log/aeneas"
    
    retention_manager = LogRetentionManager(
        log_dir=log_dir,
        retention_days=retention_days,
        archive_days=archive_days
    )
    
    return retention_manager


async def start_retention_management():
    """Start log retention management."""
    if retention_manager:
        await retention_manager.start()


async def stop_retention_management():
    """Stop log retention management."""
    if retention_manager:
        await retention_manager.stop()
