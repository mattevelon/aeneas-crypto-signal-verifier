"""Configuration hot-reload mechanism using watchdog."""

import os
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from src.config.settings import Settings
import logging

logger = logging.getLogger(__name__)


class ConfigReloadHandler(FileSystemEventHandler):
    """Handler for configuration file changes."""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        self.env_file = Path(".env")
    
    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification events."""
        if not event.is_directory:
            if Path(event.src_path).name == ".env":
                logger.info(f"Configuration file {event.src_path} modified, reloading...")
                self.callback()


class ConfigHotReloader:
    """Hot reload configuration when .env file changes."""
    
    def __init__(self):
        self.observer: Optional[Observer] = None
        self.settings: Settings = Settings()
        
    def reload_config(self):
        """Reload configuration from file."""
        try:
            # Clear cached settings
            Settings.model_config['env_file_encoding'] = 'utf-8'
            self.settings = Settings()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def start_watching(self):
        """Start watching configuration file for changes."""
        self.observer = Observer()
        handler = ConfigReloadHandler(callback=self.reload_config)
        self.observer.schedule(handler, path=".", recursive=False)
        self.observer.start()
        logger.info("Configuration hot-reload started")
    
    def stop_watching(self):
        """Stop watching configuration file."""
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("Configuration hot-reload stopped")
    
    def get_settings(self) -> Settings:
        """Get current settings instance."""
        return self.settings


# Global hot-reloader instance
config_reloader = ConfigHotReloader()


def start_config_hot_reload():
    """Start configuration hot-reload monitoring."""
    config_reloader.start_watching()


def stop_config_hot_reload():
    """Stop configuration hot-reload monitoring."""
    config_reloader.stop_watching()


def get_hot_settings() -> Settings:
    """Get current hot-reloaded settings."""
    return config_reloader.get_settings()
