"""
Log aggregation service for ELK Stack integration.
Provides log shipping to Elasticsearch/Logstash for centralized logging.
"""

import asyncio
import json
import socket
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import httpx
from elasticsearch import AsyncElasticsearch
from logstash_async.handler import AsynchronousLogstashHandler
import structlog

from src.config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()


class LogAggregator:
    """Aggregates and ships logs to ELK stack."""
    
    def __init__(
        self,
        elasticsearch_url: str = "http://localhost:9200",
        logstash_host: str = "localhost",
        logstash_port: int = 5000,
        index_prefix: str = "aeneas-logs"
    ):
        self.es_url = elasticsearch_url
        self.logstash_host = logstash_host
        self.logstash_port = logstash_port
        self.index_prefix = index_prefix
        
        # Initialize Elasticsearch client
        self.es_client = AsyncElasticsearch([elasticsearch_url])
        
        # Logstash handler for direct shipping
        self.logstash_handler = None
        
        # Buffer for batch processing
        self.log_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
        
        # Processing statistics
        self.stats = {
            "logs_processed": 0,
            "logs_shipped": 0,
            "errors": 0,
            "last_flush": datetime.utcnow()
        }
        
    async def initialize(self):
        """Initialize log aggregator and create indices."""
        try:
            # Create index template
            await self._create_index_template()
            
            # Setup Logstash handler
            self._setup_logstash_handler()
            
            # Start background flush task
            asyncio.create_task(self._background_flush())
            
            logger.info("Log aggregator initialized", 
                       elasticsearch=self.es_url,
                       logstash=f"{self.logstash_host}:{self.logstash_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize log aggregator: {e}")
            raise
            
    async def _create_index_template(self):
        """Create Elasticsearch index template for logs."""
        template_body = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "refresh_interval": "5s",
                "index.lifecycle.name": "aeneas-logs-policy",
                "index.lifecycle.rollover_alias": f"{self.index_prefix}"
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "logger": {"type": "keyword"},
                    "message": {"type": "text"},
                    "service": {"type": "keyword"},
                    "environment": {"type": "keyword"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "method": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "status_code": {"type": "integer"},
                    "duration_ms": {"type": "float"},
                    "error": {
                        "properties": {
                            "type": {"type": "keyword"},
                            "message": {"type": "text"},
                            "stack_trace": {"type": "text"}
                        }
                    },
                    "metadata": {"type": "object", "enabled": False}
                }
            }
        }
        
        try:
            await self.es_client.indices.put_template(
                name=f"{self.index_prefix}-template",
                body=template_body
            )
            logger.info(f"Created index template: {self.index_prefix}-template")
        except Exception as e:
            logger.warning(f"Could not create index template: {e}")
            
    def _setup_logstash_handler(self):
        """Setup Logstash async handler."""
        try:
            self.logstash_handler = AsynchronousLogstashHandler(
                host=self.logstash_host,
                port=self.logstash_port,
                database_path=None
            )
            logger.info("Logstash handler configured")
        except Exception as e:
            logger.warning(f"Could not setup Logstash handler: {e}")
            
    async def ship_log(self, log_entry: Dict[str, Any]):
        """Ship a single log entry to ELK stack."""
        # Add to buffer
        self.log_buffer.append(self._enrich_log(log_entry))
        self.stats["logs_processed"] += 1
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            await self.flush_buffer()
            
    def _enrich_log(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich log entry with additional metadata."""
        enriched = {
            "@timestamp": log_entry.get("timestamp", datetime.utcnow().isoformat()),
            "service": "aeneas",
            "environment": settings.environment if hasattr(settings, 'environment') else 'development',
            "host": socket.gethostname(),
            **log_entry
        }
        
        # Parse error if present
        if "error" in enriched and isinstance(enriched["error"], str):
            enriched["error"] = {
                "message": enriched["error"],
                "type": "generic"
            }
            
        return enriched
        
    async def flush_buffer(self):
        """Flush buffered logs to Elasticsearch."""
        if not self.log_buffer:
            return
            
        logs_to_ship = self.log_buffer.copy()
        self.log_buffer.clear()
        
        try:
            # Bulk index to Elasticsearch
            bulk_body = []
            index_name = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"
            
            for log in logs_to_ship:
                bulk_body.append({"index": {"_index": index_name}})
                bulk_body.append(log)
                
            if bulk_body:
                response = await self.es_client.bulk(body=bulk_body)
                if not response.get("errors"):
                    self.stats["logs_shipped"] += len(logs_to_ship)
                    logger.debug(f"Shipped {len(logs_to_ship)} logs to Elasticsearch")
                else:
                    self.stats["errors"] += 1
                    logger.error("Bulk indexing had errors", response=response)
                    
            self.stats["last_flush"] = datetime.utcnow()
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to flush logs: {e}")
            # Re-add logs to buffer for retry
            self.log_buffer.extend(logs_to_ship)
            
    async def _background_flush(self):
        """Background task to periodically flush buffer."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush_buffer()
            
    async def create_lifecycle_policy(self):
        """Create index lifecycle management policy."""
        policy = {
            "policy": {
                "phases": {
                    "hot": {
                        "min_age": "0ms",
                        "actions": {
                            "rollover": {
                                "max_age": "1d",
                                "max_size": "5GB"
                            },
                            "set_priority": {
                                "priority": 100
                            }
                        }
                    },
                    "warm": {
                        "min_age": "3d",
                        "actions": {
                            "shrink": {
                                "number_of_shards": 1
                            },
                            "forcemerge": {
                                "max_num_segments": 1
                            },
                            "set_priority": {
                                "priority": 50
                            }
                        }
                    },
                    "cold": {
                        "min_age": "7d",
                        "actions": {
                            "set_priority": {
                                "priority": 0
                            }
                        }
                    },
                    "delete": {
                        "min_age": "30d",
                        "actions": {
                            "delete": {}
                        }
                    }
                }
            }
        }
        
        try:
            await self.es_client.ilm.put_lifecycle(
                policy="aeneas-logs-policy",
                body=policy
            )
            logger.info("Created ILM policy: aeneas-logs-policy")
        except Exception as e:
            logger.warning(f"Could not create ILM policy: {e}")
            
    async def search_logs(
        self,
        query: Optional[str] = None,
        level: Optional[str] = None,
        service: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        size: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs in Elasticsearch."""
        # Build query
        must_clauses = []
        
        if query:
            must_clauses.append({
                "query_string": {
                    "query": query,
                    "default_field": "message"
                }
            })
            
        if level:
            must_clauses.append({"term": {"level": level}})
            
        if service:
            must_clauses.append({"term": {"service": service}})
            
        # Time range
        time_range = {}
        if start_time:
            time_range["gte"] = start_time.isoformat()
        if end_time:
            time_range["lte"] = end_time.isoformat()
        if time_range:
            must_clauses.append({"range": {"@timestamp": time_range}})
            
        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "size": size
        }
        
        try:
            response = await self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            logger.error(f"Failed to search logs: {e}")
            return []
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get log aggregation statistics."""
        # Get index statistics
        try:
            stats = await self.es_client.indices.stats(
                index=f"{self.index_prefix}-*"
            )
            
            total_docs = sum(
                idx["primaries"]["docs"]["count"]
                for idx in stats["indices"].values()
            )
            
            total_size = sum(
                idx["primaries"]["store"]["size_in_bytes"]
                for idx in stats["indices"].values()
            )
            
            return {
                "aggregator_stats": self.stats,
                "elasticsearch": {
                    "total_documents": total_docs,
                    "total_size_bytes": total_size,
                    "indices": list(stats["indices"].keys())
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"aggregator_stats": self.stats, "error": str(e)}
            
    async def close(self):
        """Close connections and cleanup."""
        await self.flush_buffer()
        await self.es_client.close()
        if self.logstash_handler:
            self.logstash_handler.close()
        logger.info("Log aggregator closed")


# Global instance
log_aggregator: Optional[LogAggregator] = None


async def init_log_aggregator(
    elasticsearch_url: Optional[str] = None,
    logstash_host: Optional[str] = None,
    logstash_port: Optional[int] = None
):
    """Initialize global log aggregator."""
    global log_aggregator
    
    if not elasticsearch_url:
        elasticsearch_url = getattr(settings, 'elasticsearch_url', 'http://localhost:9200')
    if not logstash_host:
        logstash_host = getattr(settings, 'logstash_host', 'localhost')
    if not logstash_port:
        logstash_port = getattr(settings, 'logstash_port', 5000)
        
    log_aggregator = LogAggregator(
        elasticsearch_url=elasticsearch_url,
        logstash_host=logstash_host,
        logstash_port=logstash_port
    )
    
    await log_aggregator.initialize()
    await log_aggregator.create_lifecycle_policy()
    
    return log_aggregator


async def get_log_aggregator() -> LogAggregator:
    """Get global log aggregator instance."""
    if not log_aggregator:
        raise RuntimeError("Log aggregator not initialized")
    return log_aggregator
