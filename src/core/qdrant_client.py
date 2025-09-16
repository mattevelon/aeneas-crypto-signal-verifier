"""
Qdrant vector database client.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, UpdateStatus
)
import structlog
import time
from functools import wraps

from src.config.settings import settings

logger = structlog.get_logger()

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
request_times = []

# Global Qdrant client
qdrant_client: Optional[QdrantClient] = None


async def init_qdrant():
    """Initialize Qdrant connection with optional API key authentication."""
    global qdrant_client
    
    try:
        # Parse URL to get host and port
        from urllib.parse import urlparse
        parsed_url = urlparse(settings.vector_db_url)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 6333
        
        # Check if API key is configured for access control
        client_params = {
            "host": host,
            "port": port,
            "grpc_port": 6334,  # Default gRPC port
            "prefer_grpc": True
        }
        
        # Add API key if configured
        if hasattr(settings, 'qdrant_api_key') and settings.qdrant_api_key:
            client_params["api_key"] = settings.qdrant_api_key
            logger.info("Qdrant client initialized with API key authentication")
        
        qdrant_client = QdrantClient(**client_params)
        
        # Create collections if they don't exist
        await create_collections()
        
        logger.info("Qdrant client initialized")
        
    except Exception as e:
        logger.error("Failed to initialize Qdrant", error=str(e))
        raise


async def create_collections():
    """Create Qdrant collections."""
    try:
        collections = qdrant_client.get_collections()
        existing_collections = [c.name for c in collections.collections]
        
        # Create signals collection with optimized index
        if settings.vector_collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=settings.vector_collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE,
                    hnsw_config={
                        "m": 16,  # Number of edges per node
                        "ef_construct": 100,  # Size of dynamic candidate list
                        "full_scan_threshold": 10000  # Use HNSW for collections > 10k vectors
                    }
                ),
                optimizers_config={
                    "indexing_threshold": 20000,  # Start indexing after 20k vectors
                    "memmap_threshold": 1000000,  # Use memmap for large collections
                    "default_segment_number": 2  # Number of segments for parallel processing
                }
            )
            logger.info(f"Created optimized collection: {settings.vector_collection_name}")
        
        # Create historical patterns collection
        if "patterns" not in existing_collections:
            qdrant_client.create_collection(
                collection_name="patterns",
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created collection: patterns")
            
    except Exception as e:
        logger.error("Failed to create Qdrant collections", error=str(e))
        raise


def get_qdrant() -> QdrantClient:
    """Get Qdrant client instance."""
    if not qdrant_client:
        raise RuntimeError("Qdrant not initialized")
    return qdrant_client


def rate_limit(func):
    """Rate limiting decorator for Qdrant operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global request_times
        current_time = time.time()
        
        # Remove old requests outside the window
        request_times = [t for t in request_times if current_time - t < RATE_LIMIT_WINDOW]
        
        # Check if we've exceeded the rate limit
        if len(request_times) >= RATE_LIMIT_REQUESTS:
            sleep_time = RATE_LIMIT_WINDOW - (current_time - request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        request_times.append(current_time)
        
        return func(*args, **kwargs)
    return wrapper


class VectorStore:
    """Vector database operations with access control and rate limiting."""
    
    def __init__(self):
        self.collection_name = "signals"
        self.vector_size = 1536  # OpenAI embeddings dimension
        self.api_key = settings.qdrant_api_key if hasattr(settings, 'qdrant_api_key') else None
        self.access_control_enabled = bool(self.api_key)
    
    @rate_limit
    async def upsert_signal(
        self,
        signal_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Insert or update a signal vector with rate limiting."""
        try:
            client = get_qdrant()
            point_id = signal_id
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            
            client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Vector upserted", collection=self.collection_name, id=point_id)
            return True
            
        except Exception as e:
            logger.error("Failed to upsert vector", error=str(e))
            raise
    
    @rate_limit
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar signals with rate limiting."""
        try:
            client = get_qdrant()
            
            # Build filter if conditions provided
            filter_obj = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                filter_obj = Filter(must=conditions)
            
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_obj
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error("Failed to search vectors", error=str(e))
            return []
    
    @rate_limit
    async def delete_signal(self, signal_id: str) -> bool:
        """Delete a signal from the vector store with rate limiting."""
        try:
            client = get_qdrant()
            client.delete(
                collection_name=self.collection_name,
                points_selector=[signal_id]
            )
            logger.debug(f"Vector deleted", collection=self.collection_name, id=signal_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete vector", error=str(e))
            return False
    
    @rate_limit
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics with rate limiting."""
        try:
            client = get_qdrant()
            results = client.get_collection_stats(
                collection_name=self.collection_name
            )
            
            if results:
                return {
                    "id": results[0].id,
                    "metadata": results[0].payload
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get vector by ID", error=str(e))
            return None


# Create vector store instances
signal_vectors = VectorStore()
pattern_vectors = VectorStore()
