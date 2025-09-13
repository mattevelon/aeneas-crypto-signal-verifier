"""
Qdrant vector database client.
"""

from typing import Optional, List, Dict, Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams
)
import structlog

from src.config.settings import settings

logger = structlog.get_logger()

# Global Qdrant client
qdrant_client: Optional[QdrantClient] = None


async def init_qdrant():
    """Initialize Qdrant client and collections."""
    global qdrant_client
    
    try:
        qdrant_client = QdrantClient(
            url=settings.vector_db_url,
            api_key=settings.qdrant_api_key,
            timeout=30
        )
        
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
        
        # Create signals collection
        if settings.vector_collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=settings.vector_collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {settings.vector_collection_name}")
        
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


class VectorStore:
    """Vector storage operations."""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.vector_collection_name
    
    async def upsert(
        self,
        vector: List[float],
        metadata: Dict[str, Any],
        id: Optional[str] = None
    ) -> str:
        """Insert or update vector with metadata."""
        try:
            client = get_qdrant()
            point_id = id or str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
            
            client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Vector upserted", collection=self.collection_name, id=point_id)
            return point_id
            
        except Exception as e:
            logger.error("Failed to upsert vector", error=str(e))
            raise
    
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
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
                query_vector=query_vector,
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
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            client = get_qdrant()
            client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            logger.debug(f"Vectors deleted", collection=self.collection_name, count=len(ids))
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            return False
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID."""
        try:
            client = get_qdrant()
            results = client.retrieve(
                collection_name=self.collection_name,
                ids=[id]
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
signal_vectors = VectorStore("signals")
pattern_vectors = VectorStore("patterns")
