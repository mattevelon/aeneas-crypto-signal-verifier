"""
Model Versioning System

Manages model versions, tracks lineage, and handles model deployment lifecycle.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import logging
import json
import hashlib
import shutil
import joblib

from sqlalchemy import select, and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from src.models import Base
from src.core.database import get_async_session
from src.core.redis_client import get_redis
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelStatus(str, Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelEnvironment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    model_type: str
    created_at: datetime
    created_by: str
    status: ModelStatus
    environment: ModelEnvironment
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    feature_schema: List[str]
    model_path: str
    parent_version: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    deployment_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'model_type': self.model_type,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'status': self.status.value,
            'environment': self.environment.value,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'feature_schema': self.feature_schema,
            'model_path': self.model_path,
            'parent_version': self.parent_version,
            'description': self.description,
            'tags': self.tags,
            'deployment_history': self.deployment_history
        }


class ModelVersionManager:
    """
    Manages model versioning, lineage tracking, and deployment lifecycle
    """
    
    def __init__(self):
        self.redis = get_redis()
        
        # Model storage paths
        self.model_root = Path("models")
        self.model_root.mkdir(exist_ok=True)
        
        self.archive_root = self.model_root / "archive"
        self.archive_root.mkdir(exist_ok=True)
        
        self.production_root = self.model_root / "production"
        self.production_root.mkdir(exist_ok=True)
        
        # Version registry
        self.version_registry = {}
        self._load_registry()
        
        # Current production models
        self.production_models = {}
        
        # Deployment lock
        self.deployment_lock = asyncio.Lock()
    
    def _load_registry(self):
        """Load model registry from disk"""
        registry_path = self.model_root / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                for model_id, version_data in data.items():
                    self.version_registry[model_id] = ModelVersion(**version_data)
    
    def _save_registry(self):
        """Save model registry to disk"""
        registry_path = self.model_root / "registry.json"
        data = {
            model_id: version.to_dict() 
            for model_id, version in self.version_registry.items()
        }
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def register_model(self,
                            model_path: str,
                            model_type: str,
                            metrics: Dict[str, float],
                            parameters: Dict[str, Any],
                            feature_schema: List[str],
                            created_by: str = "system",
                            parent_version: Optional[str] = None,
                            description: Optional[str] = None,
                            tags: Optional[List[str]] = None) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model_path: Path to saved model
            model_type: Type of model
            metrics: Model performance metrics
            parameters: Model hyperparameters
            feature_schema: List of feature names
            created_by: User/system that created the model
            parent_version: Parent model version (for lineage)
            description: Model description
            tags: Optional tags for categorization
            
        Returns:
            ModelVersion object
        """
        try:
            # Generate unique model ID
            model_id = str(uuid.uuid4())
            
            # Generate version string
            version = self._generate_version(model_type, parent_version)
            
            # Create model directory
            model_dir = self.model_root / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Copy model file
            src_path = Path(model_path)
            dst_path = model_dir / "model.pkl"
            shutil.copy2(src_path, dst_path)
            
            # Save metadata
            metadata = {
                'model_type': model_type,
                'metrics': metrics,
                'parameters': parameters,
                'feature_schema': feature_schema,
                'created_by': created_by,
                'created_at': datetime.utcnow().isoformat(),
                'version': version,
                'parent_version': parent_version,
                'description': description,
                'tags': tags or []
            }
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Create version object
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model_type=model_type,
                created_at=datetime.utcnow(),
                created_by=created_by,
                status=ModelStatus.VALIDATION,
                environment=ModelEnvironment.DEVELOPMENT,
                metrics=metrics,
                parameters=parameters,
                feature_schema=feature_schema,
                model_path=str(dst_path),
                parent_version=parent_version,
                description=description,
                tags=tags or []
            )
            
            # Register in registry
            self.version_registry[model_id] = model_version
            self._save_registry()
            
            # Cache in Redis
            await self._cache_version(model_version)
            
            # Log registration
            logger.info(f"Registered model {model_id} version {version}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    async def promote_model(self,
                           model_id: str,
                           target_environment: ModelEnvironment,
                           reason: Optional[str] = None) -> bool:
        """
        Promote model to target environment
        
        Args:
            model_id: Model identifier
            target_environment: Target deployment environment
            reason: Reason for promotion
            
        Returns:
            bool: Success status
        """
        async with self.deployment_lock:
            try:
                if model_id not in self.version_registry:
                    logger.error(f"Model {model_id} not found")
                    return False
                
                model_version = self.version_registry[model_id]
                
                # Validate promotion path
                if not self._validate_promotion(model_version, target_environment):
                    logger.error(f"Invalid promotion from {model_version.environment} to {target_environment}")
                    return False
                
                # Perform promotion
                old_environment = model_version.environment
                model_version.environment = target_environment
                
                # Update status based on environment
                if target_environment == ModelEnvironment.STAGING:
                    model_version.status = ModelStatus.STAGING
                elif target_environment == ModelEnvironment.PRODUCTION:
                    model_version.status = ModelStatus.PRODUCTION
                    
                    # Deploy to production
                    await self._deploy_to_production(model_version)
                
                # Add to deployment history
                model_version.deployment_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'from_environment': old_environment.value,
                    'to_environment': target_environment.value,
                    'reason': reason
                })
                
                # Save changes
                self._save_registry()
                await self._cache_version(model_version)
                
                logger.info(f"Promoted model {model_id} to {target_environment}")
                return True
                
            except Exception as e:
                logger.error(f"Error promoting model: {str(e)}")
                return False
    
    async def rollback_model(self,
                           current_model_id: str,
                           target_model_id: str,
                           reason: Optional[str] = None) -> bool:
        """
        Rollback from current model to target model
        
        Args:
            current_model_id: Currently deployed model
            target_model_id: Model to rollback to
            reason: Reason for rollback
            
        Returns:
            bool: Success status
        """
        async with self.deployment_lock:
            try:
                if current_model_id not in self.version_registry:
                    logger.error(f"Current model {current_model_id} not found")
                    return False
                
                if target_model_id not in self.version_registry:
                    logger.error(f"Target model {target_model_id} not found")
                    return False
                
                current_version = self.version_registry[current_model_id]
                target_version = self.version_registry[target_model_id]
                
                # Deprecate current model
                current_version.status = ModelStatus.DEPRECATED
                current_version.environment = ModelEnvironment.DEVELOPMENT
                
                # Promote target model to production
                target_version.status = ModelStatus.PRODUCTION
                target_version.environment = ModelEnvironment.PRODUCTION
                
                # Deploy target model
                await self._deploy_to_production(target_version)
                
                # Update deployment history
                current_version.deployment_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'action': 'rollback',
                    'reason': reason
                })
                
                target_version.deployment_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'action': 'rollback_target',
                    'from_model': current_model_id,
                    'reason': reason
                })
                
                # Save changes
                self._save_registry()
                await self._cache_version(current_version)
                await self._cache_version(target_version)
                
                logger.info(f"Rolled back from {current_model_id} to {target_model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error during rollback: {str(e)}")
                return False
    
    async def get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get model version metadata
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelVersion or None
        """
        try:
            # Check cache first
            cached = await self.redis.get(f"model_version:{model_id}")
            if cached:
                data = json.loads(cached)
                return ModelVersion(**data)
            
            # Check registry
            if model_id in self.version_registry:
                return self.version_registry[model_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model version: {str(e)}")
            return None
    
    async def get_production_model(self, model_type: str) -> Optional[ModelVersion]:
        """
        Get current production model for a type
        
        Args:
            model_type: Type of model
            
        Returns:
            ModelVersion in production or None
        """
        try:
            for model_id, version in self.version_registry.items():
                if (version.model_type == model_type and 
                    version.status == ModelStatus.PRODUCTION and
                    version.environment == ModelEnvironment.PRODUCTION):
                    return version
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting production model: {str(e)}")
            return None
    
    async def compare_models(self,
                            model_id1: str,
                            model_id2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            model_id1: First model ID
            model_id2: Second model ID
            
        Returns:
            Dict with comparison results
        """
        try:
            version1 = await self.get_model_version(model_id1)
            version2 = await self.get_model_version(model_id2)
            
            if not version1 or not version2:
                return {}
            
            comparison = {
                'model_1': {
                    'id': model_id1,
                    'version': version1.version,
                    'type': version1.model_type,
                    'status': version1.status.value,
                    'metrics': version1.metrics
                },
                'model_2': {
                    'id': model_id2,
                    'version': version2.version,
                    'type': version2.model_type,
                    'status': version2.status.value,
                    'metrics': version2.metrics
                },
                'metric_comparison': {},
                'feature_comparison': {
                    'common': list(set(version1.feature_schema) & set(version2.feature_schema)),
                    'only_in_1': list(set(version1.feature_schema) - set(version2.feature_schema)),
                    'only_in_2': list(set(version2.feature_schema) - set(version1.feature_schema))
                },
                'parameter_diff': self._compare_parameters(version1.parameters, version2.parameters)
            }
            
            # Compare metrics
            for metric in set(list(version1.metrics.keys()) + list(version2.metrics.keys())):
                val1 = version1.metrics.get(metric)
                val2 = version2.metrics.get(metric)
                
                if val1 is not None and val2 is not None:
                    comparison['metric_comparison'][metric] = {
                        'model_1': val1,
                        'model_2': val2,
                        'improvement': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {}
    
    async def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get model lineage (parent/child relationships)
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of models in lineage
        """
        try:
            lineage = []
            current_id = model_id
            
            # Traverse up to parents
            while current_id:
                version = await self.get_model_version(current_id)
                if not version:
                    break
                
                lineage.append({
                    'model_id': current_id,
                    'version': version.version,
                    'created_at': version.created_at.isoformat(),
                    'metrics': version.metrics,
                    'status': version.status.value
                })
                
                current_id = version.parent_version
            
            # Reverse to show oldest first
            lineage.reverse()
            
            # Find children
            children = []
            for vid, v in self.version_registry.items():
                if v.parent_version == model_id:
                    children.append({
                        'model_id': vid,
                        'version': v.version,
                        'created_at': v.created_at.isoformat(),
                        'metrics': v.metrics,
                        'status': v.status.value
                    })
            
            return {
                'ancestors': lineage[:-1] if lineage else [],
                'current': lineage[-1] if lineage else None,
                'descendants': children
            }
            
        except Exception as e:
            logger.error(f"Error getting model lineage: {str(e)}")
            return []
    
    async def archive_model(self, model_id: str) -> bool:
        """
        Archive a model
        
        Args:
            model_id: Model to archive
            
        Returns:
            bool: Success status
        """
        try:
            if model_id not in self.version_registry:
                return False
            
            version = self.version_registry[model_id]
            
            # Check if can be archived
            if version.status == ModelStatus.PRODUCTION:
                logger.error("Cannot archive production model")
                return False
            
            # Move model files to archive
            src_dir = Path(version.model_path).parent
            dst_dir = self.archive_root / model_id
            shutil.move(str(src_dir), str(dst_dir))
            
            # Update status
            version.status = ModelStatus.DEPRECATED
            version.model_path = str(dst_dir / "model.pkl")
            
            # Save changes
            self._save_registry()
            await self._cache_version(version)
            
            logger.info(f"Archived model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving model: {str(e)}")
            return False
    
    async def cleanup_old_models(self, days_old: int = 30) -> int:
        """
        Clean up old deprecated models
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of models cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cleaned = 0
            
            for model_id, version in list(self.version_registry.items()):
                if (version.status == ModelStatus.DEPRECATED and
                    version.created_at < cutoff_date):
                    
                    # Remove model files
                    model_dir = Path(version.model_path).parent
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    
                    # Remove from registry
                    del self.version_registry[model_id]
                    
                    # Remove from cache
                    await self.redis.delete(f"model_version:{model_id}")
                    
                    cleaned += 1
            
            if cleaned > 0:
                self._save_registry()
                logger.info(f"Cleaned up {cleaned} old models")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up models: {str(e)}")
            return 0
    
    # Helper methods
    def _generate_version(self, model_type: str, parent_version: Optional[str]) -> str:
        """Generate version string"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if parent_version:
            # Increment parent version
            parts = parent_version.split('.')
            if len(parts) == 3:
                parts[-1] = str(int(parts[-1]) + 1)
                return '.'.join(parts)
        
        # New version line
        return f"1.0.0_{timestamp}"
    
    def _validate_promotion(self, model_version: ModelVersion, 
                          target_environment: ModelEnvironment) -> bool:
        """Validate promotion path"""
        valid_paths = {
            (ModelEnvironment.DEVELOPMENT, ModelEnvironment.STAGING),
            (ModelEnvironment.STAGING, ModelEnvironment.PRODUCTION),
            (ModelEnvironment.DEVELOPMENT, ModelEnvironment.PRODUCTION)  # Fast track
        }
        
        return (model_version.environment, target_environment) in valid_paths
    
    async def _deploy_to_production(self, model_version: ModelVersion):
        """Deploy model to production"""
        try:
            # Copy model to production directory
            src_path = Path(model_version.model_path)
            dst_path = self.production_root / f"{model_version.model_type}.pkl"
            shutil.copy2(src_path, dst_path)
            
            # Update production models registry
            self.production_models[model_version.model_type] = model_version.model_id
            
            # Cache production model reference
            await self.redis.set(
                f"production_model:{model_version.model_type}",
                model_version.model_id
            )
            
            logger.info(f"Deployed model {model_version.model_id} to production")
            
        except Exception as e:
            logger.error(f"Error deploying to production: {str(e)}")
            raise
    
    def _compare_parameters(self, params1: Dict, params2: Dict) -> Dict[str, Any]:
        """Compare two parameter dictionaries"""
        diff = {
            'changed': {},
            'added': {},
            'removed': {}
        }
        
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            if key in params1 and key in params2:
                if params1[key] != params2[key]:
                    diff['changed'][key] = {
                        'from': params1[key],
                        'to': params2[key]
                    }
            elif key in params1:
                diff['removed'][key] = params1[key]
            else:
                diff['added'][key] = params2[key]
        
        return diff
    
    async def _cache_version(self, model_version: ModelVersion):
        """Cache model version in Redis"""
        await self.redis.setex(
            f"model_version:{model_version.model_id}",
            86400,  # 24 hours
            json.dumps(model_version.to_dict(), default=str)
        )
    
    def load_production_model(self, model_type: str) -> Any:
        """Load production model for inference"""
        try:
            model_path = self.production_root / f"{model_type}.pkl"
            
            if not model_path.exists():
                # Try to find in registry
                for model_id, version in self.version_registry.items():
                    if (version.model_type == model_type and 
                        version.status == ModelStatus.PRODUCTION):
                        model_path = Path(version.model_path)
                        break
            
            if model_path.exists():
                return joblib.load(model_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            return None
