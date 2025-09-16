"""
Model Training Framework

Trains and optimizes ML models for signal prediction and classification.
Supports multiple algorithms and hyperparameter optimization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import json
import pickle
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

from src.core.redis_client import get_redis
from src.config.settings import get_settings
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelType(str, Enum):
    """Supported model types"""
    RANDOM_FOREST_CLASSIFIER = "rf_classifier"
    RANDOM_FOREST_REGRESSOR = "rf_regressor"
    GRADIENT_BOOSTING_CLASSIFIER = "gb_classifier"
    GRADIENT_BOOSTING_REGRESSOR = "gb_regressor"
    XGBOOST_CLASSIFIER = "xgb_classifier"
    XGBOOST_REGRESSOR = "xgb_regressor"
    LIGHTGBM_CLASSIFIER = "lgb_classifier"
    LIGHTGBM_REGRESSOR = "lgb_regressor"
    LOGISTIC_REGRESSION = "logistic"
    SVM_CLASSIFIER = "svm_classifier"
    SVM_REGRESSOR = "svm_regressor"
    NEURAL_NETWORK_CLASSIFIER = "nn_classifier"
    NEURAL_NETWORK_REGRESSOR = "nn_regressor"


class TaskType(str, Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: ModelType
    task_type: TaskType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    test_split: float = 0.1
    cv_folds: int = 5
    optimization_metric: str = "f1_score"
    optimization_trials: int = 50
    early_stopping_rounds: int = 10
    random_state: int = 42
    use_time_series_split: bool = True
    save_model: bool = True


@dataclass
class TrainingResult:
    """Training results"""
    model_id: str
    model_type: ModelType
    task_type: TaskType
    training_time: float
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None


class ModelTrainer:
    """
    Trains and evaluates ML models for trading signal prediction
    """
    
    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.redis = get_redis()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # Model storage path
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.models = self._initialize_models()
        
        # Best model tracking
        self.best_model = None
        self.best_score = -np.inf
    
    def _initialize_models(self) -> Dict[ModelType, Any]:
        """Initialize available models"""
        return {
            ModelType.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
            ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier,
            ModelType.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor,
            ModelType.XGBOOST_CLASSIFIER: xgb.XGBClassifier,
            ModelType.XGBOOST_REGRESSOR: xgb.XGBRegressor,
            ModelType.LIGHTGBM_CLASSIFIER: lgb.LGBMClassifier,
            ModelType.LIGHTGBM_REGRESSOR: lgb.LGBMRegressor,
            ModelType.LOGISTIC_REGRESSION: LogisticRegression,
            ModelType.SVM_CLASSIFIER: SVC,
            ModelType.SVM_REGRESSOR: SVR,
            ModelType.NEURAL_NETWORK_CLASSIFIER: MLPClassifier,
            ModelType.NEURAL_NETWORK_REGRESSOR: MLPRegressor
        }
    
    async def train_model(self, 
                          X: pd.DataFrame,
                          y: Union[pd.Series, np.ndarray],
                          config: TrainingConfig) -> TrainingResult:
        """
        Train a model with given configuration
        
        Args:
            X: Feature matrix
            y: Target variable
            config: Training configuration
            
        Returns:
            TrainingResult with metrics and trained model
        """
        try:
            start_time = datetime.utcnow()
            model_id = str(uuid.uuid4())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.test_split,
                random_state=config.random_state,
                stratify=y if config.task_type == TaskType.CLASSIFICATION else None
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=config.validation_split,
                random_state=config.random_state,
                stratify=y_train if config.task_type == TaskType.CLASSIFICATION else None
            )
            
            # Get model class
            model_class = self.models[config.model_type]
            
            # Hyperparameter optimization
            best_params = await self._optimize_hyperparameters(
                model_class, X_train, y_train, X_val, y_val, config
            )
            
            # Train final model with best parameters
            model = model_class(**best_params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_metrics = self._evaluate_model(model, X_train, y_train, config.task_type)
            val_metrics = self._evaluate_model(model, X_val, y_val, config.task_type)
            test_metrics = self._evaluate_model(model, X_test, y_test, config.task_type)
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(
                model_class(**best_params), X_train, y_train, config
            )
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X.columns)
            
            # Additional metrics for classification
            confusion_mat = None
            class_report = None
            if config.task_type == TaskType.CLASSIFICATION:
                y_pred = model.predict(X_test)
                confusion_mat = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model if configured
            if config.save_model:
                model_path = await self._save_model(model, model_id, config)
            
            # Track if best model
            if val_metrics.get(config.optimization_metric, 0) > self.best_score:
                self.best_score = val_metrics[config.optimization_metric]
                self.best_model = model
            
            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TrainingResult(
                model_id=model_id,
                model_type=config.model_type,
                task_type=config.task_type,
                training_time=training_time,
                best_params=best_params,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                cross_val_scores=cv_scores,
                confusion_matrix=confusion_mat,
                classification_report=class_report
            )
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    async def train_ensemble(self,
                           X: pd.DataFrame,
                           y: Union[pd.Series, np.ndarray],
                           model_configs: List[TrainingConfig]) -> Dict[str, Any]:
        """
        Train ensemble of models
        
        Args:
            X: Feature matrix
            y: Target variable
            model_configs: List of model configurations
            
        Returns:
            Dict with ensemble results
        """
        try:
            results = []
            models = []
            
            # Train individual models
            for config in model_configs:
                result = await self.train_model(X, y, config)
                results.append(result)
                
                # Load trained model
                model = await self._load_model(result.model_id)
                models.append({
                    'model': model,
                    'weight': result.validation_metrics.get(config.optimization_metric, 0)
                })
            
            # Create ensemble predictions
            ensemble_result = await self._create_ensemble(models, X, y, model_configs[0].task_type)
            
            return {
                'individual_results': results,
                'ensemble_result': ensemble_result,
                'best_individual': max(results, key=lambda x: x.validation_metrics.get(
                    model_configs[0].optimization_metric, 0
                ))
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            return {}
    
    async def _optimize_hyperparameters(self,
                                       model_class: Any,
                                       X_train: pd.DataFrame,
                                       y_train: Union[pd.Series, np.ndarray],
                                       X_val: pd.DataFrame,
                                       y_val: Union[pd.Series, np.ndarray],
                                       config: TrainingConfig) -> Dict[str, Any]:
        """Optimize hyperparameters using grid or random search"""
        try:
            # Get parameter grid for model type
            param_grid = self._get_parameter_grid(config.model_type)
            
            # Use time series split if configured
            if config.use_time_series_split:
                cv = TimeSeriesSplit(n_splits=config.cv_folds)
            else:
                cv = config.cv_folds
            
            # Choose search method
            if len(param_grid) <= 10:
                search = GridSearchCV(
                    model_class(),
                    param_grid,
                    cv=cv,
                    scoring=config.optimization_metric,
                    n_jobs=-1,
                    verbose=0
                )
            else:
                search = RandomizedSearchCV(
                    model_class(),
                    param_grid,
                    n_iter=config.optimization_trials,
                    cv=cv,
                    scoring=config.optimization_metric,
                    n_jobs=-1,
                    random_state=config.random_state,
                    verbose=0
                )
            
            # Fit search
            search.fit(X_train, y_train)
            
            return search.best_params_
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            return config.hyperparameters or {}
    
    def _get_parameter_grid(self, model_type: ModelType) -> Dict[str, List]:
        """Get hyperparameter search grid for model type"""
        grids = {
            ModelType.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelType.XGBOOST_CLASSIFIER: {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            ModelType.LIGHTGBM_CLASSIFIER: {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0]
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            ModelType.NEURAL_NETWORK_CLASSIFIER: {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # Return appropriate grid or default
        return grids.get(model_type, {})
    
    def _evaluate_model(self,
                       model: Any,
                       X: pd.DataFrame,
                       y: Union[pd.Series, np.ndarray],
                       task_type: TaskType) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            y_pred = model.predict(X)
            metrics = {}
            
            if task_type == TaskType.CLASSIFICATION:
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
                
                # ROC AUC for binary classification
                if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
                    y_proba = model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_proba)
                    
            else:  # Regression
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['mae'] = mean_absolute_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2_score'] = r2_score(y, y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def _perform_cross_validation(self,
                                 model: Any,
                                 X: pd.DataFrame,
                                 y: Union[pd.Series, np.ndarray],
                                 config: TrainingConfig) -> List[float]:
        """Perform cross-validation"""
        try:
            if config.use_time_series_split:
                cv = TimeSeriesSplit(n_splits=config.cv_folds)
            else:
                cv = config.cv_folds
            
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=config.optimization_metric,
                n_jobs=-1
            )
            
            return scores.tolist()
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return []
    
    def _get_feature_importance(self,
                               model: Any,
                               feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            importance = {}
            
            # Tree-based models
            if hasattr(model, 'feature_importances_'):
                for name, imp in zip(feature_names, model.feature_importances_):
                    importance[name] = float(imp)
            
            # Linear models
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = np.abs(coef).mean(axis=0)
                for name, imp in zip(feature_names, coef):
                    importance[name] = float(abs(imp))
            
            # Sort by importance
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    async def _save_model(self,
                         model: Any,
                         model_id: str,
                         config: TrainingConfig) -> Path:
        """Save trained model to disk"""
        try:
            model_path = self.model_dir / f"{model_id}.pkl"
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save config
            config_path = self.model_dir / f"{model_id}_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_type': config.model_type.value,
                    'task_type': config.task_type.value,
                    'hyperparameters': config.hyperparameters,
                    'optimization_metric': config.optimization_metric
                }, f)
            
            # Cache model metadata
            await self.redis.setex(
                f"model:{model_id}",
                86400,  # 24 hours
                json.dumps({
                    'path': str(model_path),
                    'created_at': datetime.utcnow().isoformat(),
                    'model_type': config.model_type.value
                })
            )
            
            logger.info(f"Model saved: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    async def _load_model(self, model_id: str) -> Any:
        """Load model from disk"""
        try:
            model_path = self.model_dir / f"{model_id}.pkl"
            
            if not model_path.exists():
                # Check cache for path
                cached = await self.redis.get(f"model:{model_id}")
                if cached:
                    data = json.loads(cached)
                    model_path = Path(data['path'])
            
            if model_path.exists():
                return joblib.load(model_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    async def _create_ensemble(self,
                             models: List[Dict],
                             X: pd.DataFrame,
                             y: Union[pd.Series, np.ndarray],
                             task_type: TaskType) -> Dict[str, Any]:
        """Create ensemble predictions"""
        try:
            # Normalize weights
            total_weight = sum(m['weight'] for m in models)
            for m in models:
                m['weight'] /= total_weight
            
            # Generate ensemble predictions
            if task_type == TaskType.CLASSIFICATION:
                # Voting ensemble
                predictions = []
                for m in models:
                    pred = m['model'].predict(X)
                    predictions.append(pred * m['weight'])
                
                ensemble_pred = np.round(np.sum(predictions, axis=0))
                
            else:  # Regression
                # Weighted average
                predictions = []
                for m in models:
                    pred = m['model'].predict(X)
                    predictions.append(pred * m['weight'])
                
                ensemble_pred = np.sum(predictions, axis=0)
            
            # Evaluate ensemble
            if task_type == TaskType.CLASSIFICATION:
                metrics = {
                    'accuracy': accuracy_score(y, ensemble_pred),
                    'f1_score': f1_score(y, ensemble_pred, average='weighted')
                }
            else:
                metrics = {
                    'mse': mean_squared_error(y, ensemble_pred),
                    'r2_score': r2_score(y, ensemble_pred)
                }
            
            return {
                'predictions': ensemble_pred.tolist(),
                'metrics': metrics,
                'model_weights': {i: m['weight'] for i, m in enumerate(models)}
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            return {}
    
    def get_best_model(self) -> Any:
        """Get the best trained model"""
        return self.best_model
    
    async def predict(self,
                     model_id: str,
                     X: pd.DataFrame) -> np.ndarray:
        """Make predictions with a trained model"""
        try:
            model = await self._load_model(model_id)
            if model:
                return model.predict(X)
            return np.array([])
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
