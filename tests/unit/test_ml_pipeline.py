"""Unit tests for ML pipeline modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.ml_pipeline.feature_engineering import FeatureEngineer
from src.ml_pipeline.model_trainer import ModelTrainer
from src.ml_pipeline.model_versioning import ModelVersionManager


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 50000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(44000, 45000, 100),
            'close': np.random.uniform(45000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
    
    def test_create_price_features(self, feature_engineer, sample_data):
        """Test price feature creation."""
        features = feature_engineer.create_price_features(sample_data)
        
        assert 'returns' in features.columns
        assert 'log_returns' in features.columns
        assert 'volatility' in features.columns
        assert len(features) == len(sample_data) - 1  # Returns lose one row
    
    def test_create_technical_indicators(self, feature_engineer, sample_data):
        """Test technical indicator creation."""
        features = feature_engineer.create_technical_indicators(sample_data)
        
        expected_indicators = ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        for indicator in expected_indicators:
            assert indicator in features.columns
    
    def test_create_temporal_features(self, feature_engineer, sample_data):
        """Test temporal feature creation."""
        features = feature_engineer.create_temporal_features(sample_data)
        
        assert 'hour' in features.columns
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert features['hour'].max() <= 23
        assert features['day_of_week'].max() <= 6
    
    def test_feature_selection(self, feature_engineer):
        """Test feature selection."""
        # Create random features and target
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feat_{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        selected_features = feature_engineer.select_features(X, y, k=5)
        
        assert len(selected_features) == 5
        assert all(feat in X.columns for feat in selected_features)


class TestModelTrainer:
    """Test model training functionality."""
    
    @pytest.fixture
    def model_trainer(self):
        """Create model trainer instance."""
        return ModelTrainer()
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feat_{i}' for i in range(20)])
        y = pd.Series(np.random.randint(0, 2, 1000))
        return X, y
    
    def test_train_random_forest(self, model_trainer, training_data):
        """Test random forest training."""
        X, y = training_data
        
        model, metrics = model_trainer.train_model(X, y, model_type='random_forest')
        
        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['accuracy'] >= 0.0 and metrics['accuracy'] <= 1.0
    
    def test_train_xgboost(self, model_trainer, training_data):
        """Test XGBoost training."""
        X, y = training_data
        
        with patch('src.ml_pipeline.model_trainer.xgb'):
            model, metrics = model_trainer.train_model(X, y, model_type='xgboost')
            
            assert model is not None
            assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1'])
    
    def test_cross_validation(self, model_trainer, training_data):
        """Test cross-validation."""
        X, y = training_data
        
        cv_scores = model_trainer.cross_validate(X, y, cv_folds=3)
        
        assert 'mean_accuracy' in cv_scores
        assert 'std_accuracy' in cv_scores
        assert cv_scores['mean_accuracy'] >= 0.0 and cv_scores['mean_accuracy'] <= 1.0
    
    def test_hyperparameter_optimization(self, model_trainer, training_data):
        """Test hyperparameter optimization."""
        X, y = training_data
        
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5]
        }
        
        best_model, best_params, best_score = model_trainer.optimize_hyperparameters(
            X, y, param_grid, model_type='random_forest'
        )
        
        assert best_model is not None
        assert best_params is not None
        assert best_score >= 0.0 and best_score <= 1.0


class TestModelVersionManager:
    """Test model version management."""
    
    @pytest.fixture
    def version_manager(self):
        """Create version manager instance."""
        with patch('src.ml_pipeline.model_versioning.Path'):
            return ModelVersionManager("./test_models")
    
    def test_save_model(self, version_manager):
        """Test model saving."""
        model = Mock()
        metadata = {
            "accuracy": 0.95,
            "created_at": datetime.now().isoformat()
        }
        
        with patch('src.ml_pipeline.model_versioning.joblib.dump'):
            version_id = version_manager.save_model(model, metadata, version="1.0.0")
            
            assert version_id is not None
            assert "1.0.0" in version_id
    
    def test_load_model(self, version_manager):
        """Test model loading."""
        mock_model = Mock()
        
        with patch('src.ml_pipeline.model_versioning.joblib.load', return_value=mock_model):
            loaded_model = version_manager.load_model("model_1.0.0")
            
            assert loaded_model == mock_model
    
    def test_list_versions(self, version_manager):
        """Test listing model versions."""
        with patch.object(version_manager, 'models_dir') as mock_dir:
            mock_dir.glob.return_value = [
                Mock(name="model_1.0.0.pkl"),
                Mock(name="model_1.1.0.pkl"),
                Mock(name="model_2.0.0.pkl")
            ]
            
            versions = version_manager.list_versions()
            
            assert len(versions) == 3
            assert "1.0.0" in versions[0]
    
    def test_promote_model(self, version_manager):
        """Test model promotion."""
        with patch.object(version_manager, 'load_model'):
            with patch.object(version_manager, 'save_model'):
                result = version_manager.promote_model("model_1.0.0", environment="production")
                
                assert result is True
    
    def test_rollback_model(self, version_manager):
        """Test model rollback."""
        with patch.object(version_manager, 'load_model'):
            with patch.object(version_manager, 'promote_model'):
                result = version_manager.rollback_model("model_1.0.0", environment="production")
                
                assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
