"""
Fast unit tests for model training pipeline.
Uses synthetic data and minimal Optuna trials for quick CI/CD validation.

This test is completely self-contained - no external connections (DB, MLflow) required.
Can run in CI/CD without any .env configuration.
"""
import os
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Ensure no external connections are attempted
os.environ['LOCAL_DEVELOPMENT'] = 'false'

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


@pytest.fixture
def synthetic_dataset():
    """Generate a small synthetic dataset for fast testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate binary labels with some class imbalance (similar to real data)
    y = pd.Series(np.random.binomial(1, 0.3, n_samples))

    return X, y


@pytest.fixture
def minimal_optuna_params():
    """Minimal Optuna configuration for fast tests."""
    return {
        'n_trials': 10,
        'n_folds': 2,
        'n_startup_trials': 3,
    }


@pytest.mark.unit
@pytest.mark.training
class TestModelTrainingUnit:
    """Fast unit tests for model training components."""

    def test_xgboost_cpu_training(self, synthetic_dataset):
        """Test XGBoost trains successfully on CPU."""
        X, y = synthetic_dataset

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=10,
            max_depth=3,
            device='cpu',
            random_state=42
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_xgboost_gpu_training_if_available(self, synthetic_dataset):
        """Test XGBoost trains on GPU if available, otherwise skip."""
        X, y = synthetic_dataset

        try:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=10,
                max_depth=3,
                device='cuda',
                random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)

            assert len(predictions) == len(y)
            print("GPU training successful!")
        except Exception as e:
            if 'CUDA' in str(e) or 'GPU' in str(e) or 'cuda' in str(e):
                pytest.skip(f"GPU not available: {e}")
            raise

    def test_optuna_objective_function(self, synthetic_dataset, minimal_optuna_params):
        """Test Optuna optimization runs with minimal trials."""
        X, y = synthetic_dataset

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'device': 'cpu',
                'random_state': 42,
                'max_depth': trial.suggest_int('max_depth', 2, 4),
                'n_estimators': trial.suggest_int('n_estimators', 10, 30, step=10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            }

            cv = StratifiedKFold(n_splits=minimal_optuna_params['n_folds'], shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred, zero_division=0))

            return np.mean(scores)

        sampler = TPESampler(seed=42, n_startup_trials=minimal_optuna_params['n_startup_trials'])
        pruner = MedianPruner(n_startup_trials=2)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(objective, n_trials=minimal_optuna_params['n_trials'], show_progress_bar=False)

        assert len(study.trials) == minimal_optuna_params['n_trials']
        assert study.best_value >= 0  # F1 score should be non-negative
        assert 'max_depth' in study.best_params
        print(f"Best F1: {study.best_value:.4f}, Best params: {study.best_params}")

    def test_device_configuration(self):
        """Test USE_GPU environment variable configuration."""
        # Test default (CPU)
        os.environ.pop('USE_GPU', None)
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        assert use_gpu is False

        # Test GPU enabled
        os.environ['USE_GPU'] = 'true'
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        assert use_gpu is True

        # Test GPU disabled
        os.environ['USE_GPU'] = 'false'
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        assert use_gpu is False

        # Cleanup
        os.environ.pop('USE_GPU', None)

    def test_model_serialization(self, synthetic_dataset, tmp_path):
        """Test model can be saved and loaded."""
        X, y = synthetic_dataset

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.json"
        model.save_model(str(model_path))

        # Load model
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(str(model_path))

        # Verify predictions match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)


@pytest.mark.unit
@pytest.mark.training
class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_train_test_split_proportions(self, synthetic_dataset):
        """Test that train/test split maintains expected proportions."""
        from sklearn.model_selection import train_test_split

        X, y = synthetic_dataset
        test_size = 0.2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        expected_test_size = int(len(X) * test_size)
        assert len(X_test) == expected_test_size
        assert len(X_train) == len(X) - expected_test_size

    def test_stratified_split_preserves_distribution(self, synthetic_dataset):
        """Test stratified split preserves class distribution."""
        from sklearn.model_selection import train_test_split

        X, y = synthetic_dataset

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Check class proportions are similar
        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05
