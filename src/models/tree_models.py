"""Tree-based models for PrimeTRADE."""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class XGBoostModel:
    """
    XGBoost classifier with MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "primetrade_xgboost",
        random_state: int = 42,
    ):
        """
        Initialize XGBoostModel.

        Args:
            experiment_name: MLflow experiment name
            random_state: Random seed
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.model = None

        mlflow.set_experiment(experiment_name)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict] = None,
    ):
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            params: XGBoost parameters
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
            }

        with mlflow.start_run(run_name="xgboost"):
            mlflow.log_params(params)

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train)

            mlflow.sklearn.log_model(self.model, "model")

            logger.info("XGBoost model trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"XGBoost evaluation: {metrics}")

        return metrics


class LightGBMModel:
    """
    LightGBM classifier with MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "primetrade_lightgbm",
        random_state: int = 42,
    ):
        """
        Initialize LightGBMModel.

        Args:
            experiment_name: MLflow experiment name
            random_state: Random seed
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.model = None

        mlflow.set_experiment(experiment_name)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict] = None,
    ):
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            params: LightGBM parameters
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary',
                'random_state': self.random_state,
                'verbose': -1,
            }

        with mlflow.start_run(run_name="lightgbm"):
            mlflow.log_params(params)

            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X_train, y_train)

            mlflow.sklearn.log_model(self.model, "model")

            logger.info("LightGBM model trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"LightGBM evaluation: {metrics}")

        return metrics
