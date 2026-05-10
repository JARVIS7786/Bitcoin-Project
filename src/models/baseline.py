"""Baseline models for PrimeTRADE."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaselineModel:
    """
    Logistic Regression baseline with MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "primetrade_baseline",
        random_state: int = 42,
    ):
        """
        Initialize BaselineModel.

        Args:
            experiment_name: MLflow experiment name
            random_state: Random seed
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.model = None

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs,
    ):
        """
        Train logistic regression model.

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for LogisticRegression
        """
        with mlflow.start_run(run_name="baseline_logistic"):
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", len(X_train))

            # Train model
            self.model = LogisticRegression(random_state=self.random_state, **kwargs)
            self.model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info("Baseline model trained successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
        }

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"Model evaluation: {metrics}")

        return metrics
