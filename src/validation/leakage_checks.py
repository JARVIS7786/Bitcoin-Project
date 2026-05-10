"""Leakage detection for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LeakageDetector:
    """
    Detect data leakage in features.

    Ensures features only use data available before prediction timestamp.
    """

    def __init__(self):
        """Initialize LeakageDetector."""
        pass

    def check_future_leakage(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        timestamp_column: str = 'timestamp',
    ) -> Dict[str, bool]:
        """
        Check if features leak future information.

        Args:
            df: DataFrame with features and target
            feature_columns: List of feature columns
            target_column: Target column
            timestamp_column: Timestamp column

        Returns:
            Dictionary with leakage status per feature
        """
        results = {}

        for feature in feature_columns:
            if feature not in df.columns:
                continue

            # Check if feature has perfect correlation with target (suspicious)
            corr = df[feature].corr(df[target_column])
            if abs(corr) > 0.99:
                results[feature] = True
                logger.warning(f"Potential leakage in '{feature}': correlation={corr:.4f}")
            else:
                results[feature] = False

        return results

    def validate_temporal_order(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp',
    ) -> bool:
        """
        Validate that data is in temporal order.

        Args:
            df: DataFrame to validate
            timestamp_column: Timestamp column

        Returns:
            True if valid, False otherwise
        """
        is_sorted = df[timestamp_column].is_monotonic_increasing
        if not is_sorted:
            logger.error("Data is not in temporal order!")
        return is_sorted


class TemporalValidator:
    """
    Walk-forward validation for time series.

    Ensures proper train/test splits respecting temporal order.
    """

    def __init__(self, n_splits: int = 5, test_size: int = 30):
        """
        Initialize TemporalValidator.

        Args:
            n_splits: Number of validation splits
            test_size: Size of test set (in days or periods)
        """
        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp',
    ):
        """
        Generate train/test splits.

        Args:
            df: DataFrame to split
            timestamp_column: Timestamp column

        Yields:
            Tuples of (train_indices, test_indices)
        """
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        n = len(df)

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start

            if train_end < self.test_size:
                continue

            train_idx = df.index[:train_end].tolist()
            test_idx = df.index[test_start:test_end].tolist()

            yield train_idx, test_idx
