"""Data cleaning utilities for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    """
    Clean and validate sentiment and trading data.

    Handles deduplication, missing values, outliers, and data type validation.
    """

    def __init__(
        self,
        max_missing_ratio: float = 0.3,
        outlier_method: str = "iqr",
        iqr_multiplier: float = 3.0,
    ):
        """
        Initialize DataCleaner.

        Args:
            max_missing_ratio: Max ratio of missing values per column (0-1)
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            iqr_multiplier: IQR multiplier for outlier bounds (default: 3.0)
        """
        self.max_missing_ratio = max_missing_ratio
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates (None = all columns)
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            DataFrame with duplicates removed

        Examples:
            >>> cleaner = DataCleaner()
            >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
            >>> cleaned = cleaner.remove_duplicates(df)
            >>> len(cleaned)
            2
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(df_clean)

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows ({removed_rows/initial_rows*100:.2f}%)")
        else:
            logger.info("No duplicate rows found")

        return df_clean

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "drop_columns",
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                - 'drop_columns': Drop columns exceeding max_missing_ratio
                - 'drop_rows': Drop rows with any missing values
                - 'forward_fill': Forward fill missing values (time series)

        Returns:
            DataFrame with missing values handled

        Examples:
            >>> cleaner = DataCleaner(max_missing_ratio=0.5)
            >>> df = pd.DataFrame({'a': [1, 2, None, 4], 'b': [None, None, None, None]})
            >>> cleaned = cleaner.handle_missing_values(df, strategy='drop_columns')
            >>> 'b' in cleaned.columns
            False
        """
        initial_rows = len(df)
        initial_cols = len(df.columns)

        # Log missing value statistics
        missing_stats = df.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        if len(missing_cols) > 0:
            logger.info(f"Columns with missing values: {dict(missing_cols)}")

        if strategy == "drop_columns":
            # Drop columns with too many missing values
            missing_ratio = df.isnull().sum() / len(df)
            cols_to_drop = missing_ratio[missing_ratio > self.max_missing_ratio].index.tolist()

            if cols_to_drop:
                logger.warning(f"Dropping columns with >{self.max_missing_ratio*100}% missing: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

            # Drop rows with any remaining missing values
            df = df.dropna()

        elif strategy == "drop_rows":
            df = df.dropna()

        elif strategy == "forward_fill":
            df = df.fillna(method='ffill')
            # Drop any remaining NaN (at the start)
            df = df.dropna()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        removed_rows = initial_rows - len(df)
        removed_cols = initial_cols - len(df.columns)

        if removed_rows > 0 or removed_cols > 0:
            logger.info(f"Removed {removed_rows} rows and {removed_cols} columns due to missing values")

        return df

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.Series:
        """
        Detect outliers using IQR method.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers

        Returns:
            Boolean Series indicating outlier rows (True = outlier)

        Examples:
            >>> cleaner = DataCleaner(iqr_multiplier=1.5)
            >>> df = pd.DataFrame({'a': [1, 2, 3, 100]})
            >>> outliers = cleaner.detect_outliers_iqr(df, ['a'])
            >>> outliers.sum()
            1
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, skipping outlier detection")
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = col_outliers.sum()

            if outlier_count > 0:
                logger.info(f"Column {col}: {outlier_count} outliers detected (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")

            outlier_mask = outlier_mask | col_outliers

        return outlier_mask

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers

        Returns:
            DataFrame with outliers removed

        Examples:
            >>> cleaner = DataCleaner()
            >>> df = pd.DataFrame({'a': [1, 2, 3, 100], 'b': [4, 5, 6, 7]})
            >>> cleaned = cleaner.remove_outliers(df, ['a'])
            >>> len(cleaned)
            3
        """
        initial_rows = len(df)

        if self.outlier_method == "iqr":
            outlier_mask = self.detect_outliers_iqr(df, columns)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")

        df_clean = df[~outlier_mask].copy()
        removed_rows = initial_rows - len(df_clean)

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")

        return df_clean

    def validate_data_types(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Validate and convert data types.

        Args:
            df: Input DataFrame
            numeric_columns: Columns that should be numeric
            datetime_columns: Columns that should be datetime

        Returns:
            DataFrame with validated types

        Raises:
            ValueError: If type conversion fails

        Examples:
            >>> cleaner = DataCleaner()
            >>> df = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['2024-01-01', '2024-01-02', '2024-01-03']})
            >>> validated = cleaner.validate_data_types(df, numeric_columns=['a'], datetime_columns=['b'])
            >>> pd.api.types.is_numeric_dtype(validated['a'])
            True
        """
        df = df.copy()

        if numeric_columns:
            for col in numeric_columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in DataFrame")
                    continue

                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.info(f"Converted column {col} to numeric")
                    except Exception as e:
                        raise ValueError(f"Failed to convert {col} to numeric: {e}")

        if datetime_columns:
            for col in datetime_columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in DataFrame")
                    continue

                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted column {col} to datetime")
                    except Exception as e:
                        raise ValueError(f"Failed to convert {col} to datetime: {e}")

        return df

    def clean(
        self,
        df: pd.DataFrame,
        dedup_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        outlier_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run full cleaning pipeline.

        Args:
            df: Input DataFrame
            dedup_columns: Columns for deduplication
            numeric_columns: Columns to validate as numeric
            datetime_columns: Columns to validate as datetime
            outlier_columns: Columns to check for outliers

        Returns:
            Cleaned DataFrame

        Examples:
            >>> cleaner = DataCleaner()
            >>> df = pd.DataFrame({'timestamp': ['2024-01-01', '2024-01-01', '2024-01-02'],
            ...                    'value': [1, 1, 100]})
            >>> cleaned = cleaner.clean(df, dedup_columns=['timestamp'], outlier_columns=['value'])
        """
        logger.info(f"Starting data cleaning pipeline. Initial shape: {df.shape}")

        # Step 1: Remove duplicates
        df = self.remove_duplicates(df, subset=dedup_columns)

        # Step 2: Validate data types
        if numeric_columns or datetime_columns:
            df = self.validate_data_types(df, numeric_columns, datetime_columns)

        # Step 3: Handle missing values
        df = self.handle_missing_values(df, strategy="drop_columns")

        # Step 4: Remove outliers
        if outlier_columns:
            df = self.remove_outliers(df, outlier_columns)

        logger.info(f"Cleaning complete. Final shape: {df.shape}")

        return df
