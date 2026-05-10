"""Tests for DataCleaner class."""

import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicates."""
    return pd.DataFrame({
        'timestamp': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [10, 10, 20, 30],
        'category': ['A', 'A', 'B', 'C']
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    return pd.DataFrame({
        'a': [1, 2, None, 4, 5],
        'b': [None, None, None, None, None],  # 100% missing
        'c': [1, None, 3, 4, 5],  # 20% missing
        'd': [1, 2, 3, 4, 5]  # No missing
    })


@pytest.fixture
def sample_data_with_outliers():
    """Create sample data with outliers."""
    return pd.DataFrame({
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000 is outlier
    })


def test_datacleaner_initialization():
    """Test DataCleaner initialization."""
    cleaner = DataCleaner(max_missing_ratio=0.5, outlier_method="iqr", iqr_multiplier=2.0)
    assert cleaner.max_missing_ratio == 0.5
    assert cleaner.outlier_method == "iqr"
    assert cleaner.iqr_multiplier == 2.0


def test_remove_duplicates_all_columns(sample_data_with_duplicates):
    """Test removing duplicates considering all columns."""
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(sample_data_with_duplicates)
    assert len(cleaned) == 3  # One duplicate removed


def test_remove_duplicates_subset(sample_data_with_duplicates):
    """Test removing duplicates considering subset of columns."""
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(sample_data_with_duplicates, subset=['timestamp'])
    assert len(cleaned) == 3  # Duplicates based on timestamp only


def test_remove_duplicates_keep_last(sample_data_with_duplicates):
    """Test removing duplicates keeping last occurrence."""
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(sample_data_with_duplicates, keep='last')
    assert len(cleaned) == 3


def test_handle_missing_values_drop_columns(sample_data_with_missing):
    """Test handling missing values by dropping columns."""
    cleaner = DataCleaner(max_missing_ratio=0.5)
    cleaned = cleaner.handle_missing_values(sample_data_with_missing, strategy='drop_columns')

    # Column 'b' should be dropped (100% missing)
    assert 'b' not in cleaned.columns
    # Rows with missing values should be dropped
    assert cleaned.isnull().sum().sum() == 0


def test_handle_missing_values_drop_rows(sample_data_with_missing):
    """Test handling missing values by dropping rows."""
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(sample_data_with_missing, strategy='drop_rows')

    # All rows with any missing value should be dropped
    assert len(cleaned) == 1  # Only row with index 4 has no missing values
    assert cleaned.isnull().sum().sum() == 0


def test_handle_missing_values_forward_fill():
    """Test handling missing values with forward fill."""
    df = pd.DataFrame({
        'a': [1, None, None, 4, 5],
        'b': [10, 20, None, 40, 50]
    })
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(df, strategy='forward_fill')

    # Forward fill should propagate values
    assert cleaned['a'].tolist() == [1, 1, 1, 4, 5]
    assert cleaned['b'].tolist() == [10, 20, 20, 40, 50]


def test_handle_missing_values_invalid_strategy():
    """Test handling missing values with invalid strategy raises error."""
    cleaner = DataCleaner()
    df = pd.DataFrame({'a': [1, None, 3]})

    with pytest.raises(ValueError, match="Unknown strategy"):
        cleaner.handle_missing_values(df, strategy='invalid')


def test_detect_outliers_iqr(sample_data_with_outliers):
    """Test outlier detection using IQR method."""
    cleaner = DataCleaner(iqr_multiplier=1.5)
    outliers = cleaner.detect_outliers_iqr(sample_data_with_outliers, ['with_outliers'])

    # The value 1000 should be detected as outlier
    assert outliers.sum() == 1
    assert outliers.iloc[-1] == True  # Last row is outlier


def test_detect_outliers_iqr_no_outliers(sample_data_with_outliers):
    """Test outlier detection when no outliers present."""
    cleaner = DataCleaner(iqr_multiplier=1.5)
    outliers = cleaner.detect_outliers_iqr(sample_data_with_outliers, ['normal'])

    # No outliers in 'normal' column
    assert outliers.sum() == 0


def test_detect_outliers_iqr_missing_column(sample_data_with_outliers):
    """Test outlier detection with missing column."""
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers_iqr(sample_data_with_outliers, ['nonexistent'])

    # Should return all False
    assert outliers.sum() == 0


def test_detect_outliers_iqr_non_numeric():
    """Test outlier detection with non-numeric column."""
    df = pd.DataFrame({'text': ['a', 'b', 'c']})
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers_iqr(df, ['text'])

    # Should return all False for non-numeric
    assert outliers.sum() == 0


def test_remove_outliers(sample_data_with_outliers):
    """Test removing outliers from DataFrame."""
    cleaner = DataCleaner(iqr_multiplier=1.5)
    cleaned = cleaner.remove_outliers(sample_data_with_outliers, ['with_outliers'])

    # One outlier should be removed
    assert len(cleaned) == 9
    assert cleaned['with_outliers'].max() < 100  # Outlier 1000 removed


def test_remove_outliers_invalid_method():
    """Test removing outliers with invalid method raises error."""
    cleaner = DataCleaner(outlier_method='invalid')
    df = pd.DataFrame({'a': [1, 2, 3, 100]})

    with pytest.raises(ValueError, match="Unknown outlier method"):
        cleaner.remove_outliers(df, ['a'])


def test_validate_data_types_numeric():
    """Test validating numeric data types."""
    df = pd.DataFrame({
        'a': ['1', '2', '3'],
        'b': [4, 5, 6]
    })
    cleaner = DataCleaner()
    validated = cleaner.validate_data_types(df, numeric_columns=['a'])

    assert pd.api.types.is_numeric_dtype(validated['a'])
    assert validated['a'].tolist() == [1, 2, 3]


def test_validate_data_types_datetime():
    """Test validating datetime data types."""
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    })
    cleaner = DataCleaner()
    validated = cleaner.validate_data_types(df, datetime_columns=['date'])

    assert pd.api.types.is_datetime64_any_dtype(validated['date'])


def test_validate_data_types_missing_column():
    """Test validating data types with missing column."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    cleaner = DataCleaner()

    # Should not raise error, just log warning
    validated = cleaner.validate_data_types(df, numeric_columns=['nonexistent'])
    assert 'a' in validated.columns


def test_clean_full_pipeline():
    """Test full cleaning pipeline."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [1, 1, 2, 1000],  # Has duplicate and outlier
        'category': ['A', 'A', 'B', 'C']
    })

    cleaner = DataCleaner(iqr_multiplier=1.5)
    cleaned = cleaner.clean(
        df,
        dedup_columns=['timestamp', 'value'],
        numeric_columns=['value'],
        datetime_columns=['timestamp'],
        outlier_columns=['value']
    )

    # Should remove duplicate and outlier
    assert len(cleaned) == 2
    assert cleaned['value'].max() < 100  # Outlier removed
    assert pd.api.types.is_datetime64_any_dtype(cleaned['timestamp'])


def test_clean_preserves_data_integrity():
    """Test that cleaning preserves valid data."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })

    cleaner = DataCleaner()
    cleaned = cleaner.clean(df, numeric_columns=['a', 'b'])

    # No data should be removed
    assert len(cleaned) == 5
    pd.testing.assert_frame_equal(df, cleaned)
