"""Tests for FeatureStabilityValidator class."""

import pytest
import pandas as pd
import numpy as np
from src.validation.feature_stability import FeatureStabilityValidator


@pytest.fixture
def stable_distributions():
    """Create two samples from same distribution."""
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(0, 1, 1000))
    return baseline, current


@pytest.fixture
def unstable_distributions():
    """Create two samples from different distributions."""
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(2, 1, 1000))  # Shifted mean
    return baseline, current


def test_feature_stability_validator_initialization():
    """Test FeatureStabilityValidator initialization."""
    validator = FeatureStabilityValidator(n_bins=20, psi_threshold=0.1, ks_threshold=0.01)
    assert validator.n_bins == 20
    assert validator.psi_threshold == 0.1
    assert validator.ks_threshold == 0.01


def test_calculate_psi_stable(stable_distributions):
    """Test PSI calculation with stable distributions."""
    baseline, current = stable_distributions
    validator = FeatureStabilityValidator()
    psi = validator.calculate_psi(baseline, current)

    assert isinstance(psi, float)
    assert psi < 0.2  # Should be stable


def test_calculate_psi_unstable(unstable_distributions):
    """Test PSI calculation with unstable distributions."""
    baseline, current = unstable_distributions
    validator = FeatureStabilityValidator()
    psi = validator.calculate_psi(baseline, current)

    assert isinstance(psi, float)
    assert psi > 0.2  # Should be unstable


def test_calculate_psi_identical():
    """Test PSI with identical distributions."""
    data = pd.Series(np.random.randn(100))
    validator = FeatureStabilityValidator()
    psi = validator.calculate_psi(data, data)

    assert psi < 0.01  # Should be very close to 0


def test_calculate_psi_empty_data():
    """Test PSI with empty data raises ValueError."""
    validator = FeatureStabilityValidator()
    with pytest.raises(ValueError, match="empty data"):
        validator.calculate_psi(pd.Series([]), pd.Series([1, 2, 3]))


def test_calculate_psi_custom_bins():
    """Test PSI with custom bins."""
    baseline = pd.Series(np.random.randn(100))
    current = pd.Series(np.random.randn(100))
    bins = np.linspace(-3, 3, 11)

    validator = FeatureStabilityValidator()
    psi = validator.calculate_psi(baseline, current, bins=bins)

    assert isinstance(psi, float)


def test_ks_test_same_distribution(stable_distributions):
    """Test KS test with same distribution."""
    baseline, current = stable_distributions
    validator = FeatureStabilityValidator()
    result = validator.ks_test(baseline, current)

    assert 'statistic' in result
    assert 'p_value' in result
    assert result['p_value'] > 0.05  # Should not reject null hypothesis


def test_ks_test_different_distribution(unstable_distributions):
    """Test KS test with different distributions."""
    baseline, current = unstable_distributions
    validator = FeatureStabilityValidator()
    result = validator.ks_test(baseline, current)

    assert 'statistic' in result
    assert 'p_value' in result
    assert result['p_value'] < 0.05  # Should reject null hypothesis


def test_ks_test_empty_data():
    """Test KS test with empty data raises ValueError."""
    validator = FeatureStabilityValidator()
    with pytest.raises(ValueError, match="empty data"):
        validator.ks_test(pd.Series([]), pd.Series([1, 2, 3]))


def test_validate_feature_stable(stable_distributions):
    """Test validating stable feature."""
    baseline, current = stable_distributions
    validator = FeatureStabilityValidator()
    result = validator.validate_feature(baseline, current, 'test_feature')

    assert result['feature'] == 'test_feature'
    assert result['is_stable'] is True
    assert result['psi_stable'] is True
    assert result['ks_stable'] is True


def test_validate_feature_unstable(unstable_distributions):
    """Test validating unstable feature."""
    baseline, current = unstable_distributions
    validator = FeatureStabilityValidator()
    result = validator.validate_feature(baseline, current, 'test_feature')

    assert result['feature'] == 'test_feature'
    assert result['is_stable'] is False


def test_validate_features_all_numeric():
    """Test validating multiple features (all numeric)."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })
    current = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })

    validator = FeatureStabilityValidator()
    results = validator.validate_features(baseline, current)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3
    assert 'feature' in results.columns
    assert 'psi' in results.columns
    assert 'is_stable' in results.columns


def test_validate_features_specific_columns():
    """Test validating specific feature columns."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })
    current = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100)
    })

    validator = FeatureStabilityValidator()
    results = validator.validate_features(baseline, current, feature_columns=['f1', 'f2'])

    assert len(results) == 2
    assert set(results['feature']) == {'f1', 'f2'}


def test_validate_features_missing_column():
    """Test validating with missing column."""
    baseline = pd.DataFrame({'f1': np.random.randn(100)})
    current = pd.DataFrame({'f2': np.random.randn(100)})

    validator = FeatureStabilityValidator()
    results = validator.validate_features(baseline, current, feature_columns=['f1', 'f2'])

    # Should skip missing columns
    assert len(results) == 0


def test_validate_features_sorted_by_psi():
    """Test that results are sorted by PSI."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100)
    })
    current = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.normal(2, 1, 100)  # f2 is unstable
    })

    validator = FeatureStabilityValidator()
    results = validator.validate_features(baseline, current)

    # Results should be sorted by PSI descending
    assert results['psi'].is_monotonic_decreasing


def test_get_unstable_features():
    """Test getting list of unstable features."""
    results = pd.DataFrame({
        'feature': ['f1', 'f2', 'f3'],
        'is_stable': [True, False, True]
    })

    validator = FeatureStabilityValidator()
    unstable = validator.get_unstable_features(results)

    assert unstable == ['f2']


def test_get_unstable_features_empty():
    """Test getting unstable features from empty results."""
    validator = FeatureStabilityValidator()
    unstable = validator.get_unstable_features(pd.DataFrame())

    assert unstable == []


def test_get_unstable_features_all_stable():
    """Test getting unstable features when all are stable."""
    results = pd.DataFrame({
        'feature': ['f1', 'f2'],
        'is_stable': [True, True]
    })

    validator = FeatureStabilityValidator()
    unstable = validator.get_unstable_features(results)

    assert unstable == []


def test_summarize_stability():
    """Test generating stability summary."""
    results = pd.DataFrame({
        'feature': ['f1', 'f2', 'f3'],
        'psi': [0.05, 0.25, 0.10],
        'ks_p_value': [0.8, 0.01, 0.5],
        'is_stable': [True, False, True]
    })

    validator = FeatureStabilityValidator()
    summary = validator.summarize_stability(results)

    assert isinstance(summary, str)
    assert 'Feature Stability Report' in summary
    assert 'Stable features: 2/3' in summary
    assert 'PSI Distribution' in summary


def test_summarize_stability_empty():
    """Test summarizing empty results."""
    validator = FeatureStabilityValidator()
    summary = validator.summarize_stability(pd.DataFrame())

    assert summary == "No features validated"


def test_psi_with_nan_values():
    """Test PSI calculation handles NaN values."""
    baseline = pd.Series([1, 2, 3, np.nan, 5])
    current = pd.Series([1, 2, np.nan, 4, 5])

    validator = FeatureStabilityValidator()
    psi = validator.calculate_psi(baseline, current)

    assert isinstance(psi, float)
    assert not np.isnan(psi)


def test_ks_test_with_nan_values():
    """Test KS test handles NaN values."""
    sample1 = pd.Series([1, 2, 3, np.nan, 5])
    sample2 = pd.Series([1, 2, np.nan, 4, 5])

    validator = FeatureStabilityValidator()
    result = validator.ks_test(sample1, sample2)

    assert not np.isnan(result['statistic'])
    assert not np.isnan(result['p_value'])
