"""Tests for SignalDecayAnalyzer class."""

import pytest
import pandas as pd
import numpy as np
from src.research.signal_decay import SignalDecayAnalyzer


@pytest.fixture
def decaying_signal_data():
    """Create data with exponentially decaying correlation."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    # Signal with exponential decay
    signal = np.random.randn(n)
    target = np.zeros(n)

    # Create lagged relationship with decay
    for i in range(1, n):
        # Correlation decays exponentially with time
        decay_factor = np.exp(-0.1 * min(i, 20))
        target[i] = decay_factor * signal[max(0, i-5)] + np.random.randn() * 0.5

    return pd.DataFrame({'signal': signal, 'target': target})


@pytest.fixture
def sample_decay_df():
    """Create sample decay DataFrame."""
    return pd.DataFrame({
        'lag': [0, 1, 2, 3, 4, 5],
        'correlation': [0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        'abs_correlation': [0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        'n_samples': [100, 99, 98, 97, 96, 95]
    })


def test_signal_decay_analyzer_initialization():
    """Test SignalDecayAnalyzer initialization."""
    analyzer = SignalDecayAnalyzer()
    assert analyzer is not None


def test_exponential_decay_function():
    """Test exponential decay function."""
    t = np.array([0, 1, 2, 3, 4])
    initial = 1.0
    decay_rate = 0.5

    result = SignalDecayAnalyzer.exponential_decay(t, initial, decay_rate)

    # Check values
    assert result[0] == pytest.approx(1.0)  # At t=0
    assert result[1] == pytest.approx(np.exp(-0.5))  # At t=1
    assert result[4] == pytest.approx(np.exp(-2.0))  # At t=4


def test_calculate_correlation_decay(decaying_signal_data):
    """Test calculating correlation decay."""
    analyzer = SignalDecayAnalyzer()
    decay_df = analyzer.calculate_correlation_decay(
        decaying_signal_data,
        'signal',
        'target',
        max_lag=10
    )

    assert isinstance(decay_df, pd.DataFrame)
    assert 'lag' in decay_df.columns
    assert 'correlation' in decay_df.columns
    assert 'abs_correlation' in decay_df.columns
    assert 'n_samples' in decay_df.columns
    assert len(decay_df) <= 11  # 0 to max_lag inclusive


def test_calculate_correlation_decay_missing_column(decaying_signal_data):
    """Test correlation decay with missing column raises ValueError."""
    analyzer = SignalDecayAnalyzer()

    with pytest.raises(ValueError, match="not found"):
        analyzer.calculate_correlation_decay(
            decaying_signal_data,
            'nonexistent',
            'target'
        )


def test_calculate_correlation_decay_lag_zero(decaying_signal_data):
    """Test that lag=0 gives direct correlation."""
    analyzer = SignalDecayAnalyzer()
    decay_df = analyzer.calculate_correlation_decay(
        decaying_signal_data,
        'signal',
        'target',
        max_lag=5
    )

    # Lag 0 should be present
    assert 0 in decay_df['lag'].values


def test_fit_exponential_decay(sample_decay_df):
    """Test fitting exponential decay model."""
    analyzer = SignalDecayAnalyzer()
    params = analyzer.fit_exponential_decay(sample_decay_df)

    assert 'initial' in params
    assert 'decay_rate' in params
    assert 'half_life' in params
    assert 'r_squared' in params

    # Check parameter ranges
    assert params['initial'] > 0
    assert params['decay_rate'] > 0
    assert params['half_life'] > 0
    assert 0 <= params['r_squared'] <= 1


def test_fit_exponential_decay_insufficient_data():
    """Test fitting with insufficient data raises ValueError."""
    analyzer = SignalDecayAnalyzer()
    df = pd.DataFrame({
        'lag': [0, 1],
        'correlation': [0.8, 0.6],
        'abs_correlation': [0.8, 0.6]
    })

    with pytest.raises(ValueError, match="at least 3 data points"):
        analyzer.fit_exponential_decay(df)


def test_fit_exponential_decay_use_abs():
    """Test fitting with absolute correlation values."""
    analyzer = SignalDecayAnalyzer()
    df = pd.DataFrame({
        'lag': [0, 1, 2, 3],
        'correlation': [-0.8, -0.6, -0.4, -0.2],  # Negative correlations
        'abs_correlation': [0.8, 0.6, 0.4, 0.2]
    })

    params = analyzer.fit_exponential_decay(df, use_abs=True)
    assert params['initial'] > 0  # Should use absolute values


def test_calculate_signal_strength(sample_decay_df):
    """Test calculating signal strength metrics."""
    analyzer = SignalDecayAnalyzer()
    strength = analyzer.calculate_signal_strength(sample_decay_df, threshold=0.3)

    assert 'max_correlation' in strength
    assert 'max_lag' in strength
    assert 'effective_window' in strength
    assert 'avg_correlation' in strength

    # Check values
    assert strength['max_correlation'] == 0.8  # Maximum in sample data
    assert strength['max_lag'] == 0  # At lag 0
    assert strength['effective_window'] == 4  # Lags 0-3 are >= 0.3


def test_calculate_signal_strength_empty_df():
    """Test signal strength with empty DataFrame raises ValueError."""
    analyzer = SignalDecayAnalyzer()
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Empty decay DataFrame"):
        analyzer.calculate_signal_strength(df)


def test_analyze_multiple_signals():
    """Test analyzing multiple signals."""
    np.random.seed(42)
    n = 100

    # Create signals with different decay rates
    signal1 = np.random.randn(n)
    signal2 = np.random.randn(n)
    target = np.zeros(n)

    for i in range(5, n):
        target[i] = 0.5 * signal1[i-2] + 0.3 * signal2[i-3] + np.random.randn() * 0.1

    df = pd.DataFrame({
        'signal1': signal1,
        'signal2': signal2,
        'target': target
    })

    analyzer = SignalDecayAnalyzer()
    results = analyzer.analyze_multiple_signals(
        df,
        ['signal1', 'signal2'],
        'target',
        max_lag=10
    )

    assert isinstance(results, pd.DataFrame)
    assert 'signal' in results.columns
    assert 'half_life' in results.columns
    assert 'decay_rate' in results.columns
    assert len(results) <= 2


def test_analyze_multiple_signals_empty_list():
    """Test analyzing with empty signal list."""
    df = pd.DataFrame({'target': [1, 2, 3]})
    analyzer = SignalDecayAnalyzer()

    results = analyzer.analyze_multiple_signals(df, [], 'target')
    assert len(results) == 0


def test_get_optimal_window():
    """Test calculating optimal window."""
    analyzer = SignalDecayAnalyzer()
    decay_params = {
        'initial': 0.8,
        'decay_rate': 0.1,
        'half_life': 6.93,
        'r_squared': 0.95
    }

    window = analyzer.get_optimal_window(decay_params, min_correlation=0.2)

    assert isinstance(window, int)
    assert window > 0


def test_get_optimal_window_zero_decay():
    """Test optimal window with zero decay rate."""
    analyzer = SignalDecayAnalyzer()
    decay_params = {
        'initial': 0.8,
        'decay_rate': 0.0,  # No decay
        'half_life': np.inf,
        'r_squared': 0.95
    }

    window = analyzer.get_optimal_window(decay_params)
    assert window == 24  # Default fallback


def test_get_optimal_window_low_initial():
    """Test optimal window when initial correlation is below threshold."""
    analyzer = SignalDecayAnalyzer()
    decay_params = {
        'initial': 0.05,  # Below min_correlation
        'decay_rate': 0.1,
        'half_life': 6.93,
        'r_squared': 0.95
    }

    window = analyzer.get_optimal_window(decay_params, min_correlation=0.1)
    assert window == 0


def test_half_life_calculation():
    """Test that half-life is correctly calculated."""
    analyzer = SignalDecayAnalyzer()

    # Create perfect exponential decay
    t = np.arange(20)
    decay_rate = 0.1
    y = np.exp(-decay_rate * t)

    df = pd.DataFrame({
        'lag': t,
        'correlation': y,
        'abs_correlation': y
    })

    params = analyzer.fit_exponential_decay(df)

    # Half-life should be ln(2) / decay_rate
    expected_half_life = np.log(2) / decay_rate
    assert params['half_life'] == pytest.approx(expected_half_life, rel=0.1)


def test_correlation_decay_with_min_periods():
    """Test correlation decay respects min_periods."""
    df = pd.DataFrame({
        'signal': [1, 2, 3, 4, 5],
        'target': [2, 4, 6, 8, 10]
    })

    analyzer = SignalDecayAnalyzer()
    decay_df = analyzer.calculate_correlation_decay(
        df,
        'signal',
        'target',
        max_lag=3,
        min_periods=3
    )

    # With only 5 rows and min_periods=3, high lags should be excluded
    assert len(decay_df) <= 3


def test_r_squared_perfect_fit():
    """Test R-squared for perfect exponential fit."""
    analyzer = SignalDecayAnalyzer()

    # Create perfect exponential decay
    t = np.arange(10)
    initial = 1.0
    decay_rate = 0.2
    y = SignalDecayAnalyzer.exponential_decay(t, initial, decay_rate)

    df = pd.DataFrame({
        'lag': t,
        'correlation': y,
        'abs_correlation': y
    })

    params = analyzer.fit_exponential_decay(df)

    # R-squared should be very close to 1 for perfect fit
    assert params['r_squared'] > 0.99
