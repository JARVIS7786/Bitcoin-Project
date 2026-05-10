"""Tests for GrangerCausalityAnalyzer class."""

import pytest
import pandas as pd
import numpy as np
from src.research.causality import GrangerCausalityAnalyzer


@pytest.fixture
def causal_data():
    """Create data where X causes Y with lag."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = np.zeros(n)

    # Y depends on lagged X (lag=2)
    for i in range(2, n):
        y[i] = 0.5 * x[i-2] + 0.3 * y[i-1] + np.random.randn() * 0.1

    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def non_causal_data():
    """Create independent data with no causal relationship."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n)
    })


@pytest.fixture
def bidirectional_data():
    """Create data with bidirectional causality."""
    np.random.seed(42)
    n = 100
    x = np.zeros(n)
    y = np.zeros(n)

    # Initialize
    x[0] = np.random.randn()
    y[0] = np.random.randn()

    # Bidirectional feedback
    for i in range(1, n):
        x[i] = 0.3 * y[i-1] + np.random.randn() * 0.1
        y[i] = 0.3 * x[i-1] + np.random.randn() * 0.1

    return pd.DataFrame({'x': x, 'y': y})


def test_granger_causality_analyzer_initialization():
    """Test GrangerCausalityAnalyzer initialization."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5, significance_level=0.01)
    assert analyzer.max_lag == 5
    assert analyzer.significance_level == 0.01


def test_test_granger_causality_causal(causal_data):
    """Test Granger causality with causal data."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5, significance_level=0.05)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y')

    assert 'is_causal' in result
    assert 'best_lag' in result
    assert 'p_values' in result
    assert 'f_statistics' in result
    assert result['cause'] == 'x'
    assert result['effect'] == 'y'
    assert isinstance(result['is_causal'], bool)
    assert isinstance(result['best_lag'], int)


def test_test_granger_causality_non_causal(non_causal_data):
    """Test Granger causality with non-causal data."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5, significance_level=0.05)
    result = analyzer.test_granger_causality(non_causal_data, 'x', 'y')

    # With random data, should typically not find causality
    assert 'is_causal' in result
    assert isinstance(result['is_causal'], bool)


def test_test_granger_causality_missing_column(causal_data):
    """Test Granger causality with missing column raises ValueError."""
    analyzer = GrangerCausalityAnalyzer()
    with pytest.raises(ValueError, match="not found"):
        analyzer.test_granger_causality(causal_data, 'nonexistent', 'y')


def test_test_granger_causality_insufficient_data():
    """Test Granger causality with insufficient data raises ValueError."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    analyzer = GrangerCausalityAnalyzer(max_lag=10)

    with pytest.raises(ValueError, match="Insufficient data"):
        analyzer.test_granger_causality(df, 'x', 'y')


def test_test_granger_causality_custom_max_lag(causal_data):
    """Test Granger causality with custom max_lag."""
    analyzer = GrangerCausalityAnalyzer(max_lag=10)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y', max_lag=3)

    # Should use custom max_lag=3
    assert len(result['p_values']) == 3
    assert max(result['p_values'].keys()) == 3


def test_test_bidirectional_causality(bidirectional_data):
    """Test bidirectional causality detection."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5, significance_level=0.05)
    result = analyzer.test_bidirectional_causality(bidirectional_data, 'x', 'y')

    assert 'x_causes_y' in result
    assert 'y_causes_x' in result
    assert 'bidirectional' in result
    assert isinstance(result['bidirectional'], bool)


def test_test_bidirectional_causality_non_causal(non_causal_data):
    """Test bidirectional causality with non-causal data."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5)
    result = analyzer.test_bidirectional_causality(non_causal_data, 'x', 'y')

    # Should typically not find bidirectional causality in random data
    assert 'bidirectional' in result


def test_test_multiple_causes():
    """Test multiple causes against single effect."""
    np.random.seed(42)
    n = 100

    # Create data where x1 and x2 both cause y
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.zeros(n)

    for i in range(2, n):
        y[i] = 0.4 * x1[i-1] + 0.3 * x2[i-2] + np.random.randn() * 0.1

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': np.random.randn(n), 'y': y})

    analyzer = GrangerCausalityAnalyzer(max_lag=5)
    results = analyzer.test_multiple_causes(df, ['x1', 'x2', 'x3'], 'y')

    assert isinstance(results, pd.DataFrame)
    assert 'cause' in results.columns
    assert 'effect' in results.columns
    assert 'is_causal' in results.columns
    assert 'p_value' in results.columns
    assert len(results) <= 3  # May be less if some tests fail


def test_test_multiple_causes_empty_list(causal_data):
    """Test multiple causes with empty cause list."""
    analyzer = GrangerCausalityAnalyzer()
    results = analyzer.test_multiple_causes(causal_data, [], 'y')

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 0


def test_get_causal_graph():
    """Test building causal graph."""
    np.random.seed(42)
    n = 100

    # Create simple causal chain: x1 → x2 → x3
    x1 = np.random.randn(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)

    for i in range(1, n):
        x2[i] = 0.5 * x1[i-1] + np.random.randn() * 0.1
        x3[i] = 0.5 * x2[i-1] + np.random.randn() * 0.1

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

    analyzer = GrangerCausalityAnalyzer(max_lag=3)
    graph = analyzer.get_causal_graph(df, ['x1', 'x2', 'x3'])

    assert isinstance(graph, pd.DataFrame)
    assert graph.shape == (3, 3)
    assert list(graph.index) == ['x1', 'x2', 'x3']
    assert list(graph.columns) == ['x1', 'x2', 'x3']
    # Diagonal should be False (no self-causation)
    assert not graph.loc['x1', 'x1']
    assert not graph.loc['x2', 'x2']
    assert not graph.loc['x3', 'x3']


def test_get_causal_graph_single_variable(causal_data):
    """Test causal graph with single variable."""
    analyzer = GrangerCausalityAnalyzer()
    graph = analyzer.get_causal_graph(causal_data, ['x'])

    assert graph.shape == (1, 1)
    assert not graph.loc['x', 'x']  # No self-causation


def test_summarize_results(causal_data):
    """Test summarizing causality test results."""
    analyzer = GrangerCausalityAnalyzer(max_lag=3)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y')
    summary = analyzer.summarize_results(result)

    assert isinstance(summary, str)
    assert 'Granger Causality Test' in summary
    assert 'x → y' in summary
    assert 'Best lag' in summary
    assert 'P-value' in summary


def test_p_values_structure(causal_data):
    """Test that p_values dictionary has correct structure."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y')

    # Should have p-values for lags 1 through max_lag
    assert len(result['p_values']) == 5
    assert all(lag in result['p_values'] for lag in range(1, 6))
    assert all(0 <= p <= 1 for p in result['p_values'].values())


def test_f_statistics_structure(causal_data):
    """Test that f_statistics dictionary has correct structure."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y')

    # Should have F-statistics for lags 1 through max_lag
    assert len(result['f_statistics']) == 5
    assert all(lag in result['f_statistics'] for lag in range(1, 6))
    assert all(f >= 0 for f in result['f_statistics'].values())


def test_best_lag_selection(causal_data):
    """Test that best_lag corresponds to minimum p-value."""
    analyzer = GrangerCausalityAnalyzer(max_lag=5)
    result = analyzer.test_granger_causality(causal_data, 'x', 'y')

    best_lag = result['best_lag']
    best_p = result['p_values'][best_lag]

    # best_lag should have the minimum p-value
    assert best_p == min(result['p_values'].values())
