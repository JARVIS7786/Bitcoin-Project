"""Tests for BehavioralMetrics class."""

import pytest
import pandas as pd
import numpy as np
from src.features.behavioral_metrics import BehavioralMetrics


@pytest.fixture
def sample_trader_data():
    """Create sample trader data."""
    return pd.DataFrame({
        'trader_id': ['A'] * 10 + ['B'] * 10,
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='1H'),
        'pnl': [100, -50, 75, -100, 200, -75, 50, -25, 150, -50] * 2,
        'leverage': [5, 10, 5, 15, 5, 20, 5, 10, 5, 15] * 2,
        'position_size': [1000, 1000, 1000, 2000, 1000, 2000, 1000, 1500, 1000, 1500] * 2,
        'fear_greed_value': [50, 10, 50, 5, 50, 95, 50, 90, 50, 10] * 2
    })


def test_behavioral_metrics_initialization():
    """Test BehavioralMetrics initialization."""
    metrics = BehavioralMetrics(lookback_window=5)
    assert metrics.lookback_window == 5


def test_calculate_revenge_trading_score(sample_trader_data):
    """Test revenge trading score calculation."""
    metrics = BehavioralMetrics()
    scores = metrics.calculate_revenge_trading_score(sample_trader_data)

    assert isinstance(scores, pd.Series)
    assert len(scores) == 2  # Two traders
    assert 'A' in scores.index
    assert 'B' in scores.index


def test_calculate_revenge_trading_score_single_trade():
    """Test revenge trading score with insufficient data."""
    df = pd.DataFrame({
        'trader_id': ['A'],
        'pnl': [100],
        'leverage': [5]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_revenge_trading_score(df)

    assert pd.isna(scores['A'])


def test_calculate_overconfidence_ratio(sample_trader_data):
    """Test overconfidence ratio calculation."""
    metrics = BehavioralMetrics()
    ratios = metrics.calculate_overconfidence_ratio(sample_trader_data, win_streak_threshold=2)

    assert isinstance(ratios, pd.Series)
    assert len(ratios) == 2


def test_calculate_overconfidence_ratio_no_streaks():
    """Test overconfidence ratio with no win streaks."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 5,
        'pnl': [100, -50, 100, -50, 100],
        'position_size': [1000] * 5
    })

    metrics = BehavioralMetrics()
    ratios = metrics.calculate_overconfidence_ratio(df, win_streak_threshold=3)

    # Should return NaN or 1.0 when no streaks
    assert pd.isna(ratios['A']) or ratios['A'] == 1.0


def test_calculate_loss_aversion_asymmetry(sample_trader_data):
    """Test loss aversion asymmetry calculation."""
    metrics = BehavioralMetrics()
    scores = metrics.calculate_loss_aversion_asymmetry(sample_trader_data)

    assert isinstance(scores, pd.Series)
    assert len(scores) == 2


def test_calculate_loss_aversion_asymmetry_only_wins():
    """Test loss aversion with only winning trades."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 5,
        'pnl': [100, 50, 75, 200, 150]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_loss_aversion_asymmetry(df)

    assert pd.isna(scores['A'])


def test_calculate_loss_aversion_asymmetry_only_losses():
    """Test loss aversion with only losing trades."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 5,
        'pnl': [-100, -50, -75, -200, -150]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_loss_aversion_asymmetry(df)

    assert pd.isna(scores['A'])


def test_calculate_emotional_leverage_expansion(sample_trader_data):
    """Test emotional leverage expansion calculation."""
    metrics = BehavioralMetrics()
    scores = metrics.calculate_emotional_leverage_expansion(sample_trader_data)

    assert isinstance(scores, pd.Series)
    assert len(scores) == 2


def test_calculate_emotional_leverage_expansion_no_extreme():
    """Test emotional leverage with no extreme sentiment."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 10,
        'leverage': [5] * 10,
        'fear_greed_value': [50] * 10  # All neutral
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_emotional_leverage_expansion(df)

    # Should return NaN when no extreme periods
    assert pd.isna(scores['A'])


def test_calculate_emotional_leverage_expansion_missing_sentiment():
    """Test emotional leverage without sentiment column."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 10,
        'leverage': [5] * 10
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_emotional_leverage_expansion(df)

    assert pd.isna(scores['A'])


def test_calculate_all_metrics(sample_trader_data):
    """Test calculating all behavioral metrics."""
    metrics = BehavioralMetrics()
    all_metrics = metrics.calculate_all_metrics(sample_trader_data)

    assert isinstance(all_metrics, pd.DataFrame)
    assert len(all_metrics) == 2  # Two traders
    assert 'revenge_trading_score' in all_metrics.columns
    assert 'overconfidence_ratio' in all_metrics.columns
    assert 'loss_aversion_asymmetry' in all_metrics.columns
    assert 'emotional_leverage_expansion' in all_metrics.columns


def test_calculate_all_metrics_minimal_data():
    """Test all metrics with minimal data."""
    df = pd.DataFrame({
        'trader_id': ['A', 'A'],
        'pnl': [100, -50],
        'leverage': [5, 10],
        'position_size': [1000, 1000],
        'fear_greed_value': [50, 10]
    })

    metrics = BehavioralMetrics()
    all_metrics = metrics.calculate_all_metrics(df)

    assert isinstance(all_metrics, pd.DataFrame)
    assert len(all_metrics) == 1


def test_revenge_trading_positive_correlation():
    """Test revenge trading detects positive correlation."""
    # Create data where losses lead to leverage increases
    df = pd.DataFrame({
        'trader_id': ['A'] * 10,
        'pnl': [-100, 0, -50, 0, -75, 0, -200, 0, -150, 0],
        'leverage': [5, 10, 5, 15, 5, 20, 5, 25, 5, 30]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_revenge_trading_score(df)

    # Should detect positive correlation
    assert scores['A'] > 0


def test_overconfidence_ratio_greater_than_one():
    """Test overconfidence ratio > 1 after win streaks."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 10,
        'pnl': [100, 100, 100, 0, 0, 100, 100, 100, 0, 0],
        'position_size': [1000, 1000, 1000, 2000, 1000, 1000, 1000, 1000, 2000, 1000]
    })

    metrics = BehavioralMetrics()
    ratios = metrics.calculate_overconfidence_ratio(df, win_streak_threshold=3)

    # Position size increases after win streaks
    assert ratios['A'] >= 1.0


def test_loss_aversion_positive_asymmetry():
    """Test loss aversion detects larger losses than wins."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 10,
        'pnl': [50, -100, 50, -150, 50, -200, 50, -100, 50, -150]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_loss_aversion_asymmetry(df)

    # Losses are larger than wins
    assert scores['A'] > 0


def test_emotional_leverage_high_variance():
    """Test emotional leverage detects high variance in extreme sentiment."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 20,
        'leverage': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # Normal periods
                     5, 20, 5, 25, 5, 30, 5, 15, 5, 20],  # Extreme periods with high variance
        'fear_greed_value': [50] * 10 + [10, 5, 95, 90, 10, 5, 95, 90, 10, 5]
    })

    metrics = BehavioralMetrics()
    scores = metrics.calculate_emotional_leverage_expansion(df, extreme_threshold=20)

    # Variance should be higher in extreme periods
    assert scores['A'] > 1.0


def test_multiple_traders():
    """Test metrics work with multiple traders."""
    df = pd.DataFrame({
        'trader_id': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
        'pnl': [100, -50, 75, -100, 200] * 3,
        'leverage': [5, 10, 5, 15, 5] * 3,
        'position_size': [1000] * 15,
        'fear_greed_value': [50, 10, 50, 5, 50] * 3
    })

    metrics = BehavioralMetrics()
    all_metrics = metrics.calculate_all_metrics(df)

    assert len(all_metrics) == 3
    assert set(all_metrics.index) == {'A', 'B', 'C'}
