"""Tests for regime detection."""

import pytest
import pandas as pd
from src.regimes.fear_regime import FearGreedRegimeDetector


def test_fear_greed_regime_detector():
    """Test regime detection."""
    df = pd.DataFrame({
        'fear_greed_value': [10, 30, 50, 70, 90]
    })

    detector = FearGreedRegimeDetector()
    result = detector.detect_regime(df)

    assert 'regime' in result.columns
    assert result['regime'].iloc[0] == 'extreme_fear'
    assert result['regime'].iloc[2] == 'neutral'
    assert result['regime'].iloc[4] == 'extreme_greed'
