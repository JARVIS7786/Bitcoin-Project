"""Tests for sentiment features."""

import pytest
import pandas as pd
import numpy as np
from src.features.sentiment import create_sentiment_features


def test_create_sentiment_features():
    """Test sentiment feature creation."""
    df = pd.DataFrame({
        'fear_greed_value': [30, 40, 50, 60, 70, 80, 90, 20, 10, 95]
    })

    result = create_sentiment_features(df, window_sizes=[3])

    assert 'sentiment_ma_3' in result.columns
    assert 'sentiment_std_3' in result.columns
    assert 'is_extreme_fear' in result.columns
    assert 'is_extreme_greed' in result.columns
