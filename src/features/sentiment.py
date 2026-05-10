"""Sentiment feature engineering for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_sentiment_features(
    df: pd.DataFrame,
    sentiment_column: str = 'fear_greed_value',
    window_sizes: list = [3, 7, 14],
) -> pd.DataFrame:
    """
    Create sentiment-based features.

    Args:
        df: DataFrame with sentiment data
        sentiment_column: Sentiment column name
        window_sizes: Rolling window sizes for features

    Returns:
        DataFrame with sentiment features
    """
    df = df.copy()

    # Rolling statistics
    for window in window_sizes:
        df[f'sentiment_ma_{window}'] = df[sentiment_column].rolling(window).mean()
        df[f'sentiment_std_{window}'] = df[sentiment_column].rolling(window).std()
        df[f'sentiment_min_{window}'] = df[sentiment_column].rolling(window).min()
        df[f'sentiment_max_{window}'] = df[sentiment_column].rolling(window).max()

    # Sentiment momentum
    df['sentiment_change_1'] = df[sentiment_column].diff(1)
    df['sentiment_change_7'] = df[sentiment_column].diff(7)

    # Extreme sentiment indicators
    df['is_extreme_fear'] = (df[sentiment_column] < 20).astype(int)
    df['is_extreme_greed'] = (df[sentiment_column] > 80).astype(int)

    logger.info(f"Created {len([c for c in df.columns if 'sentiment' in c])} sentiment features")

    return df


def create_trader_features(
    df: pd.DataFrame,
    trader_id_column: str = 'trader_id',
) -> pd.DataFrame:
    """
    Create trader behavior features.

    Args:
        df: DataFrame with trader data
        trader_id_column: Trader ID column

    Returns:
        DataFrame with trader features
    """
    df = df.copy()

    # Aggregate by trader
    df['win_rate'] = df.groupby(trader_id_column)['pnl'].transform(lambda x: (x > 0).mean())
    df['avg_pnl'] = df.groupby(trader_id_column)['pnl'].transform('mean')
    df['avg_leverage'] = df.groupby(trader_id_column)['leverage'].transform('mean')
    df['trade_count'] = df.groupby(trader_id_column).cumcount() + 1

    logger.info("Created trader behavior features")

    return df


def create_risk_features(
    df: pd.DataFrame,
    window_sizes: list = [7, 14],
) -> pd.DataFrame:
    """
    Create risk-based features.

    Args:
        df: DataFrame with trading data
        window_sizes: Rolling window sizes

    Returns:
        DataFrame with risk features
    """
    df = df.copy()

    for window in window_sizes:
        df[f'pnl_volatility_{window}'] = df['pnl'].rolling(window).std()
        df[f'leverage_volatility_{window}'] = df['leverage'].rolling(window).std()

    logger.info("Created risk features")

    return df


def create_temporal_features(
    df: pd.DataFrame,
    timestamp_column: str = 'timestamp',
) -> pd.DataFrame:
    """
    Create time-based features.

    Args:
        df: DataFrame with timestamp
        timestamp_column: Timestamp column name

    Returns:
        DataFrame with temporal features
    """
    df = df.copy()

    df['hour'] = df[timestamp_column].dt.hour
    df['day_of_week'] = df[timestamp_column].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    logger.info("Created temporal features")

    return df
