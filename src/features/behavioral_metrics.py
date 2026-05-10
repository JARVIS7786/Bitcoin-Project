"""Behavioral finance metrics for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BehavioralMetrics:
    """
    Calculate behavioral finance metrics for trader analysis.

    Quantifies psychological biases: revenge trading, overconfidence, loss aversion.
    """

    def __init__(self, lookback_window: int = 10):
        """
        Initialize BehavioralMetrics.

        Args:
            lookback_window: Number of periods to look back for calculations
        """
        self.lookback_window = lookback_window

    def calculate_revenge_trading_score(
        self,
        df: pd.DataFrame,
        pnl_column: str = 'pnl',
        leverage_column: str = 'leverage',
        trader_id_column: str = 'trader_id',
    ) -> pd.Series:
        """
        Calculate revenge trading score.

        Measures correlation between prior loss and subsequent leverage increase.
        High positive score indicates revenge trading behavior.

        Args:
            df: DataFrame with trader data
            pnl_column: PnL column name
            leverage_column: Leverage column name
            trader_id_column: Trader ID column name

        Returns:
            Series with revenge trading scores per trader

        Examples:
            >>> metrics = BehavioralMetrics()
            >>> df = pd.DataFrame({
            ...     'trader_id': ['A', 'A', 'A', 'A'],
            ...     'pnl': [-100, -50, 200, -75],
            ...     'leverage': [5, 10, 3, 15]
            ... })
            >>> scores = metrics.calculate_revenge_trading_score(df)
        """
        df = df.sort_values(['trader_id', 'timestamp'] if 'timestamp' in df.columns else 'trader_id')

        def trader_revenge_score(group):
            if len(group) < 2:
                return np.nan

            # Calculate prior loss (negative PnL)
            prior_loss = group[pnl_column].shift(1)
            prior_loss_indicator = (prior_loss < 0).astype(int)

            # Calculate leverage change
            leverage_change = group[leverage_column].diff()

            # Correlation between prior loss and leverage increase
            valid_data = pd.DataFrame({
                'loss': prior_loss_indicator,
                'lev_change': leverage_change
            }).dropna()

            if len(valid_data) < 2:
                return np.nan

            corr = valid_data['loss'].corr(valid_data['lev_change'])
            return corr if not np.isnan(corr) else 0.0

        scores = df.groupby(trader_id_column).apply(trader_revenge_score)
        return scores

    def calculate_overconfidence_ratio(
        self,
        df: pd.DataFrame,
        pnl_column: str = 'pnl',
        position_size_column: str = 'position_size',
        trader_id_column: str = 'trader_id',
        win_streak_threshold: int = 3,
    ) -> pd.Series:
        """
        Calculate overconfidence ratio.

        Measures position size increase after winning streaks.
        Ratio > 1 indicates overconfidence.

        Args:
            df: DataFrame with trader data
            pnl_column: PnL column name
            position_size_column: Position size column name
            trader_id_column: Trader ID column name
            win_streak_threshold: Minimum wins to consider a streak

        Returns:
            Series with overconfidence ratios per trader

        Examples:
            >>> metrics = BehavioralMetrics()
            >>> df = pd.DataFrame({
            ...     'trader_id': ['A'] * 6,
            ...     'pnl': [100, 50, 75, 200, -50, 100],
            ...     'position_size': [1000, 1000, 1000, 2000, 1500, 1000]
            ... })
            >>> ratios = metrics.calculate_overconfidence_ratio(df)
        """
        df = df.sort_values(['trader_id', 'timestamp'] if 'timestamp' in df.columns else 'trader_id')

        def trader_overconfidence(group):
            if len(group) < win_streak_threshold + 1:
                return np.nan

            # Identify winning trades
            wins = (group[pnl_column] > 0).astype(int)

            # Calculate win streaks
            win_streak = wins.rolling(window=win_streak_threshold, min_periods=win_streak_threshold).sum()
            in_streak = (win_streak >= win_streak_threshold)

            # Position size after streak vs before
            pos_after_streak = group.loc[in_streak, position_size_column]
            pos_before_streak = group.loc[~in_streak, position_size_column]

            if len(pos_after_streak) == 0 or len(pos_before_streak) == 0:
                return 1.0

            ratio = pos_after_streak.mean() / pos_before_streak.mean()
            return ratio

        ratios = df.groupby(trader_id_column).apply(trader_overconfidence)
        return ratios

    def calculate_loss_aversion_asymmetry(
        self,
        df: pd.DataFrame,
        pnl_column: str = 'pnl',
        trader_id_column: str = 'trader_id',
    ) -> pd.Series:
        """
        Calculate loss aversion asymmetry.

        Formula: avg_loss / avg_win - 1
        Positive values indicate loss aversion (losses held longer/larger).

        Args:
            df: DataFrame with trader data
            pnl_column: PnL column name
            trader_id_column: Trader ID column name

        Returns:
            Series with loss aversion scores per trader

        Examples:
            >>> metrics = BehavioralMetrics()
            >>> df = pd.DataFrame({
            ...     'trader_id': ['A'] * 6,
            ...     'pnl': [100, -200, 50, -150, 75, -100]
            ... })
            >>> scores = metrics.calculate_loss_aversion_asymmetry(df)
        """
        def trader_loss_aversion(group):
            wins = group[group[pnl_column] > 0][pnl_column]
            losses = group[group[pnl_column] < 0][pnl_column]

            if len(wins) == 0 or len(losses) == 0:
                return np.nan

            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            asymmetry = (avg_loss / avg_win) - 1
            return asymmetry

        scores = df.groupby(trader_id_column).apply(trader_loss_aversion)
        return scores

    def calculate_emotional_leverage_expansion(
        self,
        df: pd.DataFrame,
        leverage_column: str = 'leverage',
        sentiment_column: str = 'fear_greed_value',
        trader_id_column: str = 'trader_id',
        extreme_threshold: float = 20,
    ) -> pd.Series:
        """
        Calculate emotional leverage expansion.

        Measures leverage variance during extreme sentiment periods.
        Higher variance indicates emotional decision-making.

        Args:
            df: DataFrame with trader and sentiment data
            leverage_column: Leverage column name
            sentiment_column: Sentiment column name
            trader_id_column: Trader ID column name
            extreme_threshold: Threshold for extreme sentiment (0-100 scale)

        Returns:
            Series with emotional leverage expansion scores per trader

        Examples:
            >>> metrics = BehavioralMetrics()
            >>> df = pd.DataFrame({
            ...     'trader_id': ['A'] * 10,
            ...     'leverage': [5, 5, 10, 15, 5, 5, 5, 20, 5, 5],
            ...     'fear_greed_value': [50, 50, 10, 5, 50, 50, 50, 95, 50, 50]
            ... })
            >>> scores = metrics.calculate_emotional_leverage_expansion(df)
        """
        def trader_emotional_expansion(group):
            if sentiment_column not in group.columns:
                return np.nan

            # Identify extreme sentiment periods
            extreme_fear = group[sentiment_column] < extreme_threshold
            extreme_greed = group[sentiment_column] > (100 - extreme_threshold)
            extreme_periods = extreme_fear | extreme_greed

            # Leverage variance in extreme vs normal periods
            lev_extreme = group.loc[extreme_periods, leverage_column]
            lev_normal = group.loc[~extreme_periods, leverage_column]

            if len(lev_extreme) < 2 or len(lev_normal) < 2:
                return np.nan

            var_extreme = lev_extreme.var()
            var_normal = lev_normal.var()

            if var_normal == 0:
                return np.nan

            expansion = var_extreme / var_normal
            return expansion

        scores = df.groupby(trader_id_column).apply(trader_emotional_expansion)
        return scores

    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        trader_id_column: str = 'trader_id',
    ) -> pd.DataFrame:
        """
        Calculate all behavioral metrics.

        Args:
            df: DataFrame with trader data
            trader_id_column: Trader ID column name

        Returns:
            DataFrame with all behavioral metrics per trader

        Examples:
            >>> metrics = BehavioralMetrics()
            >>> df = pd.DataFrame({
            ...     'trader_id': ['A'] * 10,
            ...     'pnl': [100, -50, 75, -100, 200, -75, 50, -25, 150, -50],
            ...     'leverage': [5, 10, 5, 15, 5, 20, 5, 10, 5, 15],
            ...     'position_size': [1000] * 10,
            ...     'fear_greed_value': [50, 10, 50, 5, 50, 95, 50, 90, 50, 10]
            ... })
            >>> all_metrics = metrics.calculate_all_metrics(df)
        """
        logger.info("Calculating all behavioral metrics")

        results = pd.DataFrame(index=df[trader_id_column].unique())

        # Revenge trading score
        try:
            results['revenge_trading_score'] = self.calculate_revenge_trading_score(df, trader_id_column=trader_id_column)
        except Exception as e:
            logger.warning(f"Failed to calculate revenge trading score: {e}")
            results['revenge_trading_score'] = np.nan

        # Overconfidence ratio
        try:
            results['overconfidence_ratio'] = self.calculate_overconfidence_ratio(df, trader_id_column=trader_id_column)
        except Exception as e:
            logger.warning(f"Failed to calculate overconfidence ratio: {e}")
            results['overconfidence_ratio'] = np.nan

        # Loss aversion asymmetry
        try:
            results['loss_aversion_asymmetry'] = self.calculate_loss_aversion_asymmetry(df, trader_id_column=trader_id_column)
        except Exception as e:
            logger.warning(f"Failed to calculate loss aversion asymmetry: {e}")
            results['loss_aversion_asymmetry'] = np.nan

        # Emotional leverage expansion
        try:
            results['emotional_leverage_expansion'] = self.calculate_emotional_leverage_expansion(df, trader_id_column=trader_id_column)
        except Exception as e:
            logger.warning(f"Failed to calculate emotional leverage expansion: {e}")
            results['emotional_leverage_expansion'] = np.nan

        logger.info(f"Calculated behavioral metrics for {len(results)} traders")

        return results
