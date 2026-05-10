"""Fear/Greed regime detection for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FearGreedRegimeDetector:
    """
    Detect market regimes based on Fear & Greed Index.

    Classifies market states: extreme_fear, fear, neutral, greed, extreme_greed.
    """

    def __init__(
        self,
        extreme_fear_threshold: float = 20,
        fear_threshold: float = 40,
        greed_threshold: float = 60,
        extreme_greed_threshold: float = 80,
    ):
        """
        Initialize FearGreedRegimeDetector.

        Args:
            extreme_fear_threshold: Threshold for extreme fear (< this value)
            fear_threshold: Threshold for fear
            greed_threshold: Threshold for greed
            extreme_greed_threshold: Threshold for extreme greed (> this value)
        """
        self.extreme_fear_threshold = extreme_fear_threshold
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        self.extreme_greed_threshold = extreme_greed_threshold

    def detect_regime(
        self,
        df: pd.DataFrame,
        sentiment_column: str = 'fear_greed_value',
    ) -> pd.DataFrame:
        """
        Detect market regime for each row.

        Args:
            df: DataFrame with sentiment data
            sentiment_column: Sentiment column name

        Returns:
            DataFrame with regime column added

        Examples:
            >>> detector = FearGreedRegimeDetector()
            >>> df = pd.DataFrame({'fear_greed_value': [10, 30, 50, 70, 90]})
            >>> result = detector.detect_regime(df)
            >>> result['regime'].tolist()
            ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        """
        df = df.copy()

        def classify_regime(value):
            if pd.isna(value):
                return 'unknown'
            elif value < self.extreme_fear_threshold:
                return 'extreme_fear'
            elif value < self.fear_threshold:
                return 'fear'
            elif value < self.greed_threshold:
                return 'neutral'
            elif value < self.extreme_greed_threshold:
                return 'greed'
            else:
                return 'extreme_greed'

        df['regime'] = df[sentiment_column].apply(classify_regime)

        logger.info(f"Detected regimes: {df['regime'].value_counts().to_dict()}")

        return df

    def get_regime_statistics(
        self,
        df: pd.DataFrame,
        regime_column: str = 'regime',
    ) -> pd.DataFrame:
        """
        Calculate statistics by regime.

        Args:
            df: DataFrame with regime column
            regime_column: Regime column name

        Returns:
            DataFrame with regime statistics
        """
        stats = df.groupby(regime_column).agg({
            'pnl': ['mean', 'std', 'count'] if 'pnl' in df.columns else ['count'],
            'leverage': ['mean', 'std'] if 'leverage' in df.columns else ['count'],
        })

        return stats
