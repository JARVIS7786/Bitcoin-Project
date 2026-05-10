"""End-to-end pipelines for PrimeTRADE."""

import pandas as pd
from typing import Optional
from src.data import DataLoader, DataCleaner, TimestampAligner
from src.features import create_sentiment_features, create_trader_features, create_risk_features, create_temporal_features
from src.models import BaselineModel, XGBoostModel
from src.backtesting import StrategySimulator
from src.utils import Config, setup_logger

logger = setup_logger(__name__)


class DataPipeline:
    """
    End-to-end data processing pipeline.
    """

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize DataPipeline.

        Args:
            config_path: Path to data configuration
        """
        self.config = Config(config_path)
        self.loader = DataLoader(self.config.get('data.raw_path'))
        self.cleaner = DataCleaner()
        self.aligner = TimestampAligner()

    def run(self) -> pd.DataFrame:
        """
        Run data pipeline.

        Returns:
            Cleaned and aligned DataFrame
        """
        logger.info("Starting data pipeline")

        # Load data
        sentiment_df = self.loader.load_sentiment_data(self.config.get('data.sentiment_file'))
        trades_df = self.loader.load_trades_data(self.config.get('data.trades_file'))

        # Clean data
        sentiment_df = self.cleaner.clean(sentiment_df, numeric_columns=['fear_greed_value'])
        trades_df = self.cleaner.clean(trades_df, numeric_columns=['pnl', 'leverage'])

        # Align timestamps
        sentiment_df = self.aligner.convert_timezone(sentiment_df)
        trades_df = self.aligner.convert_timezone(trades_df)

        # Merge datasets
        aligned_df = self.aligner.align_datasets(trades_df, sentiment_df, tolerance=pd.Timedelta('1H'))

        logger.info(f"Data pipeline complete. Output shape: {aligned_df.shape}")

        return aligned_df


class FeaturePipeline:
    """
    Feature engineering pipeline.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FeaturePipeline.

        Args:
            config_path: Path to feature configuration
        """
        self.config = Config(config_path) if config_path else None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run feature engineering pipeline.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with features
        """
        logger.info("Starting feature pipeline")

        # Create features
        df = create_sentiment_features(df)
        df = create_trader_features(df)
        df = create_risk_features(df)
        df = create_temporal_features(df)

        # Drop rows with NaN (from rolling windows)
        df = df.dropna()

        logger.info(f"Feature pipeline complete. Output shape: {df.shape}")

        return df


class TrainingPipeline:
    """
    Model training pipeline.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TrainingPipeline.

        Args:
            config_path: Path to model configuration
        """
        self.config = Config(config_path) if config_path else None

    def run(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_profitable_intraday',
        test_size: float = 0.2,
    ):
        """
        Run training pipeline.

        Args:
            df: DataFrame with features and target
            target_column: Target column name
            test_size: Test set size ratio

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Starting training pipeline")

        # Create target if not exists
        if target_column not in df.columns:
            df[target_column] = (df['pnl'] > 0).astype(int)

        # Split data temporally
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Prepare features
        feature_cols = [c for c in df.columns if c not in ['pnl', 'trader_id', 'timestamp', target_column]]
        X_train = train_df[feature_cols]
        y_train = train_df[target_column]
        X_test = test_df[feature_cols]
        y_test = test_df[target_column]

        # Train model
        model = XGBoostModel()
        model.train(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        logger.info(f"Training pipeline complete. Metrics: {metrics}")

        return model, metrics
