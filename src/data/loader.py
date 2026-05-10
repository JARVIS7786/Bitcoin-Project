"""Data loading utilities for PrimeTRADE."""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Load sentiment and trading data from CSV/Parquet files.

    Handles file I/O, basic validation, and type conversion for raw data.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_sentiment_data(
        self,
        filename: str,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load Bitcoin sentiment data (Fear & Greed Index).

        Args:
            filename: Name of sentiment data file
            parse_dates: Columns to parse as datetime (default: ['timestamp'])

        Returns:
            DataFrame with sentiment data

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            pd.errors.ParserError: If file format is invalid

        Examples:
            >>> loader = DataLoader()
            >>> df = loader.load_sentiment_data("fear_greed_index.csv")
            >>> print(df.columns)
            Index(['timestamp', 'fear_greed_value', 'classification'], dtype='object')
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading sentiment data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Sentiment data file not found: {file_path}")

        if parse_dates is None:
            parse_dates = ['timestamp']

        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, parse_dates=parse_dates)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(df)} rows of sentiment data")
        return df

    def load_trades_data(
        self,
        filename: str,
        parse_dates: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load Hyperliquid trading data.

        Args:
            filename: Name of trades data file
            parse_dates: Columns to parse as datetime (default: ['timestamp'])
            chunksize: If specified, return iterator for large files

        Returns:
            DataFrame with trading data (or iterator if chunksize specified)

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            pd.errors.ParserError: If file format is invalid

        Examples:
            >>> loader = DataLoader()
            >>> df = loader.load_trades_data("historical_data.csv")
            >>> print(df.columns)
            Index(['timestamp', 'trader_id', 'pnl', 'leverage', ...], dtype='object')
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading trades data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Trades data file not found: {file_path}")

        if parse_dates is None:
            parse_dates = ['timestamp']

        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(
                file_path,
                parse_dates=parse_dates,
                chunksize=chunksize,
            )
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if chunksize is None:
            logger.info(f"Loaded {len(df)} rows of trades data")
        else:
            logger.info(f"Loaded trades data in chunks of {chunksize}")

        return df

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None,
        format: str = "parquet",
    ) -> Path:
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory (default: data_dir)
            format: Output format ('parquet' or 'csv')

        Returns:
            Path to saved file

        Examples:
            >>> loader = DataLoader()
            >>> df = pd.DataFrame({'a': [1, 2, 3]})
            >>> path = loader.save_data(df, "output.parquet")
        """
        if output_dir is None:
            output_dir = self.data_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        logger.info(f"Saving data to {file_path}")

        if format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(df)} rows to {file_path}")
        return file_path

    def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List files in data directory matching pattern.

        Args:
            pattern: Glob pattern (default: all files)

        Returns:
            List of matching file paths

        Examples:
            >>> loader = DataLoader()
            >>> csv_files = loader.list_files("*.csv")
        """
        return list(self.data_dir.glob(pattern))
