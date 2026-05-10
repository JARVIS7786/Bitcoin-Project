"""Timestamp alignment utilities for PrimeTRADE."""

import pandas as pd
import pytz
from typing import Optional
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimestampAligner:
    """
    Align timestamps across datasets with timezone conversion.

    Handles IST→UTC conversion and asof merging for time series data.
    """

    def __init__(
        self,
        source_timezone: str = "Asia/Kolkata",
        target_timezone: str = "UTC",
    ):
        """
        Initialize TimestampAligner.

        Args:
            source_timezone: Source timezone (default: Asia/Kolkata for IST)
            target_timezone: Target timezone (default: UTC)

        Raises:
            pytz.exceptions.UnknownTimeZoneError: If timezone is invalid
        """
        self.source_tz = pytz.timezone(source_timezone)
        self.target_tz = pytz.timezone(target_timezone)
        logger.info(f"Initialized TimestampAligner: {source_timezone} → {target_timezone}")

    def convert_timezone(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Convert timestamps from source to target timezone.

        Args:
            df: Input DataFrame with timestamp column
            timestamp_column: Name of timestamp column

        Returns:
            DataFrame with converted timestamps

        Raises:
            KeyError: If timestamp column not found
            ValueError: If timestamp column is not datetime type

        Examples:
            >>> aligner = TimestampAligner()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01 12:00:00']),
            ...     'value': [100]
            ... })
            >>> converted = aligner.convert_timezone(df)
            >>> # IST 12:00 becomes UTC 06:30
        """
        if timestamp_column not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            raise ValueError(f"Column '{timestamp_column}' must be datetime type")

        df = df.copy()

        # If timestamps are naive (no timezone), localize to source timezone
        if df[timestamp_column].dt.tz is None:
            logger.info(f"Localizing naive timestamps to {self.source_tz}")
            df[timestamp_column] = df[timestamp_column].dt.tz_localize(self.source_tz)
        else:
            # If already timezone-aware, convert to source timezone first
            df[timestamp_column] = df[timestamp_column].dt.tz_convert(self.source_tz)

        # Convert to target timezone
        df[timestamp_column] = df[timestamp_column].dt.tz_convert(self.target_tz)
        logger.info(f"Converted {len(df)} timestamps to {self.target_tz}")

        return df

    def align_datasets(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_timestamp: str = "timestamp",
        right_timestamp: str = "timestamp",
        tolerance: Optional[pd.Timedelta] = None,
        direction: str = "backward",
    ) -> pd.DataFrame:
        """
        Align two datasets using asof merge.

        Performs time-based merge where each row in left_df is matched with
        the closest timestamp in right_df within tolerance.

        Args:
            left_df: Left DataFrame (typically trades data)
            right_df: Right DataFrame (typically sentiment data)
            left_timestamp: Timestamp column in left_df
            right_timestamp: Timestamp column in right_df
            tolerance: Maximum time difference for matching (e.g., pd.Timedelta('1H'))
            direction: Direction for asof merge ('backward', 'forward', 'nearest')

        Returns:
            Merged DataFrame with aligned timestamps

        Raises:
            ValueError: If DataFrames are not sorted by timestamp

        Examples:
            >>> aligner = TimestampAligner()
            >>> trades = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 11:00']),
            ...     'pnl': [100, 200]
            ... })
            >>> sentiment = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01 09:30', '2024-01-01 10:30']),
            ...     'fear_greed': [30, 70]
            ... })
            >>> aligned = aligner.align_datasets(trades, sentiment, tolerance=pd.Timedelta('1H'))
        """
        # Validate timestamp columns exist
        if left_timestamp not in left_df.columns:
            raise KeyError(f"Timestamp column '{left_timestamp}' not found in left DataFrame")
        if right_timestamp not in right_df.columns:
            raise KeyError(f"Timestamp column '{right_timestamp}' not found in right DataFrame")

        # Ensure both DataFrames are sorted by timestamp
        left_df = left_df.sort_values(left_timestamp).reset_index(drop=True)
        right_df = right_df.sort_values(right_timestamp).reset_index(drop=True)

        logger.info(f"Aligning {len(left_df)} left rows with {len(right_df)} right rows")
        logger.info(f"Tolerance: {tolerance}, Direction: {direction}")

        # Perform asof merge
        merged_df = pd.merge_asof(
            left_df,
            right_df,
            left_on=left_timestamp,
            right_on=right_timestamp,
            tolerance=tolerance,
            direction=direction,
        )

        # Count successful matches
        matches = merged_df[right_df.columns[1]].notna().sum()  # Check first non-timestamp column from right
        logger.info(f"Successfully matched {matches}/{len(merged_df)} rows")

        return merged_df

    def validate_temporal_order(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> bool:
        """
        Validate that timestamps are in ascending order.

        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column

        Returns:
            True if timestamps are sorted, False otherwise

        Examples:
            >>> aligner = TimestampAligner()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
            ... })
            >>> aligner.validate_temporal_order(df)
            True
        """
        if timestamp_column not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_column}' not found")

        is_sorted = df[timestamp_column].is_monotonic_increasing

        if is_sorted:
            logger.info("Timestamps are in ascending order")
        else:
            logger.warning("Timestamps are NOT in ascending order")

        return is_sorted

    def get_time_range(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> tuple[datetime, datetime]:
        """
        Get the time range of a DataFrame.

        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column

        Returns:
            Tuple of (min_timestamp, max_timestamp)

        Examples:
            >>> aligner = TimestampAligner()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01', '2024-01-03'])
            ... })
            >>> start, end = aligner.get_time_range(df)
        """
        if timestamp_column not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_column}' not found")

        min_ts = df[timestamp_column].min()
        max_ts = df[timestamp_column].max()

        logger.info(f"Time range: {min_ts} to {max_ts}")

        return min_ts, max_ts

    def remove_timezone_info(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Remove timezone information from timestamps (convert to naive).

        Useful for saving to formats that don't support timezones.

        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column

        Returns:
            DataFrame with naive timestamps

        Examples:
            >>> aligner = TimestampAligner()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.to_datetime(['2024-01-01']).tz_localize('UTC')
            ... })
            >>> naive_df = aligner.remove_timezone_info(df)
            >>> naive_df['timestamp'].dt.tz is None
            True
        """
        df = df.copy()

        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            if df[timestamp_column].dt.tz is not None:
                df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)
                logger.info("Removed timezone information from timestamps")

        return df
