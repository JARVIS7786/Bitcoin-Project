"""Tests for TimestampAligner class."""

import pytest
import pandas as pd
import pytz
from datetime import datetime
from src.data.alignment import TimestampAligner


@pytest.fixture
def sample_ist_data():
    """Create sample data with IST timestamps."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 12:00:00',
            '2024-01-01 13:00:00',
            '2024-01-01 14:00:00'
        ]),
        'value': [100, 200, 300]
    })


@pytest.fixture
def sample_trades_data():
    """Create sample trades data."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 12:00:00'
        ]),
        'pnl': [100, 200, 300]
    })


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 10:30:00',
            '2024-01-01 11:30:00'
        ]),
        'fear_greed': [30, 50, 70]
    })


def test_timestamp_aligner_initialization():
    """Test TimestampAligner initialization."""
    aligner = TimestampAligner(source_timezone="Asia/Kolkata", target_timezone="UTC")
    assert aligner.source_tz == pytz.timezone("Asia/Kolkata")
    assert aligner.target_tz == pytz.timezone("UTC")


def test_timestamp_aligner_invalid_timezone():
    """Test TimestampAligner with invalid timezone raises error."""
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        TimestampAligner(source_timezone="Invalid/Timezone")


def test_convert_timezone_ist_to_utc(sample_ist_data):
    """Test converting IST timestamps to UTC."""
    aligner = TimestampAligner(source_timezone="Asia/Kolkata", target_timezone="UTC")
    converted = aligner.convert_timezone(sample_ist_data)

    # IST is UTC+5:30, so 12:00 IST = 06:30 UTC
    assert converted['timestamp'].iloc[0].hour == 6
    assert converted['timestamp'].iloc[0].minute == 30
    assert converted['timestamp'].dt.tz == pytz.UTC


def test_convert_timezone_missing_column(sample_ist_data):
    """Test convert_timezone with missing column raises KeyError."""
    aligner = TimestampAligner()
    with pytest.raises(KeyError):
        aligner.convert_timezone(sample_ist_data, timestamp_column='nonexistent')


def test_convert_timezone_non_datetime_column():
    """Test convert_timezone with non-datetime column raises ValueError."""
    df = pd.DataFrame({'timestamp': ['not', 'a', 'datetime']})
    aligner = TimestampAligner()
    with pytest.raises(ValueError, match="must be datetime type"):
        aligner.convert_timezone(df)


def test_convert_timezone_already_aware():
    """Test converting already timezone-aware timestamps."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 12:00:00']).tz_localize('US/Eastern'),
        'value': [100]
    })
    aligner = TimestampAligner(source_timezone="US/Eastern", target_timezone="UTC")
    converted = aligner.convert_timezone(df)

    assert converted['timestamp'].dt.tz == pytz.UTC


def test_align_datasets_backward(sample_trades_data, sample_sentiment_data):
    """Test aligning datasets with backward direction."""
    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        sample_trades_data,
        sample_sentiment_data,
        tolerance=pd.Timedelta('1H'),
        direction='backward'
    )

    # Each trade should match with the most recent sentiment before it
    assert len(aligned) == 3
    assert 'fear_greed' in aligned.columns
    assert aligned['fear_greed'].notna().sum() == 3  # All should match


def test_align_datasets_forward(sample_trades_data, sample_sentiment_data):
    """Test aligning datasets with forward direction."""
    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        sample_trades_data,
        sample_sentiment_data,
        tolerance=pd.Timedelta('1H'),
        direction='forward'
    )

    assert len(aligned) == 3
    assert 'fear_greed' in aligned.columns


def test_align_datasets_nearest(sample_trades_data, sample_sentiment_data):
    """Test aligning datasets with nearest direction."""
    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        sample_trades_data,
        sample_sentiment_data,
        tolerance=pd.Timedelta('1H'),
        direction='nearest'
    )

    assert len(aligned) == 3
    assert 'fear_greed' in aligned.columns


def test_align_datasets_no_tolerance(sample_trades_data, sample_sentiment_data):
    """Test aligning datasets without tolerance."""
    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        sample_trades_data,
        sample_sentiment_data,
        tolerance=None,
        direction='backward'
    )

    # Without tolerance, all matches should succeed
    assert len(aligned) == 3


def test_align_datasets_strict_tolerance():
    """Test aligning datasets with strict tolerance."""
    trades = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
        'pnl': [100]
    })
    sentiment = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 08:00:00']),  # 2 hours before
        'fear_greed': [30]
    })

    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        trades,
        sentiment,
        tolerance=pd.Timedelta('1H'),  # Only 1 hour tolerance
        direction='backward'
    )

    # Should not match due to strict tolerance
    assert aligned['fear_greed'].isna().sum() == 1


def test_align_datasets_missing_column(sample_trades_data, sample_sentiment_data):
    """Test align_datasets with missing column raises KeyError."""
    aligner = TimestampAligner()
    with pytest.raises(KeyError):
        aligner.align_datasets(
            sample_trades_data,
            sample_sentiment_data,
            left_timestamp='nonexistent'
        )


def test_validate_temporal_order_sorted():
    """Test validating sorted timestamps."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
    })
    aligner = TimestampAligner()
    assert aligner.validate_temporal_order(df) is True


def test_validate_temporal_order_unsorted():
    """Test validating unsorted timestamps."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-02'])
    })
    aligner = TimestampAligner()
    assert aligner.validate_temporal_order(df) is False


def test_validate_temporal_order_missing_column():
    """Test validate_temporal_order with missing column raises KeyError."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    aligner = TimestampAligner()
    with pytest.raises(KeyError):
        aligner.validate_temporal_order(df)


def test_get_time_range():
    """Test getting time range of DataFrame."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-31'])
    })
    aligner = TimestampAligner()
    min_ts, max_ts = aligner.get_time_range(df)

    assert min_ts == pd.Timestamp('2024-01-01')
    assert max_ts == pd.Timestamp('2024-01-31')


def test_get_time_range_missing_column():
    """Test get_time_range with missing column raises KeyError."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    aligner = TimestampAligner()
    with pytest.raises(KeyError):
        aligner.get_time_range(df)


def test_remove_timezone_info():
    """Test removing timezone information."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01']).tz_localize('UTC')
    })
    aligner = TimestampAligner()
    naive_df = aligner.remove_timezone_info(df)

    assert naive_df['timestamp'].dt.tz is None


def test_remove_timezone_info_already_naive():
    """Test removing timezone info from already naive timestamps."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01'])
    })
    aligner = TimestampAligner()
    naive_df = aligner.remove_timezone_info(df)

    assert naive_df['timestamp'].dt.tz is None


def test_align_datasets_preserves_left_data(sample_trades_data, sample_sentiment_data):
    """Test that align_datasets preserves all left DataFrame data."""
    aligner = TimestampAligner()
    aligned = aligner.align_datasets(
        sample_trades_data,
        sample_sentiment_data,
        tolerance=pd.Timedelta('1H')
    )

    # All original columns from left should be present
    assert 'pnl' in aligned.columns
    assert len(aligned) == len(sample_trades_data)
    # Original pnl values should be preserved
    assert aligned['pnl'].tolist() == sample_trades_data['pnl'].tolist()
