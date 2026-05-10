"""Tests for DataLoader class."""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader


@pytest.fixture
def sample_sentiment_data(tmp_path):
    """Create sample sentiment CSV file."""
    data = {
        'timestamp': ['2024-01-01 00:00:00', '2024-01-02 00:00:00', '2024-01-03 00:00:00'],
        'fear_greed_value': [25, 75, 50],
        'classification': ['Fear', 'Greed', 'Neutral']
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "sentiment.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, df


@pytest.fixture
def sample_trades_data(tmp_path):
    """Create sample trades CSV file."""
    data = {
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
        'trader_id': ['trader1', 'trader2', 'trader1'],
        'pnl': [100.5, -50.2, 75.0],
        'leverage': [5, 10, 3],
        'position_size': [1000, 2000, 1500]
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "trades.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, df


def test_dataloader_initialization(tmp_path):
    """Test DataLoader initialization."""
    loader = DataLoader(str(tmp_path))
    assert loader.data_dir == tmp_path
    assert loader.data_dir.exists()


def test_dataloader_creates_missing_directory(tmp_path):
    """Test DataLoader creates directory if it doesn't exist."""
    new_dir = tmp_path / "new_data_dir"
    loader = DataLoader(str(new_dir))
    assert new_dir.exists()


def test_load_sentiment_data_csv(sample_sentiment_data):
    """Test loading sentiment data from CSV."""
    csv_path, expected_df = sample_sentiment_data
    loader = DataLoader(str(csv_path.parent))

    df = loader.load_sentiment_data(csv_path.name)

    assert len(df) == 3
    assert 'timestamp' in df.columns
    assert 'fear_greed_value' in df.columns
    assert 'classification' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])


def test_load_sentiment_data_file_not_found(tmp_path):
    """Test loading sentiment data raises FileNotFoundError."""
    loader = DataLoader(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        loader.load_sentiment_data("nonexistent.csv")


def test_load_trades_data_csv(sample_trades_data):
    """Test loading trades data from CSV."""
    csv_path, expected_df = sample_trades_data
    loader = DataLoader(str(csv_path.parent))

    df = loader.load_trades_data(csv_path.name)

    assert len(df) == 3
    assert 'timestamp' in df.columns
    assert 'trader_id' in df.columns
    assert 'pnl' in df.columns
    assert 'leverage' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])


def test_load_trades_data_file_not_found(tmp_path):
    """Test loading trades data raises FileNotFoundError."""
    loader = DataLoader(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        loader.load_trades_data("nonexistent.csv")


def test_save_data_parquet(tmp_path):
    """Test saving data to Parquet format."""
    loader = DataLoader(str(tmp_path))
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    output_path = loader.save_data(df, "output.parquet", format="parquet")

    assert output_path.exists()
    assert output_path.suffix == ".parquet"

    # Verify data integrity
    loaded_df = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_save_data_csv(tmp_path):
    """Test saving data to CSV format."""
    loader = DataLoader(str(tmp_path))
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    output_path = loader.save_data(df, "output.csv", format="csv")

    assert output_path.exists()
    assert output_path.suffix == ".csv"

    # Verify data integrity
    loaded_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_save_data_custom_output_dir(tmp_path):
    """Test saving data to custom output directory."""
    loader = DataLoader(str(tmp_path / "raw"))
    output_dir = tmp_path / "processed"
    df = pd.DataFrame({'a': [1, 2, 3]})

    output_path = loader.save_data(df, "output.parquet", output_dir=str(output_dir))

    assert output_path.parent == output_dir
    assert output_path.exists()


def test_save_data_unsupported_format(tmp_path):
    """Test saving data with unsupported format raises ValueError."""
    loader = DataLoader(str(tmp_path))
    df = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(ValueError, match="Unsupported format"):
        loader.save_data(df, "output.txt", format="txt")


def test_list_files(tmp_path):
    """Test listing files in data directory."""
    # Create some test files
    (tmp_path / "file1.csv").touch()
    (tmp_path / "file2.csv").touch()
    (tmp_path / "file3.parquet").touch()

    loader = DataLoader(str(tmp_path))

    # List all files
    all_files = loader.list_files()
    assert len(all_files) == 3

    # List CSV files only
    csv_files = loader.list_files("*.csv")
    assert len(csv_files) == 2

    # List Parquet files only
    parquet_files = loader.list_files("*.parquet")
    assert len(parquet_files) == 1


def test_load_sentiment_data_parquet(tmp_path):
    """Test loading sentiment data from Parquet."""
    data = {
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'fear_greed_value': [25, 75],
        'classification': ['Fear', 'Greed']
    }
    df = pd.DataFrame(data)

    parquet_path = tmp_path / "sentiment.parquet"
    df.to_parquet(parquet_path, index=False)

    loader = DataLoader(str(tmp_path))
    loaded_df = loader.load_sentiment_data(parquet_path.name)

    assert len(loaded_df) == 2
    pd.testing.assert_frame_equal(df, loaded_df)


def test_load_trades_data_parquet(tmp_path):
    """Test loading trades data from Parquet."""
    data = {
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'trader_id': ['trader1', 'trader2'],
        'pnl': [100.5, -50.2],
        'leverage': [5, 10]
    }
    df = pd.DataFrame(data)

    parquet_path = tmp_path / "trades.parquet"
    df.to_parquet(parquet_path, index=False)

    loader = DataLoader(str(tmp_path))
    loaded_df = loader.load_trades_data(parquet_path.name)

    assert len(loaded_df) == 2
    pd.testing.assert_frame_equal(df, loaded_df)


def test_load_unsupported_format(tmp_path):
    """Test loading unsupported file format raises ValueError."""
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("some data")

    loader = DataLoader(str(tmp_path))

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_sentiment_data(txt_path.name)
