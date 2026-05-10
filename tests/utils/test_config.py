"""Tests for Config class."""

import pytest
import yaml
from pathlib import Path
from src.utils.config import Config


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample config file for testing."""
    config_data = {
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
        },
        "model": {
            "type": "xgboost",
            "params": {
                "max_depth": 6,
                "learning_rate": 0.1,
            }
        },
        "logging": {
            "level": "INFO"
        }
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return config_file


def test_config_initialization(sample_config_file):
    """Test Config initialization from YAML file."""
    config = Config(str(sample_config_file))
    assert config.config_path == sample_config_file
    assert isinstance(config._config, dict)


def test_config_file_not_found():
    """Test Config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        Config("nonexistent_config.yaml")


def test_get_simple_key(sample_config_file):
    """Test getting simple top-level key."""
    config = Config(str(sample_config_file))
    assert config.get("logging") == {"level": "INFO"}


def test_get_nested_key(sample_config_file):
    """Test getting nested key with dot notation."""
    config = Config(str(sample_config_file))
    assert config.get("data.raw_path") == "data/raw"
    assert config.get("model.params.max_depth") == 6


def test_get_with_default(sample_config_file):
    """Test get returns default for missing key."""
    config = Config(str(sample_config_file))
    assert config.get("missing.key", "default") == "default"
    assert config.get("data.missing", None) is None


def test_get_required_existing_key(sample_config_file):
    """Test get_required returns value for existing key."""
    config = Config(str(sample_config_file))
    assert config.get_required("data.raw_path") == "data/raw"


def test_get_required_missing_key(sample_config_file):
    """Test get_required raises KeyError for missing key."""
    config = Config(str(sample_config_file))
    with pytest.raises(KeyError, match="Required config key not found"):
        config.get_required("missing.key")


def test_set_simple_key(sample_config_file):
    """Test setting simple key."""
    config = Config(str(sample_config_file))
    config.set("new_key", "new_value")
    assert config.get("new_key") == "new_value"


def test_set_nested_key(sample_config_file):
    """Test setting nested key with dot notation."""
    config = Config(str(sample_config_file))
    config.set("data.new_path", "data/new")
    assert config.get("data.new_path") == "data/new"


def test_set_creates_nested_structure(sample_config_file):
    """Test set creates nested structure if it doesn't exist."""
    config = Config(str(sample_config_file))
    config.set("new.nested.key", "value")
    assert config.get("new.nested.key") == "value"


def test_to_dict(sample_config_file):
    """Test to_dict returns complete config."""
    config = Config(str(sample_config_file))
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "data" in config_dict
    assert "model" in config_dict


def test_save_to_original_path(sample_config_file, tmp_path):
    """Test saving config to original path."""
    config = Config(str(sample_config_file))
    config.set("new_key", "new_value")
    config.save()

    # Reload and verify
    reloaded_config = Config(str(sample_config_file))
    assert reloaded_config.get("new_key") == "new_value"


def test_save_to_new_path(sample_config_file, tmp_path):
    """Test saving config to new path."""
    config = Config(str(sample_config_file))
    config.set("new_key", "new_value")

    new_path = tmp_path / "new_config.yaml"
    config.save(str(new_path))

    # Verify new file exists and contains changes
    assert new_path.exists()
    new_config = Config(str(new_path))
    assert new_config.get("new_key") == "new_value"


def test_repr(sample_config_file):
    """Test string representation."""
    config = Config(str(sample_config_file))
    repr_str = repr(config)
    assert "Config" in repr_str
    assert str(sample_config_file) in repr_str
