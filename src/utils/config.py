"""Configuration management for PrimeTRADE."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Centralized configuration manager that loads YAML configs.

    Provides type-safe access to configuration values with support for
    nested keys using dot notation (e.g., 'data.raw_path').
    """

    def __init__(self, config_path: str):
        """
        Initialize Config from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys using dot notation (e.g., 'data.raw_path').

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config('config.yaml')
            >>> config.get('data.raw_path')
            'data/raw'
            >>> config.get('missing.key', 'default_value')
            'default_value'
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_required(self, key: str) -> Any:
        """
        Get required configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required config key not found: {key}")
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            output_path: Output path (defaults to original config_path)
        """
        path = Path(output_path) if output_path else self.config_path
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
