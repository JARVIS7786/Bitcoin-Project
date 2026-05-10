# Bitcoin Sentiment & Trader Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an institutional-grade quantitative research system analyzing how Bitcoin market sentiment influences trader profitability on Hyperliquid through behavioral finance lens.

**Architecture:** Modular Python system with clear separation between data processing, feature engineering, signal validation, regime detection, research analysis, modeling, and backtesting. Emphasizes temporal validation, causal inference, and behavioral metrics over model complexity.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, MLflow, pytest, matplotlib, seaborn, plotly

---

## Phase 1: Foundation & Data Infrastructure

### Task 1: Project Structure Setup

**Files:**
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/__init__.py`

- [ ] **Step 1: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="primetrade",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "mlflow>=2.8.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "statsmodels>=0.14.0",
        "imbalanced-learn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ]
    },
)
```

- [ ] **Step 2: Create requirements.txt**

```txt
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
mlflow>=2.8.0
pyarrow>=14.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
pydantic>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
statsmodels>=0.14.0
imbalanced-learn>=0.11.0

# Dev dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.1.0
mypy>=1.5.0
```

- [ ] **Step 3: Create .gitignore**

```txt
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/processed/
data/features/
*.parquet
*.csv

# Experiments
experiments/mlruns/
mlruns/

# Outputs
outputs/

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Testing
.coverage
htmlcov/
.pytest_cache/

# MyPy
.mypy_cache/
```

- [ ] **Step 4: Create basic README.md**

```markdown
# PrimeTRADE: Bitcoin Sentiment & Trader Performance Prediction

Institutional-grade quantitative research system analyzing how Bitcoin market sentiment (Fear & Greed Index) influences trader profitability on Hyperliquid through behavioral finance lens.

## Core Thesis

Extreme market sentiment induces predictable behavioral biases in trader risk-taking that persist across multiple time horizons, with profitability inversely correlated to emotional leverage expansion during regime transitions.

## Key Features

- **Behavioral Finance Focus**: Quantifies emotional trading patterns (revenge trading, overconfidence, loss aversion)
- **Multi-Horizon Prediction**: Intraday, next-day, and 7-day profitability forecasts
- **Regime-Aware Modeling**: Adapts to fear/greed/neutral market regimes
- **Temporal Validation**: Walk-forward validation prevents future leakage
- **Signal Decay Analysis**: Measures how predictive power degrades over time
- **Causal Inference**: Granger causality tests, not just correlation
- **Realistic Backtesting**: Transaction costs, slippage, market impact

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
PrimeTRADE/
├── src/                    # Source code
│   ├── data/              # Data loading & preprocessing
│   ├── features/          # Feature engineering
│   ├── signals/           # Validated predictive signals
│   ├── regimes/           # Market regime detection
│   ├── research/          # Hypothesis testing & analysis
│   ├── validation/        # Leakage checks & robustness tests
│   ├── models/            # ML models
│   ├── backtesting/       # Strategy simulation
│   └── pipelines/         # End-to-end workflows
├── notebooks/             # Research notebooks
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── docs/                  # Documentation
└── data/                  # Data storage
```

## Quick Start

```python
# Load and process data
from src.pipelines.data_pipeline import DataPipeline

pipeline = DataPipeline()
data = pipeline.run()

# Engineer features
from src.pipelines.feature_pipeline import FeaturePipeline

features = FeaturePipeline().run(data)

# Train model
from src.pipelines.training_pipeline import TrainingPipeline

model = TrainingPipeline().run(features)
```

## Research Methodology

See `docs/METHODOLOGY.md` for detailed research approach.

## License

MIT
```

- [ ] **Step 5: Create src/__init__.py**

```python
"""PrimeTRADE: Bitcoin Sentiment & Trader Performance Prediction System"""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create directory structure**

Run:
```bash
mkdir -p src/data src/features src/signals src/regimes src/research src/validation src/models src/backtesting src/pipelines src/analysis src/visualization src/utils
mkdir -p tests/data tests/features tests/models tests/validation
mkdir -p notebooks configs experiments/mlruns outputs/figures outputs/tables outputs/reports
mkdir -p data/raw data/processed data/features
mkdir -p docs/superpowers/specs docs/superpowers/plans
```

- [ ] **Step 7: Commit foundation**

```bash
git add .
git commit -m "feat: initialize project structure and dependencies

- Add setup.py with all required dependencies
- Add requirements.txt for easy installation
- Add comprehensive .gitignore
- Add README with project overview
- Create modular directory structure
- Initialize src package

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

### Task 2: Configuration Management

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/config.py`
- Create: `src/utils/logger.py`
- Create: `configs/data_config.yaml`
- Create: `tests/utils/test_config.py`

- [ ] **Step 1: Write test for config loading**

```python
# tests/utils/test_config.py
import pytest
from pathlib import Path
from src.utils.config import Config


def test_config_loads_yaml():
    """Test that Config can load YAML files"""
    config = Config("configs/data_config.yaml")
    assert config.get("raw_data_path") is not None


def test_config_get_with_default():
    """Test Config.get() with default value"""
    config = Config("configs/data_config.yaml")
    value = config.get("nonexistent_key", default="default_value")
    assert value == "default_value"


def test_config_get_nested():
    """Test Config.get() with nested keys"""
    config = Config("configs/data_config.yaml")
    value = config.get("processing.remove_duplicates")
    assert isinstance(value, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.utils.config'"

- [ ] **Step 3: Implement Config class**

```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration management using YAML files"""
    
    def __init__(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., "processing.remove_duplicates")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key not found: {key}")
        return value
```

- [ ] **Step 4: Create data_config.yaml**

```yaml
# configs/data_config.yaml
raw_data_path: "data/raw"
processed_data_path: "data/processed"
features_data_path: "data/features"

sentiment_file: "fear_greed_index.csv"
trades_file: "historical_data.csv"

processing:
  remove_duplicates: true
  handle_missing: "drop"  # Options: drop, fill, interpolate
  outlier_detection: true
  outlier_threshold: 3.0  # Standard deviations

timestamp:
  timezone: "UTC"
  sentiment_column: "timestamp"
  trades_column: "Timestamp"
  
validation:
  check_future_leakage: true
  check_temporal_ordering: true
  min_trades_per_day: 10
```

- [ ] **Step 5: Implement logger utility**

```python
# src/utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and optional file output
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger
```

- [ ] **Step 6: Create utils __init__.py**

```python
# src/utils/__init__.py
from .config import Config
from .logger import setup_logger

__all__ = ['Config', 'setup_logger']
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/utils/test_config.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 8: Commit configuration utilities**

```bash
git add src/utils/ configs/data_config.yaml tests/utils/
git commit -m "feat: add configuration management and logging utilities

- Implement Config class with YAML loading and nested key support
- Add logger utility with console and file output
- Create data_config.yaml with processing parameters
- Add unit tests for Config class

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

### Task 3: Data Loading Module

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/loader.py`
- Create: `tests/data/test_loader.py`

- [ ] **Step 1: Write test for data loader**

```python
# tests/data/test_loader.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader


@pytest.fixture
def data_loader():
    """Create DataLoader instance"""
    return DataLoader("configs/data_config.yaml")


def test_load_sentiment_data(data_loader):
    """Test loading sentiment data"""
    df = data_loader.load_sentiment_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'timestamp' in df.columns
    assert 'value' in df.columns
    assert 'classification' in df.columns
    assert 'date' in df.columns
    assert len(df) > 0


def test_load_trades_data(data_loader):
    """Test loading trades data"""
    df = data_loader.load_trades_data()
    
    assert isinstance(df, pd.DataFrame)
    assert 'Account' in df.columns
    assert 'Closed PnL' in df.columns
    assert 'Timestamp' in df.columns
    assert len(df) > 0


def test_sentiment_data_types(data_loader):
    """Test sentiment data has correct types"""
    df = data_loader.load_sentiment_data()
    
    assert pd.api.types.is_integer_dtype(df['timestamp'])
    assert pd.api.types.is_integer_dtype(df['value'])
    assert pd.api.types.is_string_dtype(df['classification'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_loader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.data.loader'"

- [ ] **Step 3: Implement DataLoader class**

```python
# src/data/loader.py
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import Config
from src.utils.logger import setup_logger


class DataLoader:
    """Load raw data from CSV files"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize DataLoader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = setup_logger(__name__)
        
        self.raw_data_path = Path(self.config.get("raw_data_path"))
        self.sentiment_file = self.config.get("sentiment_file")
        self.trades_file = self.config.get("trades_file")
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """
        Load Fear & Greed Index data
        
        Returns:
            DataFrame with columns: timestamp, value, classification, date
        """
        file_path = self.raw_data_path / self.sentiment_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Sentiment data not found: {file_path}")
        
        self.logger.info(f"Loading sentiment data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'value', 'classification', 'date']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} sentiment records")
        
        return df
    
    def load_trades_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load Hyperliquid historical trades data
        
        Args:
            nrows: Optional number of rows to load (for testing)
            
        Returns:
            DataFrame with trade execution data
        """
        file_path = self.raw_data_path / self.trades_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Trades data not found: {file_path}")
        
        self.logger.info(f"Loading trades data from {file_path}")
        
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Validate required columns
        required_cols = ['Account', 'Timestamp', 'Closed PnL']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} trade records")
        
        return df
```

- [ ] **Step 4: Create data __init__.py**

```python
# src/data/__init__.py
from .loader import DataLoader

__all__ = ['DataLoader']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_loader.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit data loader**

```bash
git add src/data/ tests/data/
git commit -m "feat: implement data loading module

- Add DataLoader class for sentiment and trades data
- Validate required columns on load
- Add logging for data loading operations
- Add unit tests for data loading

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

## Phase 2: Data Cleaning & Validation

### Task 4: Data Cleaning Module

**Files:**
- Create: `src/data/cleaner.py`
- Create: `tests/data/test_cleaner.py`

- [ ] **Step 1: Write test for data cleaner**

```python
# tests/data/test_cleaner.py
import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data with issues"""
    return pd.DataFrame({
        'timestamp': [1, 2, 2, 3, 4],  # Duplicate at index 2
        'value': [50, 60, 60, np.nan, 70],  # Missing value
        'classification': ['Neutral', 'Greed', 'Greed', 'Fear', 'Greed'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-04']
    })


@pytest.fixture
def sample_trades_data():
    """Create sample trades data with issues"""
    return pd.DataFrame({
        'Account': ['0xabc', '0xabc', '0xdef', '0xdef'],
        'Timestamp': ['01-01-2024 10:00', '01-01-2024 11:00', '01-01-2024 12:00', '01-01-2024 13:00'],
        'Closed PnL': [100, -50, 200, 1000000],  # Outlier at index 3
        'Size USD': [1000, 2000, 1500, 3000]
    })


def test_remove_duplicates(sample_sentiment_data):
    """Test duplicate removal"""
    cleaner = DataCleaner()
    cleaned = cleaner.remove_duplicates(sample_sentiment_data, subset=['timestamp'])
    
    assert len(cleaned) == 4  # One duplicate removed
    assert cleaned['timestamp'].is_unique


def test_handle_missing_values_drop(sample_sentiment_data):
    """Test missing value handling with drop strategy"""
    cleaner = DataCleaner()
    cleaned = cleaner.handle_missing_values(sample_sentiment_data, strategy='drop')
    
    assert len(cleaned) == 4  # One row with NaN removed
    assert not cleaned['value'].isna().any()


def test_detect_outliers(sample_trades_data):
    """Test outlier detection"""
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers(sample_trades_data, column='Closed PnL', threshold=3.0)
    
    assert len(outliers) == 1  # One outlier detected
    assert outliers.iloc[0]['Closed PnL'] == 1000000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_cleaner.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.data.cleaner'"

- [ ] **Step 3: Implement DataCleaner class**

```python
# src/data/cleaner.py
import pandas as pd
import numpy as np
from typing import Optional, List
from src.utils.logger import setup_logger


class DataCleaner:
    """Clean and validate data"""
    
    def __init__(self):
        """Initialize DataCleaner"""
        self.logger = setup_logger(__name__)
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for identifying duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        removed = initial_len - len(df_clean)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate rows")
        
        return df_clean
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'drop',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill', or 'interpolate'
            fill_value: Value to use for 'fill' strategy
            
        Returns:
            DataFrame with missing values handled
        """
        initial_missing = df.isna().sum().sum()
        
        if strategy == 'drop':
            df_clean = df.dropna()
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value required for 'fill' strategy")
            df_clean = df.fillna(fill_value)
        elif strategy == 'interpolate':
            df_clean = df.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        final_missing = df_clean.isna().sum().sum()
        handled = initial_missing - final_missing
        
        if handled > 0:
            self.logger.info(f"Handled {handled} missing values using '{strategy}' strategy")
        
        return df_clean
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using z-score method
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            threshold: Z-score threshold (default: 3.0 standard deviations)
            
        Returns:
            DataFrame containing only outlier rows
        """
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
        
        # Calculate z-scores
        mean = df[column].mean()
        std = df[column].std()
        
        if std == 0:
            self.logger.warning(f"Column '{column}' has zero standard deviation")
            return pd.DataFrame()
        
        z_scores = np.abs((df[column] - mean) / std)
        outliers = df[z_scores > threshold]
        
        if len(outliers) > 0:
            self.logger.info(f"Detected {len(outliers)} outliers in column '{column}'")
        
        return outliers
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        outliers = self.detect_outliers(df, column, threshold)
        df_clean = df[~df.index.isin(outliers.index)]
        
        removed = len(df) - len(df_clean)
        if removed > 0:
            self.logger.info(f"Removed {removed} outlier rows from column '{column}'")
        
        return df_clean
```

- [ ] **Step 4: Update data __init__.py**

```python
# src/data/__init__.py
from .loader import DataLoader
from .cleaner import DataCleaner

__all__ = ['DataLoader', 'DataCleaner']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_cleaner.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit data cleaner**

```bash
git add src/data/cleaner.py tests/data/test_cleaner.py
git commit -m "feat: implement data cleaning module

- Add DataCleaner class with duplicate removal
- Implement missing value handling (drop, fill, interpolate)
- Add outlier detection using z-score method
- Add unit tests for cleaning operations

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

### Task 5: Timestamp Alignment Module

**Files:**
- Create: `src/data/alignment.py`
- Create: `tests/data/test_alignment.py`

- [ ] **Step 1: Write test for timestamp alignment**

```python
# tests/data/test_alignment.py
import pytest
import pandas as pd
from datetime import datetime
from src.data.alignment import TimestampAligner


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data"""
    return pd.DataFrame({
        'timestamp': [1733184000, 1733270400, 1733356800],  # Unix timestamps
        'value': [50, 60, 70],
        'classification': ['Neutral', 'Greed', 'Greed'],
        'date': ['2024-12-03', '2024-12-04', '2024-12-05']
    })


@pytest.fixture
def sample_trades_data():
    """Create sample trades data"""
    return pd.DataFrame({
        'Account': ['0xabc', '0xabc', '0xdef'],
        'Timestamp IST': ['03-12-2024 10:00', '04-12-2024 11:00', '05-12-2024 12:00'],
        'Closed PnL': [100, -50, 200]
    })


def test_convert_to_datetime():
    """Test timestamp conversion to datetime"""
    aligner = TimestampAligner()
    
    # Unix timestamp
    dt = aligner.convert_to_datetime(1733184000, format='unix')
    assert isinstance(dt, pd.Timestamp)
    
    # String timestamp
    dt = aligner.convert_to_datetime('03-12-2024 10:00', format='%d-%m-%Y %H:%M')
    assert isinstance(dt, pd.Timestamp)


def test_align_to_daily(sample_sentiment_data, sample_trades_data):
    """Test aligning trades to daily sentiment"""
    aligner = TimestampAligner()
    
    # Convert timestamps
    sentiment_df = sample_sentiment_data.copy()
    sentiment_df['datetime'] = pd.to_datetime(sentiment_df['date'])
    
    trades_df = sample_trades_data.copy()
    trades_df['datetime'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    
    # Align
    aligned = aligner.align_to_daily(trades_df, sentiment_df, on='datetime')
    
    assert 'value' in aligned.columns  # Sentiment value merged
    assert 'classification' in aligned.columns
    assert len(aligned) == len(trades_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_alignment.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.data.alignment'"

- [ ] **Step 3: Implement TimestampAligner class**

```python
# src/data/alignment.py
import pandas as pd
from datetime import datetime
from typing import Union, Optional
from src.utils.logger import setup_logger


class TimestampAligner:
    """Align timestamps across different data sources"""
    
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize TimestampAligner
        
        Args:
            timezone: Target timezone for alignment
        """
        self.timezone = timezone
        self.logger = setup_logger(__name__)
    
    def convert_to_datetime(
        self,
        timestamp: Union[int, str],
        format: str = 'unix'
    ) -> pd.Timestamp:
        """
        Convert timestamp to datetime
        
        Args:
            timestamp: Unix timestamp or string
            format: 'unix' or strftime format string
            
        Returns:
            Pandas Timestamp in UTC
        """
        if format == 'unix':
            dt = pd.to_datetime(timestamp, unit='s', utc=True)
        else:
            dt = pd.to_datetime(timestamp, format=format)
            if dt.tz is None:
                dt = dt.tz_localize(self.timezone)
            else:
                dt = dt.tz_convert(self.timezone)
        
        return dt
    
    def align_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align sentiment data timestamps
        
        Args:
            df: Sentiment DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with 'datetime' column in UTC
        """
        df = df.copy()
        
        # Convert Unix timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
        # Extract date for daily aggregation
        df['date_only'] = df['datetime'].dt.date
        
        self.logger.info(f"Aligned {len(df)} sentiment records to UTC")
        
        return df
    
    def align_trades_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'Timestamp IST',
        format: str = '%d-%m-%Y %H:%M'
    ) -> pd.DataFrame:
        """
        Align trades data timestamps
        
        Args:
            df: Trades DataFrame
            timestamp_col: Name of timestamp column
            format: Timestamp format string
            
        Returns:
            DataFrame with 'datetime' column in UTC
        """
        df = df.copy()
        
        # Convert string timestamp to datetime
        df['datetime'] = pd.to_datetime(df[timestamp_col], format=format)
        
        # Convert IST to UTC (IST is UTC+5:30)
        df['datetime'] = df['datetime'] - pd.Timedelta(hours=5, minutes=30)
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        
        # Extract date for daily aggregation
        df['date_only'] = df['datetime'].dt.date
        
        self.logger.info(f"Aligned {len(df)} trade records to UTC")
        
        return df
    
    def align_to_daily(
        self,
        trades_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        on: str = 'date_only'
    ) -> pd.DataFrame:
        """
        Merge trades with daily sentiment
        
        Args:
            trades_df: Trades DataFrame with datetime
            sentiment_df: Sentiment DataFrame with datetime
            on: Column to merge on
            
        Returns:
            Merged DataFrame
        """
        # Merge on date
        merged = trades_df.merge(
            sentiment_df[['date_only', 'value', 'classification']],
            on=on,
            how='left'
        )
        
        self.logger.info(f"Merged {len(merged)} records with daily sentiment")
        
        return merged
```

- [ ] **Step 4: Update data __init__.py**

```python
# src/data/__init__.py
from .loader import DataLoader
from .cleaner import DataCleaner
from .alignment import TimestampAligner

__all__ = ['DataLoader', 'DataCleaner', 'TimestampAligner']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_alignment.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit timestamp alignment**

```bash
git add src/data/alignment.py tests/data/test_alignment.py
git commit -m "feat: implement timestamp alignment module

- Add TimestampAligner for cross-source timestamp alignment
- Convert Unix timestamps and string timestamps to UTC
- Align trades data (IST) to sentiment data (UTC)
- Merge trades with daily sentiment
- Add unit tests for alignment operations

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

## Phase 2: Core Research Components (Institutional-Grade)

### Task 6: Causal Inference - Granger Causality Tests

**Files:**
- Create: `src/research/__init__.py`
- Create: `src/research/causality.py`
- Create: `tests/research/test_causality.py`

- [ ] **Step 1: Write test for Granger causality**

```python
# tests/research/test_causality.py
import pytest
import pandas as pd
import numpy as np
from src.research.causality import CausalityAnalyzer


@pytest.fixture
def sample_time_series():
    """Create sample time series data"""
    np.random.seed(42)
    n = 100
    
    # Create causal relationship: X causes Y with 1-day lag
    x = np.random.randn(n)
    y = np.zeros(n)
    y[0] = np.random.randn()
    
    for i in range(1, n):
        y[i] = 0.7 * x[i-1] + 0.3 * y[i-1] + np.random.randn() * 0.1
    
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'sentiment': x,
        'profitability': y
    })


def test_granger_causality_test(sample_time_series):
    """Test Granger causality detection"""
    analyzer = CausalityAnalyzer()
    
    result = analyzer.granger_causality_test(
        sample_time_series,
        cause='sentiment',
        effect='profitability',
        max_lag=5
    )
    
    assert 'p_value' in result
    assert 'f_statistic' in result
    assert 'optimal_lag' in result
    assert isinstance(result['p_value'], float)


def test_bidirectional_causality(sample_time_series):
    """Test bidirectional causality analysis"""
    analyzer = CausalityAnalyzer()
    
    result = analyzer.bidirectional_causality(
        sample_time_series,
        var1='sentiment',
        var2='profitability',
        max_lag=5
    )
    
    assert 'sentiment_causes_profitability' in result
    assert 'profitability_causes_sentiment' in result
    assert isinstance(result['sentiment_causes_profitability']['p_value'], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_causality.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.research.causality'"

- [ ] **Step 3: Implement CausalityAnalyzer class**

```python
# src/research/causality.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from statsmodels.tsa.stattools import grangercausalitytests
from src.utils.logger import setup_logger


class CausalityAnalyzer:
    """Analyze causal relationships using statistical tests"""
    
    def __init__(self):
        """Initialize CausalityAnalyzer"""
        self.logger = setup_logger(__name__)
    
    def granger_causality_test(
        self,
        df: pd.DataFrame,
        cause: str,
        effect: str,
        max_lag: int = 7,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if 'cause' Granger-causes 'effect'
        
        Granger causality tests whether past values of X help predict Y
        beyond what Y's own past values can predict.
        
        Args:
            df: DataFrame with time series data
            cause: Column name of potential cause
            effect: Column name of potential effect
            max_lag: Maximum lag to test
            significance_level: Significance threshold
            
        Returns:
            Dictionary with test results
        """
        # Prepare data (effect, cause order for statsmodels)
        data = df[[effect, cause]].dropna()
        
        if len(data) < max_lag + 10:
            raise ValueError(f"Insufficient data: need at least {max_lag + 10} samples")
        
        # Run Granger causality test
        try:
            test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        except Exception as e:
            self.logger.error(f"Granger causality test failed: {e}")
            raise
        
        # Extract results for each lag
        results_by_lag = {}
        for lag in range(1, max_lag + 1):
            # Get F-test results (ssr_ftest)
            f_test = test_result[lag][0]['ssr_ftest']
            results_by_lag[lag] = {
                'f_statistic': f_test[0],
                'p_value': f_test[1]
            }
        
        # Find optimal lag (lowest p-value)
        optimal_lag = min(results_by_lag.keys(), 
                         key=lambda k: results_by_lag[k]['p_value'])
        
        optimal_result = results_by_lag[optimal_lag]
        is_causal = optimal_result['p_value'] < significance_level
        
        result = {
            'cause': cause,
            'effect': effect,
            'optimal_lag': optimal_lag,
            'p_value': optimal_result['p_value'],
            'f_statistic': optimal_result['f_statistic'],
            'is_causal': is_causal,
            'significance_level': significance_level,
            'results_by_lag': results_by_lag
        }
        
        if is_causal:
            self.logger.info(
                f"'{cause}' Granger-causes '{effect}' at lag {optimal_lag} "
                f"(p={optimal_result['p_value']:.4f})"
            )
        else:
            self.logger.info(
                f"No Granger causality detected from '{cause}' to '{effect}' "
                f"(p={optimal_result['p_value']:.4f})"
            )
        
        return result
    
    def bidirectional_causality(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        max_lag: int = 7
    ) -> Dict[str, Any]:
        """
        Test bidirectional causality between two variables
        
        Tests both:
        - Does var1 cause var2?
        - Does var2 cause var1?
        
        Args:
            df: DataFrame with time series data
            var1: First variable
            var2: Second variable
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with bidirectional test results
        """
        # Test var1 -> var2
        result_1_to_2 = self.granger_causality_test(
            df, cause=var1, effect=var2, max_lag=max_lag
        )
        
        # Test var2 -> var1
        result_2_to_1 = self.granger_causality_test(
            df, cause=var2, effect=var1, max_lag=max_lag
        )
        
        # Determine relationship type
        if result_1_to_2['is_causal'] and result_2_to_1['is_causal']:
            relationship = 'bidirectional'
        elif result_1_to_2['is_causal']:
            relationship = f'{var1} -> {var2}'
        elif result_2_to_1['is_causal']:
            relationship = f'{var2} -> {var1}'
        else:
            relationship = 'no_causality'
        
        return {
            f'{var1}_causes_{var2}': result_1_to_2,
            f'{var2}_causes_{var1}': result_2_to_1,
            'relationship': relationship
        }
```

- [ ] **Step 4: Create research __init__.py**

```python
# src/research/__init__.py
from .causality import CausalityAnalyzer

__all__ = ['CausalityAnalyzer']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/research/test_causality.py -v`
Expected: PASS (all 2 tests)

- [ ] **Step 6: Commit causality analysis**

```bash
git add src/research/ tests/research/
git commit -m "feat: implement Granger causality analysis

- Add CausalityAnalyzer for testing causal relationships
- Implement Granger causality tests with multiple lags
- Add bidirectional causality testing
- Find optimal lag with lowest p-value
- Add unit tests for causality detection

This addresses the institutional critique: we now test causation,
not just correlation.

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

### Task 7: Signal Decay Framework

**Files:**
- Create: `src/research/signal_decay.py`
- Create: `tests/research/test_signal_decay.py`

- [ ] **Step 1: Write test for signal decay analysis**

```python
# tests/research/test_signal_decay.py
import pytest
import pandas as pd
import numpy as np
from src.research.signal_decay import SignalDecayAnalyzer


@pytest.fixture
def sample_signal_data():
    """Create sample data with decaying signal"""
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range('2024-01-01', periods=n)
    signal = np.random.randn(n)
    
    # Create target with decaying correlation
    target = np.zeros(n)
    for i in range(n):
        # Signal effect decays over 10 days
        for lag in range(1, 11):
            if i >= lag:
                decay_factor = np.exp(-lag / 5)  # Exponential decay
                target[i] += signal[i - lag] * decay_factor * 0.3
        target[i] += np.random.randn() * 0.5
    
    return pd.DataFrame({
        'date': dates,
        'signal': signal,
        'target': target
    })


def test_compute_lagged_correlation(sample_signal_data):
    """Test lagged correlation computation"""
    analyzer = SignalDecayAnalyzer()
    
    correlations = analyzer.compute_lagged_correlation(
        sample_signal_data,
        signal_col='signal',
        target_col='target',
        max_lag=15
    )
    
    assert len(correlations) == 15
    assert all(isinstance(c, float) for c in correlations.values())
    assert all(-1 <= c <= 1 for c in correlations.values())


def test_find_half_life(sample_signal_data):
    """Test half-life calculation"""
    analyzer = SignalDecayAnalyzer()
    
    correlations = analyzer.compute_lagged_correlation(
        sample_signal_data,
        signal_col='signal',
        target_col='target',
        max_lag=15
    )
    
    half_life = analyzer.find_half_life(correlations)
    
    assert isinstance(half_life, (int, float))
    assert half_life > 0


def test_measure_predictive_decay(sample_signal_data):
    """Test predictive power decay measurement"""
    analyzer = SignalDecayAnalyzer()
    
    decay_results = analyzer.measure_predictive_decay(
        sample_signal_data,
        signal_col='signal',
        target_col='target',
        max_horizon=10
    )
    
    assert 'correlations' in decay_results
    assert 'half_life' in decay_results
    assert 'peak_lag' in decay_results
    assert 'peak_correlation' in decay_results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/research/test_signal_decay.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.research.signal_decay'"

- [ ] **Step 3: Implement SignalDecayAnalyzer class**

```python
# src/research/signal_decay.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy.stats import spearmanr
from src.utils.logger import setup_logger


class SignalDecayAnalyzer:
    """Analyze how signal predictive power decays over time"""
    
    def __init__(self):
        """Initialize SignalDecayAnalyzer"""
        self.logger = setup_logger(__name__)
    
    def compute_lagged_correlation(
        self,
        df: pd.DataFrame,
        signal_col: str,
        target_col: str,
        max_lag: int = 30,
        method: str = 'spearman'
    ) -> Dict[int, float]:
        """
        Compute correlation between signal and target at different lags
        
        Args:
            df: DataFrame with signal and target
            signal_col: Signal column name
            target_col: Target column name
            max_lag: Maximum lag to test (days)
            method: 'spearman' or 'pearson'
            
        Returns:
            Dictionary mapping lag to correlation
        """
        correlations = {}
        
        for lag in range(1, max_lag + 1):
            # Shift signal by lag
            signal_lagged = df[signal_col].shift(lag)
            target = df[target_col]
            
            # Remove NaN values
            valid_idx = ~(signal_lagged.isna() | target.isna())
            signal_clean = signal_lagged[valid_idx]
            target_clean = target[valid_idx]
            
            if len(signal_clean) < 10:
                correlations[lag] = np.nan
                continue
            
            # Compute correlation
            if method == 'spearman':
                corr, _ = spearmanr(signal_clean, target_clean)
            elif method == 'pearson':
                corr = np.corrcoef(signal_clean, target_clean)[0, 1]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            correlations[lag] = corr
        
        self.logger.info(f"Computed lagged correlations for {max_lag} lags")
        
        return correlations
    
    def find_half_life(self, correlations: Dict[int, float]) -> Optional[float]:
        """
        Find half-life of signal (lag where correlation drops to 50% of peak)
        
        Args:
            correlations: Dictionary mapping lag to correlation
            
        Returns:
            Half-life in days (or None if not found)
        """
        # Find peak correlation
        valid_corrs = {k: abs(v) for k, v in correlations.items() if not np.isnan(v)}
        
        if not valid_corrs:
            return None
        
        peak_lag = max(valid_corrs.keys(), key=lambda k: valid_corrs[k])
        peak_corr = valid_corrs[peak_lag]
        
        if peak_corr == 0:
            return None
        
        # Find lag where correlation drops to 50% of peak
        half_corr = peak_corr * 0.5
        
        for lag in sorted(valid_corrs.keys()):
            if lag > peak_lag and valid_corrs[lag] <= half_corr:
                return lag
        
        # If never drops to 50%, return None
        return None
    
    def measure_predictive_decay(
        self,
        df: pd.DataFrame,
        signal_col: str,
        target_col: str,
        max_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Comprehensive signal decay analysis
        
        Args:
            df: DataFrame with signal and target
            signal_col: Signal column name
            target_col: Target column name
            max_horizon: Maximum forecast horizon
            
        Returns:
            Dictionary with decay analysis results
        """
        # Compute lagged correlations
        correlations = self.compute_lagged_correlation(
            df, signal_col, target_col, max_lag=max_horizon
        )
        
        # Find peak correlation
        valid_corrs = {k: abs(v) for k, v in correlations.items() if not np.isnan(v)}
        peak_lag = max(valid_corrs.keys(), key=lambda k: valid_corrs[k])
        peak_correlation = correlations[peak_lag]
        
        # Find half-life
        half_life = self.find_half_life(correlations)
        
        # Compute decay rate (exponential fit)
        lags = np.array(list(valid_corrs.keys()))
        corrs = np.array([valid_corrs[k] for k in lags])
        
        # Fit exponential decay: corr = a * exp(-b * lag)
        if len(lags) > 3:
            log_corrs = np.log(corrs + 1e-10)  # Avoid log(0)
            decay_rate = -np.polyfit(lags, log_corrs, 1)[0]
        else:
            decay_rate = None
        
        result = {
            'correlations': correlations,
            'peak_lag': peak_lag,
            'peak_correlation': peak_correlation,
            'half_life': half_life,
            'decay_rate': decay_rate,
            'signal_col': signal_col,
            'target_col': target_col
        }
        
        self.logger.info(
            f"Signal decay analysis: peak at lag {peak_lag} "
            f"(corr={peak_correlation:.3f}), half-life={half_life} days"
        )
        
        return result
    
    def compare_regime_decay(
        self,
        df: pd.DataFrame,
        signal_col: str,
        target_col: str,
        regime_col: str,
        max_horizon: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare signal decay across different regimes
        
        Args:
            df: DataFrame with signal, target, and regime
            signal_col: Signal column name
            target_col: Target column name
            regime_col: Regime column name
            max_horizon: Maximum forecast horizon
            
        Returns:
            Dictionary mapping regime to decay results
        """
        regimes = df[regime_col].unique()
        results = {}
        
        for regime in regimes:
            regime_df = df[df[regime_col] == regime].copy()
            
            if len(regime_df) < 50:
                self.logger.warning(f"Insufficient data for regime '{regime}' (n={len(regime_df)})")
                continue
            
            decay_result = self.measure_predictive_decay(
                regime_df, signal_col, target_col, max_horizon
            )
            
            results[regime] = decay_result
        
        self.logger.info(f"Compared signal decay across {len(results)} regimes")
        
        return results
```

- [ ] **Step 4: Update research __init__.py**

```python
# src/research/__init__.py
from .causality import CausalityAnalyzer
from .signal_decay import SignalDecayAnalyzer

__all__ = ['CausalityAnalyzer', 'SignalDecayAnalyzer']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/research/test_signal_decay.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Commit signal decay framework**

```bash
git add src/research/signal_decay.py tests/research/test_signal_decay.py
git commit -m "feat: implement signal decay analysis framework

- Add SignalDecayAnalyzer for measuring predictive power decay
- Compute lagged correlations to measure temporal persistence
- Calculate signal half-life (50% decay point)
- Fit exponential decay rate
- Compare decay across market regimes
- Add unit tests for decay analysis

This addresses institutional requirement: understanding how long
signals remain predictive.

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

Due to length constraints, I'll create a summary of the remaining tasks and then you can request specific sections to be expanded. The complete plan would include:

## Remaining Phases Summary

**Phase 2 (continued):**
- Task 8: Feature Stability & Robustness Tests
- Task 9: Hypothesis Testing Framework
- Task 10: Lagged Effects Analysis

**Phase 3: Behavioral Metrics & Regime Detection**
- Task 11: Behavioral Metrics (revenge trading, overconfidence, loss aversion)
- Task 12: Regime Detection (fear/greed, volatility, liquidity)
- Task 13: Regime Transition Analysis

**Phase 4: Feature Engineering & Modeling**
- Task 14: Sentiment Features
- Task 15: Trader Behavior Features
- Task 16: Risk Metrics
- Task 17: Target Construction (intraday, next-day, 7-day)
- Task 18: Baseline Model (Logistic Regression)
- Task 19: Tree Models (XGBoost, LightGBM)
- Task 20: Temporal Validation
- Task 21: MLflow Integration

**Phase 5: Backtesting & Documentation**
- Task 22: Transaction Costs Module
- Task 23: Backtest Engine
- Task 24: Performance Metrics
- Task 25: Research Journal
- Task 26: Findings Document
- Task 27: Notebooks

Would you like me to:
1. **Continue with full detail for all remaining tasks** (will add ~4000 more lines)
2. **Expand specific phases** (e.g., just Phase 3 in full detail)
3. **Start implementation** of what we have so far

Which would you prefer?
# Implementation Plan - Part 2 (Continuation)

## Phase 2 (Continued): Core Research Components

### Task 8: Feature Stability & Robustness Tests

**Files:**
- Create: `src/validation/__init__.py`
- Create: `src/validation/feature_stability.py`
- Create: `tests/validation/test_feature_stability.py`

- [ ] **Step 1: Write test for feature stability**

```python
# tests/validation/test_feature_stability.py
import pytest
import pandas as pd
import numpy as np
from src.validation.feature_stability import FeatureStabilityValidator


@pytest.fixture
def sample_feature_data():
    """Create sample feature data with drift"""
    np.random.seed(42)
    
    # Train period: stable distribution
    train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'target': np.random.binomial(1, 0.5, 100)
    })
    
    # Test period: drifted distribution
    test = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 100),  # Mean shift + variance increase
        'feature2': np.random.normal(5, 2, 100),  # Stable
        'target': np.random.binomial(1, 0.5, 100)
    })
    
    return train, test


def test_compute_psi(sample_feature_data):
    """Test Population Stability Index calculation"""
    train, test = sample_feature_data
    validator = FeatureStabilityValidator()
    
    psi = validator.compute_psi(
        train['feature1'].values,
        test['feature1'].values
    )
    
    assert isinstance(psi, float)
    assert psi >= 0


def test_detect_distribution_shift(sample_feature_data):
    """Test distribution shift detection"""
    train, test = sample_feature_data
    validator = FeatureStabilityValidator()
    
    results = validator.detect_distribution_shift(train, test)
    
    assert 'feature1' in results
    assert 'feature2' in results
    assert 'psi' in results['feature1']
    assert 'ks_statistic' in results['feature1']
    assert 'p_value' in results['feature1']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_feature_stability.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement FeatureStabilityValidator**

```python
# src/validation/feature_stability.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import ks_2samp
from src.utils.logger import setup_logger


class FeatureStabilityValidator:
    """Validate feature stability across time periods"""
    
    def __init__(self):
        """Initialize validator"""
        self.logger = setup_logger(__name__)
    
    def compute_psi(
        self,
        train_dist: np.ndarray,
        test_dist: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI)
        
        PSI measures distribution shift between train and test.
        PSI < 0.1: Stable
        0.1 < PSI < 0.25: Moderate drift
        PSI > 0.25: Significant drift
        
        Args:
            train_dist: Training distribution
            test_dist: Test distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on training distribution
        bin_edges = np.histogram_bin_edges(train_dist, bins=bins)
        
        # Compute percentages in each bin
        train_pct, _ = np.histogram(train_dist, bins=bin_edges)
        test_pct, _ = np.histogram(test_dist, bins=bin_edges)
        
        # Avoid division by zero
        train_pct = train_pct / len(train_dist) + 1e-10
        test_pct = test_pct / len(test_dist) + 1e-10
        
        # Compute PSI
        psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        
        return psi
    
    def detect_distribution_shift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exclude_cols: list = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect distribution shifts for all features
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            exclude_cols: Columns to exclude from analysis
            
        Returns:
            Dictionary with shift metrics per feature
        """
        if exclude_cols is None:
            exclude_cols = []
        
        results = {}
        
        for col in train_df.columns:
            if col in exclude_cols:
                continue
            
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                continue
            
            train_vals = train_df[col].dropna().values
            test_vals = test_df[col].dropna().values
            
            if len(train_vals) < 10 or len(test_vals) < 10:
                continue
            
            # Compute PSI
            psi = self.compute_psi(train_vals, test_vals)
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(train_vals, test_vals)
            
            # Determine stability
            if psi < 0.1:
                stability = 'stable'
            elif psi < 0.25:
                stability = 'moderate_drift'
            else:
                stability = 'significant_drift'
            
            results[col] = {
                'psi': psi,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'stability': stability
            }
            
            if stability != 'stable':
                self.logger.warning(
                    f"Feature '{col}' shows {stability} (PSI={psi:.3f})"
                )
        
        return results
```

- [ ] **Step 4: Create validation __init__.py**

```python
# src/validation/__init__.py
from .feature_stability import FeatureStabilityValidator

__all__ = ['FeatureStabilityValidator']
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/validation/test_feature_stability.py -v`
Expected: PASS

- [ ] **Step 6: Commit feature stability validation**

```bash
git add src/validation/ tests/validation/
git commit -m "feat: implement feature stability validation

- Add FeatureStabilityValidator for detecting distribution drift
- Implement Population Stability Index (PSI) calculation
- Add Kolmogorov-Smirnov test for distribution comparison
- Classify features as stable/moderate_drift/significant_drift
- Add unit tests for stability validation

Addresses institutional critique: detecting unstable features
that will fail out-of-sample.

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

## Phase 3: Behavioral Metrics & Regime Detection

### Task 9: Behavioral Metrics Module

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/behavioral_metrics.py`
- Create: `tests/features/test_behavioral_metrics.py`

- [ ] **Step 1: Write test for behavioral metrics**

```python
# tests/features/test_behavioral_metrics.py
import pytest
import pandas as pd
import numpy as np
from src.features.behavioral_metrics import BehavioralMetricsEngine


@pytest.fixture
def sample_trader_data():
    """Create sample trader data"""
    np.random.seed(42)
    
    return pd.DataFrame({
        'account': ['0xabc'] * 10,
        'datetime': pd.date_range('2024-01-01', periods=10),
        'closed_pnl': [100, -50, 200, -100, 150, -75, 300, -200, 250, -150],
        'leverage': [5, 8, 6, 12, 7, 15, 6, 20, 7, 25],  # Increasing after losses
        'size_usd': [1000, 1500, 1200, 2000, 1300, 2500, 1400, 3000, 1500, 3500]
    })


def test_revenge_trading_score(sample_trader_data):
    """Test revenge trading detection"""
    engine = BehavioralMetricsEngine()
    
    score = engine.compute_revenge_trading_score(sample_trader_data)
    
    assert isinstance(score, float)
    assert score >= 0


def test_overconfidence_ratio(sample_trader_data):
    """Test overconfidence detection"""
    engine = BehavioralMetricsEngine()
    
    ratio = engine.compute_overconfidence_ratio(sample_trader_data)
    
    assert isinstance(ratio, float)
    assert ratio >= 0


def test_loss_aversion_asymmetry(sample_trader_data):
    """Test loss aversion measurement"""
    engine = BehavioralMetricsEngine()
    
    asymmetry = engine.compute_loss_aversion_asymmetry(sample_trader_data)
    
    assert isinstance(asymmetry, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_behavioral_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BehavioralMetricsEngine**

```python
# src/features/behavioral_metrics.py
import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import setup_logger


class BehavioralMetricsEngine:
    """Compute behavioral finance metrics"""
    
    def __init__(self):
        """Initialize engine"""
        self.logger = setup_logger(__name__)
    
    def compute_revenge_trading_score(
        self,
        df: pd.DataFrame,
        pnl_col: str = 'closed_pnl',
        leverage_col: str = 'leverage'
    ) -> float:
        """
        Measure revenge trading: leverage increase after losses
        
        Revenge trading occurs when traders increase risk after losses
        in an attempt to recover quickly.
        
        Args:
            df: Trader DataFrame sorted by time
            pnl_col: P&L column name
            leverage_col: Leverage column name
            
        Returns:
            Revenge trading score (0-1, higher = more revenge trading)
        """
        df = df.sort_values('datetime').copy()
        
        # Identify losses
        df['is_loss'] = df[pnl_col] < 0
        
        # Compute leverage change after each trade
        df['leverage_change'] = df[leverage_col].diff()
        
        # Leverage increases after losses
        revenge_trades = df[df['is_loss'].shift(1) == True]
        
        if len(revenge_trades) == 0:
            return 0.0
        
        # Average leverage increase after losses
        avg_increase_after_loss = revenge_trades['leverage_change'].mean()
        
        # Average leverage change overall
        avg_change_overall = df['leverage_change'].mean()
        
        # Score: how much more leverage increases after losses vs. overall
        if avg_change_overall <= 0:
            score = 1.0 if avg_increase_after_loss > 0 else 0.0
        else:
            score = max(0, min(1, avg_increase_after_loss / (avg_change_overall + 1e-10)))
        
        return score
    
    def compute_overconfidence_ratio(
        self,
        df: pd.DataFrame,
        pnl_col: str = 'closed_pnl',
        size_col: str = 'size_usd'
    ) -> float:
        """
        Measure overconfidence: position size increase after wins
        
        Overconfident traders increase position sizes after winning streaks,
        believing they have "hot hands".
        
        Args:
            df: Trader DataFrame sorted by time
            pnl_col: P&L column name
            size_col: Position size column name
            
        Returns:
            Overconfidence ratio (>1 = overconfident)
        """
        df = df.sort_values('datetime').copy()
        
        # Identify wins
        df['is_win'] = df[pnl_col] > 0
        
        # Compute size change
        df['size_change_pct'] = df[size_col].pct_change()
        
        # Size changes after wins vs. after losses
        size_change_after_win = df[df['is_win'].shift(1) == True]['size_change_pct'].mean()
        size_change_after_loss = df[df['is_win'].shift(1) == False]['size_change_pct'].mean()
        
        if pd.isna(size_change_after_win) or pd.isna(size_change_after_loss):
            return 1.0
        
        # Ratio: how much more size increases after wins vs. losses
        if size_change_after_loss <= 0:
            ratio = 2.0 if size_change_after_win > 0 else 1.0
        else:
            ratio = (size_change_after_win + 1) / (size_change_after_loss + 1)
        
        return ratio
    
    def compute_loss_aversion_asymmetry(
        self,
        df: pd.DataFrame,
        pnl_col: str = 'closed_pnl'
    ) -> float:
        """
        Measure loss aversion: asymmetric behavior in gains vs. losses
        
        Loss-averse traders hold losing positions longer (hoping for recovery)
        and exit winning positions faster (fear of giving back gains).
        
        Args:
            df: Trader DataFrame
            pnl_col: P&L column name
            
        Returns:
            Asymmetry score (>0 = loss averse)
        """
        wins = df[df[pnl_col] > 0][pnl_col]
        losses = df[df[pnl_col] < 0][pnl_col]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        # Average win size vs. average loss size
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Loss aversion: losses are larger than wins (hold losers too long)
        asymmetry = avg_loss / avg_win - 1
        
        return asymmetry
```

- [ ] **Step 4: Create features __init__.py**

```python
# src/features/__init__.py
from .behavioral_metrics import BehavioralMetricsEngine

__all__ = ['BehavioralMetricsEngine']
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/features/test_behavioral_metrics.py -v`
Expected: PASS

- [ ] **Step 6: Commit behavioral metrics**

```bash
git add src/features/ tests/features/
git commit -m "feat: implement behavioral finance metrics

- Add BehavioralMetricsEngine for quantifying emotional trading
- Implement revenge trading score (leverage increase after losses)
- Implement overconfidence ratio (size increase after wins)
- Implement loss aversion asymmetry (holding losers too long)
- Add unit tests for behavioral metrics

This is the project's unique intellectual contribution:
quantifying trader psychology.

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

---

## Remaining Implementation Roadmap

**The plan now includes 9 detailed tasks with complete TDD implementation:**

**Phase 1: Foundation** (Tasks 1-5) ✅
- Project structure, config, logging
- Data loading, cleaning, alignment

**Phase 2: Core Research** (Tasks 6-8) ✅
- Granger causality (causal inference)
- Signal decay framework
- Feature stability validation

**Phase 3: Behavioral Metrics** (Task 9) ✅
- Revenge trading, overconfidence, loss aversion

**Remaining Work** (follow same TDD pattern):

**Phase 3 (continued):**
- Task 10: Regime Detection (fear/greed, volatility)
- Task 11: Regime Transition Analysis

**Phase 4: Feature Engineering & Modeling:**
- Task 12: Sentiment Features
- Task 13: Trader Behavior Features  
- Task 14: Risk Metrics
- Task 15: Target Construction (leakage-safe)
- Task 16: Baseline Model + MLflow
- Task 17: Tree Models
- Task 18: Temporal Validation

**Phase 5: Backtesting & Documentation:**
- Task 19: Transaction Costs
- Task 20: Backtest Engine
- Task 21: Performance Metrics
- Task 22: Research Journal
- Task 23: Findings Document
- Task 24: Notebooks

---

## Next Steps

**Execute the plan using:**

```bash
# Option 1: Subagent-driven (recommended)
# Invoke superpowers:subagent-driven-development skill

# Option 2: Inline execution
# Invoke superpowers:executing-plans skill
```

**The detailed tasks (1-9) provide:**
- Complete TDD workflow
- Exact code implementations
- Test commands with expected outputs
- Git commit messages
- Institutional-grade components

**Start with Task 1 and work sequentially through Task 9 to build the foundation.**
