# PrimeTRADE

**Institutional-grade Bitcoin Sentiment & Trader Performance Prediction System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

PrimeTRADE is a research-driven trading system that exploits the relationship between extreme Bitcoin sentiment and predictable behavioral biases in Hyperliquid traders. Built on rigorous causal analysis and behavioral finance principles.

## Core Thesis

**Extreme Bitcoin sentiment drives predictable behavioral biases in Hyperliquid traders — quantifiable and exploitable.**

## Key Features

- **Causal Analysis**: Granger causality tests (not just correlation)
- **Behavioral Metrics**: Revenge trading, overconfidence, loss aversion quantification
- **Signal Decay Framework**: Half-life calculation for feature relevance
- **Regime Detection**: Fear/greed market state identification
- **Temporal Validation**: Walk-forward validation with strict no-leakage guarantees
- **Transaction Cost Modeling**: Realistic backtesting with slippage and fees
- **MLflow Integration**: Full experiment tracking and model versioning

## Project Structure

```
PrimeTRADE/
├── src/
│   ├── data/           # Data loading, cleaning, alignment
│   ├── features/       # Feature engineering (behavioral, sentiment, risk, temporal)
│   ├── signals/        # Signal generation and decay analysis
│   ├── regimes/        # Market regime detection
│   ├── research/       # Causality, hypothesis testing, lagged effects
│   ├── validation/     # Feature stability, leakage checks, temporal validation
│   ├── models/         # Baseline, tree models, training, evaluation
│   ├── backtesting/    # Transaction costs, strategy simulation, performance metrics
│   ├── pipelines/      # End-to-end orchestration
│   ├── analysis/       # Statistical analysis utilities
│   ├── visualization/  # Plotting and reporting
│   └── utils/          # Config, logging, helpers
├── tests/              # Comprehensive test suite (80%+ coverage)
├── notebooks/          # Exploratory analysis
├── configs/            # YAML configuration files
├── experiments/        # MLflow experiment tracking
├── outputs/            # Figures, tables, reports
├── data/               # Raw and processed data (gitignored)
└── docs/               # Methodology, findings, research journal

```

## Installation

```bash
# Clone the repository
git clone https://github.com/JARVIS7786/PrimeTRADE.git
cd PrimeTRADE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### 1. Data Pipeline
```python
from primetrade.pipelines import DataPipeline

pipeline = DataPipeline(config_path="configs/data_config.yaml")
cleaned_data = pipeline.run()
```

### 2. Feature Engineering
```python
from primetrade.pipelines import FeaturePipeline

feature_pipeline = FeaturePipeline(config_path="configs/feature_config.yaml")
features = feature_pipeline.run(cleaned_data)
```

### 3. Model Training
```python
from primetrade.pipelines import TrainingPipeline

training_pipeline = TrainingPipeline(config_path="configs/model_config.yaml")
model, metrics = training_pipeline.run(features)
```

### 4. Backtesting
```python
from primetrade.backtesting import StrategySimulator

simulator = StrategySimulator(config_path="configs/backtest_config.yaml")
results = simulator.run(model, features)
```

## Behavioral Metrics

- **Revenge Trading Score**: Correlation between prior loss and subsequent leverage increase
- **Overconfidence Ratio**: Position size increase after winning streaks
- **Loss Aversion Asymmetry**: `avg_loss / avg_win - 1`
- **Emotional Leverage Expansion**: Leverage variance during extreme sentiment

## Prediction Targets

- `is_profitable_intraday`: Same-day profitability
- `is_profitable_next_day`: Next-day profitability
- `is_profitable_7d`: 7-day profitability

## Validation Principles

1. **Walk-forward validation only** (no k-fold on time series)
2. **All features use ONLY data available before prediction timestamp**
3. **No target leakage** (strict temporal ordering)
4. **Feature stability checks** (PSI, KS tests)
5. **Realistic transaction costs** (slippage, fees, market impact)

## Testing

```bash
# Run all tests with coverage
pytest --cov=src tests/ -v

# Run specific test module
pytest tests/data/test_loader.py -v

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

## Documentation

- [Methodology](docs/METHODOLOGY.md): Research approach and statistical methods
- [Findings](docs/FINDINGS.md): Key insights and results
- [Research Journal](docs/research_journal.md): Experiment log and decisions

## Tech Stack

- **Python 3.10+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm, imbalanced-learn
- **Statistics**: scipy, statsmodels
- **Experiment Tracking**: mlflow
- **Visualization**: plotly
- **Testing**: pytest, pytest-cov

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a research project. Contributions welcome via pull requests.

## Disclaimer

This system is for research and educational purposes only. Not financial advice. Trading cryptocurrencies carries significant risk.

---

**Built with rigor. Tested with discipline. Deployed with caution.**
