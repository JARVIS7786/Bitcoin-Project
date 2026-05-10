# PrimeTRADE Methodology

## Research Approach

### 1. Causal Analysis Framework

**Granger Causality Testing**
- Tests whether past values of Bitcoin sentiment help predict future trader behavior
- Establishes temporal precedence (not just correlation)
- Maximum lag testing: 10 periods
- Significance level: α = 0.05

**Signal Decay Analysis**
- Exponential decay model: `y = initial * exp(-decay_rate * t)`
- Half-life calculation: `t_half = ln(2) / decay_rate`
- Determines optimal feature windows
- Identifies when signals lose predictive power

### 2. Behavioral Finance Metrics

**Revenge Trading Score**
- Correlation between prior loss and subsequent leverage increase
- Positive score indicates revenge trading behavior
- Formula: `corr(prior_loss_indicator, leverage_change)`

**Overconfidence Ratio**
- Position size increase after winning streaks
- Ratio > 1 indicates overconfidence
- Formula: `avg_position_after_streak / avg_position_before_streak`

**Loss Aversion Asymmetry**
- Measures if losses are held longer/larger than wins
- Formula: `(avg_loss / avg_win) - 1`
- Positive values indicate loss aversion

**Emotional Leverage Expansion**
- Leverage variance during extreme sentiment vs normal periods
- Formula: `var_extreme / var_normal`
- Higher values indicate emotional decision-making

### 3. Feature Engineering

**Sentiment Features**
- Rolling statistics (MA, std, min, max) over windows [3, 7, 14, 30]
- Momentum indicators (1-day, 7-day changes)
- Extreme sentiment flags (fear < 20, greed > 80)

**Trader Behavior Features**
- Win rate, average PnL, average leverage
- Trade count, position sizing patterns
- Behavioral metrics per trader

**Risk Features**
- PnL volatility over rolling windows
- Leverage volatility
- Drawdown metrics

**Temporal Features**
- Hour of day, day of week
- Weekend indicator
- Time-based patterns

### 4. Validation Strategy

**Walk-Forward Validation**
- No k-fold cross-validation (violates temporal ordering)
- Sequential train/test splits
- Test size: 20% of data
- Ensures no future information leakage

**Feature Stability Checks**
- Population Stability Index (PSI)
  - PSI < 0.1: No significant change
  - 0.1 ≤ PSI < 0.2: Moderate change
  - PSI ≥ 0.2: Significant change (action required)
- Kolmogorov-Smirnov test for distribution equality
- Threshold: p-value > 0.05

**Leakage Detection**
- Check for perfect correlations (|corr| > 0.99)
- Validate temporal ordering
- Ensure features use only past data

### 5. Model Selection

**Baseline: Logistic Regression**
- Simple, interpretable
- Establishes performance floor
- Class weight balancing for imbalanced data

**Tree Models: XGBoost & LightGBM**
- Handle non-linear relationships
- Feature importance analysis
- Hyperparameters:
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 100

**Evaluation Metrics**
- Primary: ROC-AUC
- Secondary: Precision, Recall, F1, Accuracy
- Focus on precision (minimize false positives)

### 6. Backtesting Framework

**Transaction Costs**
- Maker fee: 0.02%
- Taker fee: 0.05%
- Slippage: 5 basis points
- Realistic cost modeling prevents overfitting

**Performance Metrics**
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

**Risk Management**
- Maximum drawdown limit: 20%
- Stop loss: 5%
- Take profit: 15%
- Position sizing: Fixed $1000 per trade

## Statistical Methods

### Hypothesis Testing
- Null hypothesis: Sentiment does NOT Granger-cause trader behavior
- Alternative: Sentiment Granger-causes trader behavior
- Test statistic: F-test on lagged coefficients
- Decision rule: Reject H0 if p-value < 0.05

### Time Series Considerations
- Stationarity checks (ADF test)
- Autocorrelation analysis
- Seasonal decomposition if needed

## Data Quality Standards

**Missing Values**
- Maximum missing ratio: 30% per column
- Strategy: Drop columns exceeding threshold, then drop rows

**Outlier Detection**
- IQR method with multiplier = 3.0
- Bounds: [Q1 - 3*IQR, Q3 + 3*IQR]
- Remove outliers from numeric features

**Timestamp Alignment**
- Source: IST (Asia/Kolkata)
- Target: UTC
- Merge strategy: asof with 1-hour tolerance

## Reproducibility

- Random seed: 42
- MLflow experiment tracking
- Version control for all code
- Configuration files for all parameters
- Comprehensive test coverage (>80%)

## Limitations

1. **Data Scope**: Limited to Hyperliquid traders and Bitcoin sentiment
2. **Market Conditions**: Trained on specific market regime
3. **Execution**: Assumes perfect order execution (backtesting)
4. **Slippage**: Simplified slippage model
5. **Market Impact**: Not modeled for large positions

## Future Work

- Multi-asset expansion
- Real-time prediction system
- Advanced position sizing (Kelly criterion)
- Regime-adaptive models
- Ensemble methods
