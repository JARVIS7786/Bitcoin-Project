---
name: Bitcoin Sentiment & Trader Performance Prediction System
description: Institutional-grade quantitative research system analyzing the relationship between market sentiment and trader profitability on Hyperliquid
type: design
date: 2026-05-08
---

# Bitcoin Sentiment & Trader Performance Prediction System

## Executive Summary

This project builds a production-grade quantitative research system that explores how Bitcoin market sentiment (Fear & Greed Index) influences trader performance on Hyperliquid, a decentralized perpetual futures exchange. The system combines behavioral finance theory with machine learning to predict trader profitability across multiple time horizons (intraday, next-day, 7-day).

**Core Thesis:** Market sentiment creates predictable behavioral patterns in trader decision-making, particularly around leverage usage, position sizing, and risk-taking behavior. These patterns can be quantified and used to predict profitability.

**Key Differentiators:**
- Behavioral finance as the intellectual foundation
- Separation of features vs. validated signals
- Regime-aware modeling
- Temporal validation to prevent leakage
- Backtesting with transaction costs
- Research journal documenting hypothesis evolution

---

## 1. Project Architecture

### 1.1 Final Folder Structure

```
PrimeTRADE/
├── data/
│   ├── raw/                          # Original datasets (immutable)
│   │   ├── fear_greed_index.csv
│   │   └── historical_data.csv
│   ├── processed/                    # Cleaned, aligned data
│   │   ├── sentiment_daily.parquet
│   │   ├── trades_cleaned.parquet
│   │   └── merged_dataset.parquet
│   └── features/                     # Versioned feature stores
│       ├── features_v1.parquet
│       └── features_v2.parquet
│
├── src/
│   ├── data/                         # Data ingestion & preprocessing
│   │   ├── loader.py                 # Load raw datasets
│   │   ├── cleaner.py                # Data validation & cleaning
│   │   └── alignment.py              # Timestamp alignment logic
│   │
│   ├── features/                     # Feature engineering (raw variables)
│   │   ├── sentiment.py              # Sentiment-based features
│   │   ├── trader_behavior.py        # Trading pattern features
│   │   ├── risk_metrics.py           # Leverage, volatility, drawdown
│   │   ├── behavioral_metrics.py     # ⭐ Behavioral finance metrics
│   │   ├── temporal.py               # Time-based features
│   │   └── registry.py               # Feature versioning & metadata
│   │
│   ├── signals/                      # ⭐ Validated predictive signals
│   │   ├── sentiment_signals.py      # Extreme greed/fear reversals
│   │   ├── behavioral_signals.py     # Revenge trading, overconfidence
│   │   └── composite_signals.py      # Multi-factor signal combinations
│   │
│   ├── regimes/                      # ⭐ Market regime detection
│   │   ├── fear_regime.py            # Fear/greed regime classification
│   │   ├── volatility_regime.py      # High/low volatility states
│   │   ├── liquidity_regime.py       # Market depth analysis
│   │   └── composite_regime.py       # Multi-dimensional regime detection
│   │
│   ├── research/                     # ⭐ Hypothesis-driven research
│   │   ├── hypothesis_tests.py       # Statistical hypothesis testing
│   │   ├── alpha_signals.py          # Signal discovery & validation
│   │   ├── lagged_effects.py         # Temporal causality analysis
│   │   ├── regime_transitions.py     # Regime shift behavior
│   │   └── behavioral_factors.py     # Behavioral finance factor analysis
│   │
│   ├── validation/                   # ⭐ Financial ML validation
│   │   ├── leakage_checks.py         # Future leakage detection
│   │   ├── temporal_validation.py    # Walk-forward validation
│   │   └── robustness_tests.py       # Stability & sensitivity analysis
│   │
│   ├── models/                       # Model training & evaluation
│   │   ├── baseline.py               # Logistic Regression baseline
│   │   ├── tree_models.py            # RF, XGBoost, LightGBM
│   │   ├── trainer.py                # Training orchestration
│   │   └── evaluator.py              # Metrics & model comparison
│   │
│   ├── backtesting/                  # ⭐ Strategy simulation
│   │   ├── strategy_simulator.py     # Trade execution simulation
│   │   ├── pnl_engine.py             # P&L calculation
│   │   ├── transaction_costs.py      # Slippage, fees, market impact
│   │   └── performance_metrics.py    # Sharpe, Sortino, max drawdown
│   │
│   ├── pipelines/                    # ⭐ End-to-end orchestration
│   │   ├── data_pipeline.py          # Raw → processed → features
│   │   ├── feature_pipeline.py       # Feature generation workflow
│   │   ├── training_pipeline.py      # Model training workflow
│   │   └── inference_pipeline.py     # Prediction generation
│   │
│   ├── analysis/                     # Statistical analysis
│   │   ├── correlation.py            # Correlation & lag analysis
│   │   ├── regime.py                 # Regime-specific analysis
│   │   └── clustering.py             # Trader segmentation
│   │
│   ├── visualization/                # Plotting utilities
│   │   ├── eda_plots.py              # EDA visualizations
│   │   ├── model_plots.py            # Model performance plots
│   │   └── behavioral_plots.py       # Behavioral finance charts
│   │
│   └── utils/                        # Shared utilities
│       ├── config.py                 # Config management
│       ├── logger.py                 # Logging setup
│       └── metrics.py                # Custom metrics
│
├── notebooks/                        # Research narrative
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_behavioral_research.ipynb  # ⭐ Behavioral deep-dive
│   ├── 05_modeling_experiments.ipynb
│   ├── 06_backtesting_results.ipynb  # ⭐ Strategy performance
│   └── 07_insights_and_conclusions.ipynb
│
├── configs/                          # Configuration files
│   ├── data_config.yaml              # Data processing params
│   ├── feature_config.yaml           # Feature engineering specs
│   ├── model_config.yaml             # Model hyperparameters
│   └── backtest_config.yaml          # Backtest parameters
│
├── experiments/                      # MLflow tracking
│   └── mlruns/
│
├── outputs/                          # Generated artifacts
│   ├── figures/                      # Publication-quality plots
│   ├── tables/                       # Model comparison tables
│   └── reports/                      # Generated insights
│
├── tests/                            # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_validation.py
│
├── docs/
│   ├── superpowers/specs/
│   ├── research_journal.md           # ⭐ Research log
│   ├── METHODOLOGY.md                # Research methodology
│   └── FINDINGS.md                   # Key insights
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

### 1.2 Technology Stack

**Core Libraries:**
- Python 3.10+
- pandas, numpy, scipy
- pyarrow (parquet format)
- polars (optional, for performance)

**Machine Learning:**
- scikit-learn (baseline models, preprocessing)
- xgboost (gradient boosting)
- lightgbm (fast gradient boosting)
- imbalanced-learn (class imbalance handling)

**Experiment Tracking:**
- MLflow (lightweight, local-first)

**Configuration:**
- PyYAML (config files)
- python-dotenv (environment variables)
- pydantic (config validation)

**Visualization:**
- matplotlib, seaborn (static plots)
- plotly (interactive visualizations)

**Testing:**
- pytest
- pytest-cov (coverage)

**Development:**
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

---

## 2. Data Architecture

### 2.1 Dataset Overview

**Fear & Greed Index:**
- **Period:** Feb 2018 - May 2025 (2,645 days)
- **Columns:** timestamp, value (0-100), classification (Extreme Fear, Fear, Neutral, Greed), date
- **Granularity:** Daily
- **Source:** Bitcoin market sentiment aggregator

**Hyperliquid Trading Data:**
- **Period:** Dec 2024 - Apr 2025 (~5 months)
- **Records:** 211,225 trades
- **Columns:** Account, Coin, Execution Price, Size Tokens, Size USD, Side, Timestamp IST, Start Position, Direction, Closed PnL, Transaction Hash, Order ID, Crossed, Fee, Trade ID, Timestamp
- **Granularity:** Trade-level (sub-second)

**Overlap Period:** Dec 2024 - Apr 2025 (ideal for correlation analysis)

### 2.2 Data Processing Pipeline

**Stage 1: Data Cleaning**
- Remove duplicates, handle missing values
- Validate data types and ranges
- Detect and handle outliers
- Standardize timestamp formats

**Stage 2: Timestamp Alignment**
- Convert all timestamps to UTC
- Align trade-level data with daily sentiment
- Create temporal windows (intraday, daily, weekly)

**Stage 3: Data Validation**
- Check for future leakage
- Verify temporal ordering
- Validate P&L calculations
- Ensure no look-ahead bias

**Stage 4: Feature Store Creation**
- Generate versioned feature datasets
- Store in parquet format for efficiency
- Maintain feature metadata registry

---

## 3. Feature Engineering Strategy

### 3.1 Feature Categories

#### A. Sentiment Features (sentiment.py)

**Raw Sentiment:**
- `sentiment_value` - Daily Fear/Greed index (0-100)
- `sentiment_class` - Categorical classification
- `is_extreme_fear` - Binary flag (value < 25)
- `is_extreme_greed` - Binary flag (value > 75)

**Rolling Sentiment:**
- `sentiment_ma_3d`, `sentiment_ma_7d`, `sentiment_ma_14d` - Moving averages
- `sentiment_std_7d` - Sentiment volatility
- `sentiment_momentum_7d` - Rate of change

**Sentiment Regime:**
- `days_in_fear` - Consecutive days in fear
- `days_in_greed` - Consecutive days in greed
- `sentiment_regime_change` - Binary flag for regime transitions
- `greed_persistence` - Duration of greed regime
- `fear_persistence` - Duration of fear regime

#### B. Trader Behavior Features (trader_behavior.py)

**Trading Activity:**
- `trade_count_1d`, `trade_count_7d` - Trading frequency
- `avg_trade_size_usd` - Average position size
- `trade_size_std` - Position sizing consistency
- `buy_sell_ratio` - Directional bias

**Position Management:**
- `avg_hold_duration` - Average time in position
- `position_turnover` - How often positions flip
- `max_position_size` - Largest position held
- `position_concentration` - Diversification metric

**Performance Metrics:**
- `win_rate_7d` - Percentage of profitable trades
- `avg_win_size`, `avg_loss_size` - Win/loss asymmetry
- `profit_factor` - Gross profit / gross loss
- `consecutive_wins`, `consecutive_losses` - Streaks

#### C. Risk Metrics (risk_metrics.py)

**Leverage Analysis:**
- `avg_leverage` - Average leverage used
- `max_leverage` - Peak leverage
- `leverage_std` - Leverage volatility
- `leverage_trend_7d` - Increasing/decreasing leverage

**Volatility Metrics:**
- `pnl_volatility_7d` - P&L standard deviation
- `returns_volatility` - Return volatility
- `sharpe_ratio_7d` - Risk-adjusted returns
- `sortino_ratio_7d` - Downside risk-adjusted returns

**Drawdown Metrics:**
- `max_drawdown_7d` - Largest peak-to-trough decline
- `current_drawdown` - Current drawdown from peak
- `drawdown_duration` - Days in drawdown
- `recovery_time` - Time to recover from drawdown

#### D. Behavioral Metrics (behavioral_metrics.py) ⭐

**Emotional Trading:**
- `revenge_trading_score` - Leverage increase after losses
- `overconfidence_ratio` - Position size increase after wins
- `panic_frequency` - Rate of exits during fear spikes
- `greed_chasing` - Entry rate during extreme greed

**Loss Aversion:**
- `loss_aversion_asymmetry` - Different behavior in gains vs losses
- `stop_loss_discipline` - Adherence to risk limits
- `profit_taking_speed` - How quickly profits are realized

**Recency Bias:**
- `recent_pnl_weight` - Over-weighting recent performance
- `momentum_chasing` - Following recent trends
- `mean_reversion_tendency` - Contrarian behavior

**Risk Escalation:**
- `risk_escalation_after_wins` - Leverage changes post-profit
- `risk_reduction_after_losses` - Leverage changes post-loss
- `emotional_leverage_expansion` - Irrational leverage increases

**Herding Behavior:**
- `correlation_with_market` - Following crowd
- `contrarian_score` - Going against the crowd
- `herd_panic_indicator` - Mass exit behavior

#### E. Temporal Features (temporal.py)

**Time-Based:**
- `hour_of_day`, `day_of_week`, `day_of_month`
- `is_weekend`, `is_month_end`
- `trading_session` - Asian/European/US session

**Lagged Features:**
- `sentiment_lag_1d`, `sentiment_lag_3d`, `sentiment_lag_7d`
- `pnl_lag_1d`, `pnl_lag_3d`, `pnl_lag_7d`
- `leverage_lag_1d`, `leverage_lag_3d`

### 3.2 Feature vs. Signal Distinction

**Features** = Raw engineered variables (e.g., `rolling_7d_sentiment_mean`)

**Signals** = Validated predictive hypotheses (e.g., `extreme_greed_reversal_signal`)

**Signal Generation Process:**
1. Engineer raw features
2. Test features for predictive power
3. Combine features into validated signals
4. Track signal performance over time

**Example Signals:**
- `extreme_greed_reversal_signal` - Extreme greed + high leverage → mean reversion
- `fear_capitulation_signal` - Extreme fear + mass exits → buying opportunity
- `revenge_trading_signal` - Loss + leverage increase → high risk of further loss
- `overconfidence_signal` - Win streak + position size increase → elevated risk

---

## 4. Market Regime Detection

### 4.1 Regime Types

**Fear/Greed Regimes (fear_regime.py):**
- Extreme Fear (0-25)
- Fear (25-45)
- Neutral (45-55)
- Greed (55-75)
- Extreme Greed (75-100)

**Volatility Regimes (volatility_regime.py):**
- Low volatility (< 25th percentile)
- Medium volatility (25th-75th percentile)
- High volatility (> 75th percentile)

**Liquidity Regimes (liquidity_regime.py):**
- High liquidity (high trade volume, many active traders)
- Medium liquidity
- Low liquidity (low trade volume, few active traders)
- Note: Uses trade volume and trader count as proxies since order book data unavailable

**Composite Regimes (composite_regime.py):**
- Combine multiple regime dimensions
- Example: "High Fear + High Volatility + Low Liquidity" = Crisis regime

### 4.2 Regime-Aware Modeling

**Why Regimes Matter:**
- Trader behavior changes across regimes
- Signals that work in greed may fail in fear
- Models need regime-specific parameters

**Implementation:**
- Train separate models per regime
- Use regime as a feature
- Ensemble models weighted by regime confidence

---

## 5. Research Hypotheses

### 5.1 Core Hypotheses

**H1: Sentiment-Performance Relationship**
- Extreme sentiment predicts mean reversion in trader profitability
- Traders are more profitable during fear (contrarian opportunity)
- Traders are less profitable during extreme greed (overconfidence)

**H2: Behavioral Patterns**
- Traders increase leverage after wins (overconfidence)
- Traders increase leverage after losses (revenge trading)
- Both behaviors lead to worse outcomes

**H3: Temporal Effects**
- Sentiment effects have lag (1-3 days)
- Immediate sentiment has weak predictive power
- Lagged sentiment captures delayed behavioral response

**H4: Regime Transitions**
- Regime transitions (fear → greed) are high-risk periods
- Traders are slow to adapt to new regimes
- Transition periods have higher prediction accuracy

**H5: Trader Segmentation**
- Experienced traders (consistent behavior) are less sentiment-driven
- Novice traders (erratic behavior) are highly sentiment-driven
- Segmentation improves model performance

### 5.2 Hypothesis Testing Framework

**Statistical Tests:**
- Correlation analysis (Pearson, Spearman)
- Granger causality tests (temporal causality)
- Mann-Whitney U tests (regime differences)
- Chi-square tests (categorical relationships)

**Validation:**
- Out-of-sample testing
- Walk-forward validation
- Regime-specific validation

---

## 6. Modeling Strategy

### 6.1 Prediction Targets

**Multi-Horizon Classification:**
- `is_profitable_intraday` - Profitable within same day
- `is_profitable_next_day` - Profitable next trading day
- `is_profitable_7d` - Profitable over next 7 days

**Why Multiple Horizons:**
- Different signals work at different timescales
- Sentiment effects may have delayed impact
- Demonstrates temporal reasoning

### 6.2 Model Architecture

**Baseline Model:**
- Logistic Regression
- Simple, interpretable
- Establishes performance floor

**Tree-Based Models:**
- Random Forest (ensemble robustness)
- XGBoost (gradient boosting)
- LightGBM (fast, efficient)

**Model Selection Criteria:**
- Precision/Recall trade-off (cost of false positives vs. false negatives)
- AUC-ROC (overall discrimination)
- Calibration (predicted probabilities match reality)
- Feature importance (interpretability)

### 6.3 Training Strategy

**Temporal Validation:**
- Walk-forward validation (no future leakage)
- Expanding window (train on all past data)
- Fixed window (train on recent N days)

**Class Imbalance Handling:**
- SMOTE (synthetic minority oversampling)
- Class weights
- Threshold tuning

**Hyperparameter Optimization:**
- Grid search for baseline
- Bayesian optimization for complex models
- Cross-validation within temporal folds

**MLflow Tracking:**
- Log all hyperparameters
- Track metrics across experiments
- Version models and features
- Compare models across horizons

---

## 7. Validation Framework

### 7.1 Leakage Prevention (leakage_checks.py)

**Common Leakage Sources:**
- Using future data in features
- Target leakage (features derived from target)
- Data snooping (testing on training data)

**Validation Checks:**
- Verify all features use only past data
- Check feature generation timestamps
- Validate train/test split integrity

### 7.2 Temporal Validation (temporal_validation.py)

**Walk-Forward Validation:**
```
Train: [Day 1 -------- Day 60]
Test:                          [Day 61 - Day 67]

Train: [Day 1 --------------- Day 67]
Test:                                 [Day 68 - Day 74]
```

**Expanding Window:**
- Train on all historical data
- Test on next period
- Mimics real-world deployment

**Fixed Window:**
- Train on recent N days
- Test on next period
- Adapts to regime changes faster

### 7.3 Robustness Testing (robustness_tests.py)

**Stability Tests:**
- Performance across different time periods
- Performance across different regimes
- Sensitivity to hyperparameters

**Stress Tests:**
- Performance during extreme events
- Performance with missing data
- Performance with noisy features

---

## 8. Backtesting Framework

### 8.1 Strategy Simulation (strategy_simulator.py)

**Trading Strategy:**
- Use model predictions to generate trade signals
- Long traders predicted to be profitable
- Short/avoid traders predicted to be unprofitable

**Execution Logic:**
- Entry: Model predicts profitability > threshold
- Exit: End of prediction horizon
- Position sizing: Based on prediction confidence

### 8.2 Transaction Costs (transaction_costs.py)

**Cost Components:**
- Trading fees (maker/taker fees)
- Slippage (price impact)
- Funding rates (perpetual futures)

**Realistic Assumptions:**
- Hyperliquid fee structure
- Market impact based on trade size
- Bid-ask spread costs

### 8.3 Performance Metrics (performance_metrics.py)

**Return Metrics:**
- Total return
- Annualized return
- Cumulative P&L

**Risk Metrics:**
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside risk-adjusted)
- Maximum drawdown
- Calmar ratio (return / max drawdown)

**Trade Metrics:**
- Win rate
- Profit factor
- Average win / average loss
- Expectancy

---

## 9. Visualization Strategy

### 9.1 EDA Visualizations (eda_plots.py)

**Sentiment Analysis:**
- Sentiment distribution over time
- Sentiment regime transitions
- Sentiment vs. Bitcoin price

**Trading Activity:**
- Trade volume over time
- Trade size distribution
- Leverage distribution

**Performance Analysis:**
- P&L distribution
- Win rate over time
- Drawdown analysis

### 9.2 Behavioral Visualizations (behavioral_plots.py)

**Emotional Trading:**
- Leverage changes after wins/losses
- Position sizing after wins/losses
- Trade frequency during fear/greed

**Regime Behavior:**
- Performance by sentiment regime
- Leverage by sentiment regime
- Trade characteristics by regime

### 9.3 Model Visualizations (model_plots.py)

**Model Performance:**
- ROC curves (all models, all horizons)
- Precision-Recall curves
- Calibration plots
- Confusion matrices

**Feature Importance:**
- SHAP values (feature contribution)
- Permutation importance
- Feature correlation heatmaps

**Temporal Analysis:**
- Performance over time
- Regime-specific performance
- Prediction confidence over time

---

## 10. Key Design Principles

### 10.1 Depth Over Decoration

**Focus Areas:**
1. **Temporal Behavioral Patterns** - How behavior evolves over time
2. **Regime Intelligence** - How models adapt to market conditions
3. **Human Irrationality** - Quantifying emotional decision-making

**Avoid:**
- Building folders without implementing depth
- Fake sophistication
- Over-engineering without insights

### 10.2 Reproducibility

**Requirements:**
- Fixed random seeds
- Versioned datasets
- Versioned features
- Logged experiments
- Documented decisions

### 10.3 Testability

**Critical Tests:**
- Data validation tests
- Feature generation tests
- Leakage detection tests
- Model performance tests

### 10.4 Modularity

**Benefits:**
- Easy to extend (add new features, models)
- Easy to debug (isolated components)
- Easy to test (unit tests per module)
- Easy to maintain (clear responsibilities)

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Set up project structure
2. Implement data loading & cleaning
3. Implement timestamp alignment
4. Create initial EDA notebook
5. Set up MLflow tracking

### Phase 2: Feature Engineering (Week 2)
1. Implement sentiment features
2. Implement trader behavior features
3. Implement risk metrics
4. Implement behavioral metrics ⭐
5. Implement temporal features
6. Create feature registry
7. Generate feature store v1

### Phase 3: Research & Analysis (Week 3)
1. Implement hypothesis tests
2. Implement correlation analysis
3. Implement lag analysis
4. Implement regime detection
5. Implement trader clustering
6. Create behavioral research notebook

### Phase 4: Signal Development (Week 4)
1. Validate features for predictive power
2. Create sentiment signals
3. Create behavioral signals
4. Create composite signals
5. Test signals across regimes

### Phase 5: Modeling (Week 5)
1. Implement baseline model
2. Implement tree-based models
3. Implement training pipeline
4. Implement temporal validation
5. Run experiments across horizons
6. Compare models in MLflow

### Phase 6: Validation & Backtesting (Week 6)
1. Implement leakage checks
2. Implement robustness tests
3. Implement backtesting framework
4. Run strategy simulations
5. Calculate performance metrics
6. Create backtesting notebook

### Phase 7: Documentation & Polish (Week 7)
1. Write research journal
2. Write methodology document
3. Write findings document
4. Create final visualizations
5. Write comprehensive README
6. Prepare for presentation

---

## 12. Success Criteria

### 12.1 Technical Success

**Model Performance:**
- AUC-ROC > 0.60 (better than random)
- Precision > 0.55 (more right than wrong)
- Calibration error < 0.10 (probabilities are accurate)

**Backtesting Performance:**
- Positive Sharpe ratio (> 0.5)
- Maximum drawdown < 30%
- Win rate > 50%

**Code Quality:**
- 80%+ test coverage
- All critical functions tested
- No leakage detected
- Clean linting (black, flake8)

### 12.2 Research Success

**Hypothesis Validation:**
- At least 3 hypotheses validated
- Clear behavioral patterns identified
- Regime effects demonstrated

**Novel Insights:**
- Unique behavioral metrics created
- Temporal patterns discovered
- Regime-specific strategies identified

### 12.3 Portfolio Success

**Interview Readiness:**
- Can explain every design decision
- Can discuss trade-offs
- Can extend to new directions
- Can defend methodology

**GitHub Quality:**
- Professional README
- Clear documentation
- Publication-quality visualizations
- Research journal demonstrates rigor

---

## 13. Advanced Extensions (Future Work)

### 13.1 Model Extensions

**Sequence Models:**
- LSTM for temporal dependencies
- Transformer for attention mechanisms
- GRU for efficient sequence modeling

**Graph-Based Analysis:**
- Trader network analysis
- Influence propagation
- Community detection

**Anomaly Detection:**
- Detect unusual trading patterns
- Identify market manipulation
- Flag high-risk behavior

### 13.2 System Extensions

**Real-Time Pipeline:**
- Stream sentiment data
- Real-time feature generation
- Online model updates

**Reinforcement Learning:**
- Learn optimal trading strategies
- Adaptive position sizing
- Dynamic risk management

**Multi-Asset Extension:**
- Extend to other cryptocurrencies
- Cross-asset sentiment effects
- Portfolio optimization

---

## 14. Research Journal Structure

**docs/research_journal.md** will track:

**Date: YYYY-MM-DD**
- **Hypothesis:** What I'm testing
- **Approach:** How I'm testing it
- **Results:** What I found
- **Interpretation:** What it means
- **Next Steps:** Where to go from here

**Failed Experiments:**
- What didn't work and why
- Lessons learned
- Alternative approaches

**Surprising Findings:**
- Unexpected patterns
- Counterintuitive results
- New hypotheses generated

**Open Questions:**
- Unresolved issues
- Areas needing more investigation
- Potential improvements

---

## 15. Interview Talking Points

### 15.1 Technical Depth

**Feature Engineering:**
- "I separated features from signals because features are raw variables, while signals are validated predictive hypotheses. This mirrors how institutional quant systems evolve."

**Temporal Validation:**
- "I used walk-forward validation to prevent future leakage, which is the #1 failure mode in financial ML. Standard k-fold cross-validation would give artificially high performance."

**Behavioral Metrics:**
- "I created behavioral metrics like 'revenge trading score' and 'emotional leverage expansion' based on behavioral finance theory. These capture human irrationality that traditional features miss."

### 15.2 Design Decisions

**Why Multiple Horizons:**
- "Different signals work at different timescales. Sentiment effects may be immediate (intraday) or delayed (7-day). Testing multiple horizons reveals temporal decay of signals."

**Why Regime Detection:**
- "Trader behavior changes across market regimes. A signal that works during fear may fail during greed. Regime-aware modeling improves robustness."

**Why Backtesting:**
- "Prediction accuracy doesn't equal profitability. Transaction costs, slippage, and market impact can destroy a seemingly good model. Backtesting validates economic viability."

### 15.3 Behavioral Finance Insights

**Loss Aversion:**
- "Traders hold losing positions too long (hoping for recovery) and exit winning positions too early (fear of giving back gains). This asymmetry is quantifiable."

**Overconfidence:**
- "After winning streaks, traders increase position sizes and leverage, believing they have 'hot hands'. This leads to elevated risk and eventual losses."

**Herding:**
- "During extreme sentiment, traders follow the crowd. Contrarian strategies (buying fear, selling greed) can be profitable if timed correctly."

---

## 16. Conclusion

This design creates an institutional-grade quantitative research system that:

1. **Demonstrates technical depth** - Advanced feature engineering, temporal validation, regime detection
2. **Shows research rigor** - Hypothesis-driven, documented experiments, failed attempts tracked
3. **Proves production thinking** - Modular architecture, experiment tracking, backtesting
4. **Highlights unique insights** - Behavioral finance focus, signal validation, regime intelligence

**The project's edge comes from:**
- Thoughtful hypotheses (not just throwing features at models)
- Strong temporal reasoning (preventing leakage, understanding lag effects)
- Behavioral insights (quantifying human irrationality)
- Regime intelligence (adapting to market conditions)

**NOT from:**
- Folder count
- Library choices
- Model complexity
- Visualization quantity

**Next Step:** Create detailed implementation plan with specific tasks, code structure, and execution order.
