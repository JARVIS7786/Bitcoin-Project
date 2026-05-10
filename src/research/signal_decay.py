"""Signal decay analysis for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SignalDecayAnalyzer:
    """
    Analyze signal decay and calculate half-life of features.

    Measures how quickly a signal's predictive power degrades over time,
    helping determine optimal feature windows and signal freshness.
    """

    def __init__(self):
        """Initialize SignalDecayAnalyzer."""
        pass

    @staticmethod
    def exponential_decay(t: np.ndarray, initial: float, decay_rate: float) -> np.ndarray:
        """
        Exponential decay function: y = initial * exp(-decay_rate * t)

        Args:
            t: Time array
            initial: Initial value at t=0
            decay_rate: Decay rate (lambda)

        Returns:
            Decayed values
        """
        return initial * np.exp(-decay_rate * t)

    def calculate_correlation_decay(
        self,
        df: pd.DataFrame,
        signal_column: str,
        target_column: str,
        max_lag: int = 24,
        min_periods: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate how correlation decays with increasing time lag.

        Args:
            df: DataFrame with time series data (sorted by time)
            signal_column: Signal/feature column
            target_column: Target variable column
            max_lag: Maximum lag to test (in time periods)
            min_periods: Minimum periods required for correlation calculation

        Returns:
            DataFrame with columns: lag, correlation, abs_correlation

        Examples:
            >>> analyzer = SignalDecayAnalyzer()
            >>> df = pd.DataFrame({
            ...     'sentiment': [30, 40, 50, 60, 70],
            ...     'pnl': [100, 150, 200, 180, 220]
            ... })
            >>> decay = analyzer.calculate_correlation_decay(df, 'sentiment', 'pnl', max_lag=3)
        """
        if signal_column not in df.columns:
            raise ValueError(f"Signal column '{signal_column}' not found")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        correlations = []

        for lag in range(0, max_lag + 1):
            if lag == 0:
                # No lag - direct correlation
                signal = df[signal_column]
                target = df[target_column]
            else:
                # Shift signal by lag periods
                signal = df[signal_column].shift(lag)
                target = df[target_column]

            # Calculate correlation, dropping NaN values
            valid_data = pd.DataFrame({'signal': signal, 'target': target}).dropna()

            if len(valid_data) >= min_periods:
                corr = valid_data['signal'].corr(valid_data['target'])
                correlations.append({
                    'lag': lag,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'n_samples': len(valid_data)
                })
            else:
                logger.warning(f"Insufficient data at lag {lag}: {len(valid_data)} < {min_periods}")

        decay_df = pd.DataFrame(correlations)
        logger.info(f"Calculated correlation decay for {len(decay_df)} lags")

        return decay_df

    def fit_exponential_decay(
        self,
        decay_df: pd.DataFrame,
        use_abs: bool = True,
    ) -> Dict[str, float]:
        """
        Fit exponential decay model to correlation decay data.

        Args:
            decay_df: DataFrame from calculate_correlation_decay
            use_abs: Use absolute correlation values (default: True)

        Returns:
            Dictionary with fitted parameters:
                - initial: Initial correlation value
                - decay_rate: Decay rate (lambda)
                - half_life: Half-life in time periods
                - r_squared: Goodness of fit

        Examples:
            >>> analyzer = SignalDecayAnalyzer()
            >>> decay_df = analyzer.calculate_correlation_decay(df, 'sentiment', 'pnl')
            >>> params = analyzer.fit_exponential_decay(decay_df)
            >>> print(f"Half-life: {params['half_life']:.2f} periods")
        """
        if len(decay_df) < 3:
            raise ValueError("Need at least 3 data points to fit exponential decay")

        t = decay_df['lag'].values
        y = decay_df['abs_correlation'].values if use_abs else decay_df['correlation'].values

        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y)
        t = t[valid_mask]
        y = y[valid_mask]

        if len(t) < 3:
            raise ValueError("Insufficient valid data points after filtering")

        try:
            # Fit exponential decay: y = initial * exp(-decay_rate * t)
            # Initial guess: initial = y[0], decay_rate = 0.1
            popt, pcov = curve_fit(
                self.exponential_decay,
                t,
                y,
                p0=[y[0], 0.1],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10000
            )

            initial, decay_rate = popt

            # Calculate half-life: t_half = ln(2) / decay_rate
            half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf

            # Calculate R-squared
            y_pred = self.exponential_decay(t, initial, decay_rate)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            logger.info(
                f"Fitted exponential decay: initial={initial:.4f}, "
                f"decay_rate={decay_rate:.4f}, half_life={half_life:.2f}, "
                f"R²={r_squared:.4f}"
            )

            return {
                'initial': initial,
                'decay_rate': decay_rate,
                'half_life': half_life,
                'r_squared': r_squared,
            }

        except Exception as e:
            logger.error(f"Failed to fit exponential decay: {e}")
            raise

    def calculate_signal_strength(
        self,
        decay_df: pd.DataFrame,
        threshold: float = 0.1,
    ) -> Dict[str, any]:
        """
        Calculate signal strength metrics.

        Args:
            decay_df: DataFrame from calculate_correlation_decay
            threshold: Correlation threshold for "significant" signal

        Returns:
            Dictionary with signal strength metrics:
                - max_correlation: Maximum absolute correlation
                - max_lag: Lag at maximum correlation
                - effective_window: Number of lags above threshold
                - avg_correlation: Average absolute correlation

        Examples:
            >>> analyzer = SignalDecayAnalyzer()
            >>> decay_df = analyzer.calculate_correlation_decay(df, 'sentiment', 'pnl')
            >>> strength = analyzer.calculate_signal_strength(decay_df, threshold=0.1)
        """
        if len(decay_df) == 0:
            raise ValueError("Empty decay DataFrame")

        max_idx = decay_df['abs_correlation'].idxmax()
        max_correlation = decay_df.loc[max_idx, 'abs_correlation']
        max_lag = decay_df.loc[max_idx, 'lag']

        # Count lags above threshold
        effective_window = (decay_df['abs_correlation'] >= threshold).sum()

        # Average correlation
        avg_correlation = decay_df['abs_correlation'].mean()

        logger.info(
            f"Signal strength: max_corr={max_correlation:.4f} at lag={max_lag}, "
            f"effective_window={effective_window} periods"
        )

        return {
            'max_correlation': max_correlation,
            'max_lag': int(max_lag),
            'effective_window': int(effective_window),
            'avg_correlation': avg_correlation,
        }

    def analyze_multiple_signals(
        self,
        df: pd.DataFrame,
        signal_columns: List[str],
        target_column: str,
        max_lag: int = 24,
    ) -> pd.DataFrame:
        """
        Analyze decay for multiple signals.

        Args:
            df: DataFrame with time series data
            signal_columns: List of signal column names
            target_column: Target variable column
            max_lag: Maximum lag to test

        Returns:
            DataFrame with decay analysis for each signal

        Examples:
            >>> analyzer = SignalDecayAnalyzer()
            >>> signals = ['sentiment', 'volatility', 'volume']
            >>> results = analyzer.analyze_multiple_signals(df, signals, 'pnl')
        """
        results = []

        for signal in signal_columns:
            try:
                # Calculate correlation decay
                decay_df = self.calculate_correlation_decay(
                    df, signal, target_column, max_lag
                )

                # Fit exponential decay
                decay_params = self.fit_exponential_decay(decay_df)

                # Calculate signal strength
                strength = self.calculate_signal_strength(decay_df)

                results.append({
                    'signal': signal,
                    'target': target_column,
                    'half_life': decay_params['half_life'],
                    'decay_rate': decay_params['decay_rate'],
                    'initial_correlation': decay_params['initial'],
                    'r_squared': decay_params['r_squared'],
                    'max_correlation': strength['max_correlation'],
                    'max_lag': strength['max_lag'],
                    'effective_window': strength['effective_window'],
                })

            except Exception as e:
                logger.warning(f"Failed to analyze signal '{signal}': {e}")
                continue

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Sort by half-life (signals with longer half-life are more stable)
            results_df = results_df.sort_values('half_life', ascending=False)

        logger.info(f"Analyzed {len(results_df)} signals successfully")

        return results_df

    def get_optimal_window(
        self,
        decay_params: Dict[str, float],
        min_correlation: float = 0.1,
    ) -> int:
        """
        Calculate optimal feature window based on decay parameters.

        Args:
            decay_params: Parameters from fit_exponential_decay
            min_correlation: Minimum acceptable correlation

        Returns:
            Optimal window size in time periods

        Examples:
            >>> analyzer = SignalDecayAnalyzer()
            >>> decay_params = {'initial': 0.5, 'decay_rate': 0.1}
            >>> window = analyzer.get_optimal_window(decay_params, min_correlation=0.1)
        """
        initial = decay_params['initial']
        decay_rate = decay_params['decay_rate']

        if decay_rate <= 0:
            logger.warning("Invalid decay rate, returning default window")
            return 24

        # Solve: min_correlation = initial * exp(-decay_rate * t)
        # t = -ln(min_correlation / initial) / decay_rate
        if initial <= min_correlation:
            return 0

        optimal_window = -np.log(min_correlation / initial) / decay_rate
        optimal_window = int(np.ceil(optimal_window))

        logger.info(f"Optimal window: {optimal_window} periods (min_corr={min_correlation})")

        return optimal_window
