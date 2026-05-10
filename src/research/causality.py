"""Causality analysis for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import grangercausalitytests
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GrangerCausalityAnalyzer:
    """
    Perform Granger causality tests to establish temporal precedence.

    Tests whether past values of X help predict future values of Y,
    establishing causal (not just correlational) relationships.
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        """
        Initialize GrangerCausalityAnalyzer.

        Args:
            max_lag: Maximum number of lags to test
            significance_level: P-value threshold for significance (default: 0.05)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test_granger_causality(
        self,
        df: pd.DataFrame,
        cause_column: str,
        effect_column: str,
        max_lag: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Test if cause_column Granger-causes effect_column.

        Args:
            df: DataFrame with time series data (must be sorted by time)
            cause_column: Column name of potential cause variable
            effect_column: Column name of effect variable
            max_lag: Maximum lag to test (uses self.max_lag if None)

        Returns:
            Dictionary with test results:
                - 'is_causal': bool indicating if causality detected
                - 'best_lag': optimal lag with lowest p-value
                - 'p_values': dict of p-values for each lag
                - 'f_statistics': dict of F-statistics for each lag

        Raises:
            ValueError: If columns not found or data insufficient

        Examples:
            >>> analyzer = GrangerCausalityAnalyzer(max_lag=5)
            >>> df = pd.DataFrame({
            ...     'sentiment': [30, 40, 50, 60, 70, 80],
            ...     'leverage': [2, 3, 4, 5, 6, 7]
            ... })
            >>> result = analyzer.test_granger_causality(df, 'sentiment', 'leverage')
            >>> print(result['is_causal'])
        """
        if cause_column not in df.columns:
            raise ValueError(f"Cause column '{cause_column}' not found")
        if effect_column not in df.columns:
            raise ValueError(f"Effect column '{effect_column}' not found")

        if max_lag is None:
            max_lag = self.max_lag

        # Prepare data: [effect, cause] order required by statsmodels
        data = df[[effect_column, cause_column]].dropna()

        if len(data) < max_lag + 10:
            raise ValueError(
                f"Insufficient data: need at least {max_lag + 10} rows, got {len(data)}"
            )

        logger.info(
            f"Testing Granger causality: {cause_column} → {effect_column} (max_lag={max_lag})"
        )

        # Run Granger causality test
        try:
            test_results = grangercausalitytests(
                data, maxlag=max_lag, verbose=False
            )
        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            raise

        # Extract p-values and F-statistics for each lag
        p_values = {}
        f_statistics = {}

        for lag in range(1, max_lag + 1):
            # Use F-test (ssr_ftest) as primary test
            ssr_ftest = test_results[lag][0]['ssr_ftest']
            p_values[lag] = ssr_ftest[1]  # p-value
            f_statistics[lag] = ssr_ftest[0]  # F-statistic

        # Find best lag (lowest p-value)
        best_lag = min(p_values, key=p_values.get)
        best_p_value = p_values[best_lag]

        # Determine if causal relationship exists
        is_causal = best_p_value < self.significance_level

        result = {
            'is_causal': is_causal,
            'best_lag': best_lag,
            'best_p_value': best_p_value,
            'p_values': p_values,
            'f_statistics': f_statistics,
            'cause': cause_column,
            'effect': effect_column,
        }

        if is_causal:
            logger.info(
                f"✓ Causality detected: {cause_column} → {effect_column} "
                f"(lag={best_lag}, p={best_p_value:.4f})"
            )
        else:
            logger.info(
                f"✗ No causality: {cause_column} → {effect_column} "
                f"(best p={best_p_value:.4f})"
            )

        return result

    def test_bidirectional_causality(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        max_lag: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Test for bidirectional Granger causality between two variables.

        Args:
            df: DataFrame with time series data
            var1: First variable name
            var2: Second variable name
            max_lag: Maximum lag to test

        Returns:
            Dictionary with bidirectional test results:
                - 'var1_causes_var2': result of var1 → var2 test
                - 'var2_causes_var1': result of var2 → var1 test
                - 'bidirectional': bool indicating if both directions causal

        Examples:
            >>> analyzer = GrangerCausalityAnalyzer()
            >>> result = analyzer.test_bidirectional_causality(df, 'sentiment', 'leverage')
            >>> if result['bidirectional']:
            ...     print("Feedback loop detected!")
        """
        logger.info(f"Testing bidirectional causality: {var1} ↔ {var2}")

        # Test var1 → var2
        result_1_to_2 = self.test_granger_causality(df, var1, var2, max_lag)

        # Test var2 → var1
        result_2_to_1 = self.test_granger_causality(df, var2, var1, max_lag)

        bidirectional = result_1_to_2['is_causal'] and result_2_to_1['is_causal']

        if bidirectional:
            logger.warning(f"⚠ Bidirectional causality detected: {var1} ↔ {var2}")

        return {
            f'{var1}_causes_{var2}': result_1_to_2,
            f'{var2}_causes_{var1}': result_2_to_1,
            'bidirectional': bidirectional,
        }

    def test_multiple_causes(
        self,
        df: pd.DataFrame,
        cause_columns: List[str],
        effect_column: str,
        max_lag: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Test multiple potential causes against a single effect.

        Args:
            df: DataFrame with time series data
            cause_columns: List of potential cause variable names
            effect_column: Effect variable name
            max_lag: Maximum lag to test

        Returns:
            DataFrame with test results for each cause, sorted by p-value

        Examples:
            >>> analyzer = GrangerCausalityAnalyzer()
            >>> causes = ['sentiment', 'volatility', 'volume']
            >>> results = analyzer.test_multiple_causes(df, causes, 'leverage')
            >>> print(results[results['is_causal']])
        """
        results = []

        for cause in cause_columns:
            try:
                result = self.test_granger_causality(df, cause, effect_column, max_lag)
                results.append({
                    'cause': cause,
                    'effect': effect_column,
                    'is_causal': result['is_causal'],
                    'best_lag': result['best_lag'],
                    'p_value': result['best_p_value'],
                    'f_statistic': result['f_statistics'][result['best_lag']],
                })
            except Exception as e:
                logger.warning(f"Failed to test {cause} → {effect_column}: {e}")
                continue

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values('p_value')

        logger.info(
            f"Tested {len(cause_columns)} causes for {effect_column}. "
            f"Found {results_df['is_causal'].sum()} causal relationships."
        )

        return results_df

    def get_causal_graph(
        self,
        df: pd.DataFrame,
        variables: List[str],
        max_lag: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build a causal graph showing all pairwise relationships.

        Args:
            df: DataFrame with time series data
            variables: List of variable names to test
            max_lag: Maximum lag to test

        Returns:
            DataFrame adjacency matrix where entry (i,j) indicates if
            variable i Granger-causes variable j

        Examples:
            >>> analyzer = GrangerCausalityAnalyzer()
            >>> vars = ['sentiment', 'leverage', 'pnl']
            >>> graph = analyzer.get_causal_graph(df, vars)
            >>> print(graph)
        """
        n = len(variables)
        causal_matrix = pd.DataFrame(
            False, index=variables, columns=variables
        )

        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if i == j:
                    continue  # Skip self-causation

                try:
                    result = self.test_granger_causality(df, cause, effect, max_lag)
                    causal_matrix.loc[cause, effect] = result['is_causal']
                except Exception as e:
                    logger.warning(f"Failed to test {cause} → {effect}: {e}")
                    continue

        logger.info(f"Built causal graph for {n} variables")

        return causal_matrix

    def summarize_results(self, results: Dict[str, any]) -> str:
        """
        Generate human-readable summary of causality test results.

        Args:
            results: Results dictionary from test_granger_causality

        Returns:
            Formatted summary string

        Examples:
            >>> analyzer = GrangerCausalityAnalyzer()
            >>> result = analyzer.test_granger_causality(df, 'sentiment', 'leverage')
            >>> print(analyzer.summarize_results(result))
        """
        summary = []
        summary.append(f"Granger Causality Test: {results['cause']} → {results['effect']}")
        summary.append(f"{'='*60}")
        summary.append(f"Causal relationship: {'YES' if results['is_causal'] else 'NO'}")
        summary.append(f"Best lag: {results['best_lag']}")
        summary.append(f"P-value at best lag: {results['best_p_value']:.6f}")
        summary.append(f"Significance level: {self.significance_level}")
        summary.append(f"\nP-values by lag:")

        for lag, p_val in sorted(results['p_values'].items()):
            sig_marker = "*" if p_val < self.significance_level else " "
            summary.append(f"  Lag {lag}: {p_val:.6f} {sig_marker}")

        return "\n".join(summary)
