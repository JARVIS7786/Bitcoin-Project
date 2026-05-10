"""Feature stability validation for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureStabilityValidator:
    """
    Validate feature stability across time periods using PSI and KS tests.

    Ensures features maintain consistent distributions to prevent model drift.
    """

    def __init__(self, n_bins: int = 10, psi_threshold: float = 0.2, ks_threshold: float = 0.05):
        """
        Initialize FeatureStabilityValidator.

        Args:
            n_bins: Number of bins for PSI calculation
            psi_threshold: PSI threshold for stability (>0.2 indicates instability)
            ks_threshold: KS test p-value threshold
        """
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def calculate_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures distribution shift between two datasets.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (action required)

        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution
            bins: Custom bin edges (auto-calculated if None)

        Returns:
            PSI value

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> expected = pd.Series([1, 2, 3, 4, 5] * 20)
            >>> actual = pd.Series([1, 2, 3, 4, 5] * 20)
            >>> psi = validator.calculate_psi(expected, actual)
            >>> psi < 0.1  # Stable
            True
        """
        # Remove NaN values
        expected = expected.dropna()
        actual = actual.dropna()

        if len(expected) == 0 or len(actual) == 0:
            raise ValueError("Cannot calculate PSI with empty data")

        # Create bins based on expected distribution
        if bins is None:
            _, bins = pd.qcut(expected, q=self.n_bins, retbins=True, duplicates='drop')

        # Bin both distributions
        expected_binned = pd.cut(expected, bins=bins, include_lowest=True, duplicates='drop')
        actual_binned = pd.cut(actual, bins=bins, include_lowest=True, duplicates='drop')

        # Calculate proportions
        expected_props = expected_binned.value_counts(normalize=True, sort=False)
        actual_props = actual_binned.value_counts(normalize=True, sort=False)

        # Align indices
        expected_props, actual_props = expected_props.align(actual_props, fill_value=0.0001)

        # Avoid log(0) by adding small epsilon
        expected_props = expected_props.replace(0, 0.0001)
        actual_props = actual_props.replace(0, 0.0001)

        # Calculate PSI
        psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))

        return float(psi)

    def ks_test(
        self,
        sample1: pd.Series,
        sample2: pd.Series,
    ) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution equality.

        Args:
            sample1: First sample
            sample2: Second sample

        Returns:
            Dictionary with 'statistic' and 'p_value'

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> s1 = pd.Series(np.random.normal(0, 1, 100))
            >>> s2 = pd.Series(np.random.normal(0, 1, 100))
            >>> result = validator.ks_test(s1, s2)
            >>> result['p_value'] > 0.05  # Same distribution
            True
        """
        sample1 = sample1.dropna()
        sample2 = sample2.dropna()

        if len(sample1) == 0 or len(sample2) == 0:
            raise ValueError("Cannot perform KS test with empty data")

        statistic, p_value = stats.ks_2samp(sample1, sample2)

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
        }

    def validate_feature(
        self,
        baseline: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> Dict[str, any]:
        """
        Validate single feature stability.

        Args:
            baseline: Baseline (training) distribution
            current: Current (validation/test) distribution
            feature_name: Name of feature

        Returns:
            Dictionary with validation results

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> baseline = pd.Series(np.random.normal(0, 1, 1000))
            >>> current = pd.Series(np.random.normal(0, 1, 1000))
            >>> result = validator.validate_feature(baseline, current, 'feature1')
            >>> result['is_stable']
            True
        """
        # Calculate PSI
        psi = self.calculate_psi(baseline, current)

        # Perform KS test
        ks_result = self.ks_test(baseline, current)

        # Determine stability
        psi_stable = psi < self.psi_threshold
        ks_stable = ks_result['p_value'] > self.ks_threshold
        is_stable = psi_stable and ks_stable

        result = {
            'feature': feature_name,
            'psi': psi,
            'psi_stable': psi_stable,
            'ks_statistic': ks_result['statistic'],
            'ks_p_value': ks_result['p_value'],
            'ks_stable': ks_stable,
            'is_stable': is_stable,
        }

        if not is_stable:
            logger.warning(
                f"Feature '{feature_name}' is UNSTABLE: "
                f"PSI={psi:.4f} (threshold={self.psi_threshold}), "
                f"KS p-value={ks_result['p_value']:.4f} (threshold={self.ks_threshold})"
            )
        else:
            logger.info(f"Feature '{feature_name}' is stable")

        return result

    def validate_features(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Validate multiple features.

        Args:
            baseline_df: Baseline DataFrame
            current_df: Current DataFrame
            feature_columns: List of feature columns (None = all numeric)

        Returns:
            DataFrame with validation results for each feature

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> baseline = pd.DataFrame({'f1': np.random.randn(100), 'f2': np.random.randn(100)})
            >>> current = pd.DataFrame({'f1': np.random.randn(100), 'f2': np.random.randn(100)})
            >>> results = validator.validate_features(baseline, current)
        """
        if feature_columns is None:
            feature_columns = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for feature in feature_columns:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature '{feature}' not found in both DataFrames")
                continue

            try:
                result = self.validate_feature(
                    baseline_df[feature],
                    current_df[feature],
                    feature
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate feature '{feature}': {e}")
                continue

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Sort by PSI (most unstable first)
            results_df = results_df.sort_values('psi', ascending=False)

            stable_count = results_df['is_stable'].sum()
            total_count = len(results_df)
            logger.info(
                f"Feature stability: {stable_count}/{total_count} features stable "
                f"({stable_count/total_count*100:.1f}%)"
            )

        return results_df

    def get_unstable_features(
        self,
        validation_results: pd.DataFrame,
    ) -> List[str]:
        """
        Get list of unstable features.

        Args:
            validation_results: Results from validate_features

        Returns:
            List of unstable feature names

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> results = validator.validate_features(baseline, current)
            >>> unstable = validator.get_unstable_features(results)
        """
        if len(validation_results) == 0:
            return []

        unstable = validation_results[~validation_results['is_stable']]['feature'].tolist()
        return unstable

    def summarize_stability(
        self,
        validation_results: pd.DataFrame,
    ) -> str:
        """
        Generate human-readable stability summary.

        Args:
            validation_results: Results from validate_features

        Returns:
            Formatted summary string

        Examples:
            >>> validator = FeatureStabilityValidator()
            >>> results = validator.validate_features(baseline, current)
            >>> print(validator.summarize_stability(results))
        """
        if len(validation_results) == 0:
            return "No features validated"

        summary = []
        summary.append("Feature Stability Report")
        summary.append("=" * 60)

        stable_count = validation_results['is_stable'].sum()
        total_count = len(validation_results)
        summary.append(f"Stable features: {stable_count}/{total_count} ({stable_count/total_count*100:.1f}%)")
        summary.append("")

        # PSI summary
        summary.append("PSI Distribution:")
        summary.append(f"  Mean: {validation_results['psi'].mean():.4f}")
        summary.append(f"  Max: {validation_results['psi'].max():.4f}")
        summary.append(f"  Features with PSI > {self.psi_threshold}: {(validation_results['psi'] > self.psi_threshold).sum()}")
        summary.append("")

        # Unstable features
        unstable = self.get_unstable_features(validation_results)
        if unstable:
            summary.append(f"Unstable features ({len(unstable)}):")
            for feature in unstable[:10]:  # Show top 10
                row = validation_results[validation_results['feature'] == feature].iloc[0]
                summary.append(f"  - {feature}: PSI={row['psi']:.4f}, KS p-value={row['ks_p_value']:.4f}")
            if len(unstable) > 10:
                summary.append(f"  ... and {len(unstable) - 10} more")

        return "\n".join(summary)
