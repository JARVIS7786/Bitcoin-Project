"""Backtesting modules for PrimeTRADE."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TransactionCostModel:
    """
    Model transaction costs for realistic backtesting.
    """

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_bps: float = 5.0,
    ):
        """
        Initialize TransactionCostModel.

        Args:
            maker_fee: Maker fee (0.02%)
            taker_fee: Taker fee (0.05%)
            slippage_bps: Slippage in basis points
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps

    def calculate_costs(
        self,
        position_size: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate transaction costs.

        Args:
            position_size: Position size in USD
            is_maker: Whether order is maker (limit) or taker (market)

        Returns:
            Total transaction cost
        """
        fee = self.maker_fee if is_maker else self.taker_fee
        slippage = position_size * (self.slippage_bps / 10000)
        total_cost = position_size * fee + slippage

        return total_cost


class StrategySimulator:
    """
    Simulate trading strategy with realistic costs.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        cost_model: Optional[TransactionCostModel] = None,
    ):
        """
        Initialize StrategySimulator.

        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()

    def simulate(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        position_size_column: str = 'position_size',
    ) -> pd.DataFrame:
        """
        Simulate strategy performance.

        Args:
            df: DataFrame with trading data
            predictions: Model predictions (1 = trade, 0 = no trade)
            position_size_column: Position size column

        Returns:
            DataFrame with simulation results
        """
        df = df.copy()
        df['prediction'] = predictions

        # Calculate costs
        df['transaction_cost'] = df.apply(
            lambda row: self.cost_model.calculate_costs(row[position_size_column])
            if row['prediction'] == 1 else 0,
            axis=1
        )

        # Calculate net PnL
        df['net_pnl'] = df.apply(
            lambda row: row['pnl'] - row['transaction_cost']
            if row['prediction'] == 1 else 0,
            axis=1
        )

        # Calculate cumulative returns
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        df['portfolio_value'] = self.initial_capital + df['cumulative_pnl']

        logger.info(f"Simulation complete. Final portfolio value: ${df['portfolio_value'].iloc[-1]:.2f}")

        return df

    def calculate_metrics(self, simulation_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            simulation_df: Simulation results

        Returns:
            Dictionary of performance metrics
        """
        total_return = (simulation_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        total_trades = (simulation_df['prediction'] == 1).sum()
        winning_trades = ((simulation_df['prediction'] == 1) & (simulation_df['net_pnl'] > 0)).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Sharpe ratio (simplified)
        returns = simulation_df['net_pnl']
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0

        metrics = {
            'total_return': total_return,
            'total_trades': int(total_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'final_value': simulation_df['portfolio_value'].iloc[-1],
        }

        logger.info(f"Performance metrics: {metrics}")

        return metrics
