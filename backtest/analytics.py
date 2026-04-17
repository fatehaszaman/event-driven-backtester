"""
analytics.py
------------
Performance analytics for backtest results.

Computes standard strategy performance metrics from an equity curve:
Sharpe ratio, max drawdown, CAGR, win rate, and cost breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trade_history: pd.DataFrame,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute performance metrics from an equity curve and trade history.

    Parameters
    ----------
    equity_curve : pd.DataFrame
        Output of Portfolio.equity_curve(). Must have 'equity' and 'timestamp' columns.
    trade_history : pd.DataFrame
        Output of Portfolio.trade_history().
    risk_free_rate : float
        Annualized risk-free rate. Default 5%.
    periods_per_year : int
        Number of bars per year. Default 252 (daily).

    Returns
    -------
    dict of performance metrics.
    """
    if equity_curve.empty:
        return {}

    equity = equity_curve["equity"].values
    returns = pd.Series(equity).pct_change().dropna()

    # ── Returns ───────────────────────────────────────────────────────────────
    total_return = (equity[-1] / equity[0]) - 1
    n_periods = len(equity)
    cagr = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # ── Risk ──────────────────────────────────────────────────────────────────
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
              if returns.std() > 0 else 0.0)

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    sortino = (excess_returns.mean() / downside.std() * np.sqrt(periods_per_year)
               if len(downside) > 0 and downside.std() > 0 else 0.0)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cummax = pd.Series(equity).cummax()
    drawdown = (pd.Series(equity) - cummax) / cummax
    max_drawdown = drawdown.min()

    # Drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_lengths = in_drawdown[in_drawdown].groupby(dd_groups[in_drawdown]).count()
    max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # ── Trade stats ───────────────────────────────────────────────────────────
    trade_metrics = {}
    if not trade_history.empty and "price" in trade_history.columns:
        buys  = trade_history[trade_history["side"] == "BUY"]
        sells = trade_history[trade_history["side"] == "SELL"]
        n_trades = len(trade_history)

        # Pair buys and sells for round-trip P&L
        min_pairs = min(len(buys), len(sells))
        if min_pairs > 0:
            buy_prices  = buys["price"].values[:min_pairs]
            sell_prices = sells["price"].values[:min_pairs]
            round_trip_pnl = sell_prices - buy_prices
            wins = (round_trip_pnl > 0).sum()
            win_rate = wins / min_pairs
            avg_win  = round_trip_pnl[round_trip_pnl > 0].mean() if wins > 0 else 0
            avg_loss = round_trip_pnl[round_trip_pnl <= 0].mean() if (min_pairs - wins) > 0 else 0
            profit_factor = (
                abs(round_trip_pnl[round_trip_pnl > 0].sum()) /
                abs(round_trip_pnl[round_trip_pnl <= 0].sum())
                if round_trip_pnl[round_trip_pnl <= 0].sum() != 0 else float("inf")
            )
        else:
            win_rate = profit_factor = avg_win = avg_loss = 0.0

        trade_metrics = {
            "n_trades": n_trades,
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
        }

    final_snap = equity_curve.iloc[-1]

    return {
        # Returns
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        # Risk
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        # Drawdown
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "max_dd_duration_bars": max_dd_duration,
        # Costs
        "total_commission": round(final_snap.get("total_commission", 0), 2),
        "total_slippage": round(final_snap.get("total_slippage", 0), 2),
        **trade_metrics,
    }


def print_report(metrics: dict, label: str = "Backtest Results") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    sections = {
        "Returns": ["total_return_pct", "cagr_pct"],
        "Risk": ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "ann_volatility_pct"],
        "Drawdown": ["max_drawdown_pct", "max_dd_duration_bars"],
        "Trades": ["n_trades", "win_rate_pct", "profit_factor", "avg_win", "avg_loss"],
        "Costs": ["total_commission", "total_slippage"],
    }
    for section, keys in sections.items():
        print(f"\n  {section}")
        for k in keys:
            if k in metrics:
                print(f"    {k:<28} {metrics[k]}")
    print(f"{'='*50}\n")
