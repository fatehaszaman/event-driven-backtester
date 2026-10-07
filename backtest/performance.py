import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Standard deviations below this are treated as zero volatility. An exact
# `== 0` check misses floating-point residue (a constant 1% series has
# std ~1.8e-18), which used to report a Sharpe near 8.7e16.
ZERO_VOL_TOLERANCE = 1e-12


def create_sharpe_ratio(returns, periods=252):
    """Annualized Sharpe ratio. Use periods=252 for daily, 52 for weekly."""
    if not returns.std() >= ZERO_VOL_TOLERANCE:  # also catches NaN std
        return 0.0
    return np.sqrt(periods) * returns.mean() / returns.std()


def create_drawdowns(equity_curve):
    """Returns a drawdown Series and the max drawdown scalar."""
    hwm = equity_curve.cummax()
    drawdown = (equity_curve - hwm) / hwm
    return drawdown, drawdown.min()


def create_cagr(equity_curve, periods=252):
    """Compound annual growth rate from a daily equity curve."""
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods
    return total_return ** (1.0 / years) - 1.0


def plot_tearsheet(holdings_df, initial_capital, title="Strategy Tearsheet"):
    equity = holdings_df["total"]
    returns = equity.pct_change().dropna()
    drawdown, max_dd = create_drawdowns(equity)
    sharpe = create_sharpe_ratio(returns)
    cagr = create_cagr(equity)
    total_return = equity.iloc[-1] / initial_capital - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(equity.index, equity.values, color="steelblue", linewidth=1.2)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title(
        f"Total Return: {total_return*100:.1f}%  |  "
        f"CAGR: {cagr*100:.1f}%  |  "
        f"Sharpe: {sharpe:.2f}  |  "
        f"Max DD: {max_dd*100:.1f}%"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="tomato", alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
