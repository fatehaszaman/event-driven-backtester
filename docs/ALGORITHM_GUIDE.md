# Algorithm guide

These cards cover the moving-average signal calculations, not the total cost
of data loading, portfolio accounting, execution, or performance reporting.
Inputs are already available through the data handler.

## Simple moving-average crossover

Implementation: [`SMACrossoverStrategy.calculate_signals`](../strategies/moving_average.py).
Window retrieval: [`get_latest_bars_values`](../backtest/data.py).

```text
# SMA Crossover / Window Scan + Position Flag
# Input: S symbols, W long-window bars per symbol, short window <= W
# Output: zero or one LONG/EXIT event per symbol per invocation
# Time: O(S*W) per invocation; O(B*S*W) over B market updates
# Memory: O(W) temporary window + O(S) persistent position flags
# Output: up to O(S) new queued events per invocation

FOR each symbol:
    bars = COPY up to W latest adjusted closes
    IF fewer than W bars: CONTINUE
    short_mean = MEAN(short-window suffix)
    long_mean = MEAN(long-window suffix)
    IF short_mean > long_mean AND not marked bought:
        ENQUEUE LONG; mark bought
    ELSE IF short_mean < long_mean AND marked bought:
        ENQUEUE EXIT; clear bought flag
```

The implementation recomputes means and materializes window values. It is not
an O(1)-per-symbol rolling-sum implementation. The data handler also retains bar
history; that O(BS) storage is separate from the strategy's O(W+S) working state.

## Exponential moving-average crossover

Implementation: [`EMACrossoverStrategy._ema` and `calculate_signals`](../strategies/moving_average.py).

```text
# EMA Crossover / Recompute Finite Windows
# Input: the same symbol/window contract as SMA
# Output: LONG/EXIT events and updated position flags
# Time: O(S*W) per invocation
# Memory: O(W+S) with retrieved windows and position flags, excluding event queue

FUNCTION ema(values, period):
    alpha = 2 / (period + 1)
    value = first supplied observation
    FOR each remaining observation:
        value = alpha * observation + (1-alpha) * value
    RETURN value

FOR each symbol with W available bars:
    short_ema = ema(short-window suffix, short period)
    long_ema = ema(long-window bars, long period)
    APPLY the same LONG/EXIT conditions as SMA
```

Here EMA restarts from the first value of each finite window on every call.
It is not a recursively maintained full-history EMA.

Watch: equality emits no event; warm-up emits no event; windows should be
positive and short <= long. The bought flag changes when a signal is emitted,
not when an order is confirmed filled, so signal state and executed holdings
are different concepts. These cards do not claim profitability or eliminate
the separate execution assumptions described elsewhere in the repository.
