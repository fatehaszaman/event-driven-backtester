# Known issues

Found by writing the test suite against this code. Every item below has a test
that currently **passes by asserting the defective behaviour**, so the bug
cannot drift unnoticed and fixing it will visibly break its test — at which
point the test and the entry should both be deleted.

Listing these is deliberate. A backtester whose failure modes are unknown is
more dangerous than one whose failure modes are written down.

| # | Severity | Issue |
|---|---|---|
| 1 | medium | Extra market event at end of data duplicates the final bar |
| 7 | medium | CAGR returns nan, or worse than −100%, on blown-up equity |
| 9 | medium | Strategy position flag tracks signals, not fills |
| 3 | medium | Unpriceable orders are silently discarded |
| 4 | low | Commission cap overrides the stated minimum |
| 10 | low | EMA windows seeded on inconsistent histories |
| 5 | low | Zero-strength signals create zero-quantity orders |
| 6 | low | Sharpe assumes a zero risk-free rate |

---

### 1. Extra market event at end of data — MEDIUM

`CSVDataHandler.update_bars` emits a `MarketEvent` unconditionally, including
on the final call when the iterator is exhausted and no bar is appended. Ten
bars produce eleven market events.

The portfolio therefore marks the last bar twice, and 60 bars yield 61 holdings
rows. The duplicate row contributes a spurious 0% return, which lowers the
standard deviation of the return series and so **inflates the reported Sharpe
ratio**. Small on long histories, material on short ones.

**Fix:** only emit the event when at least one symbol advanced.
Tests: `test_emits_a_market_event_even_when_data_is_exhausted`,
`test_equity_curve_has_one_extra_row_from_the_duplicate_final_bar`

### 7. CAGR on non-positive equity — MEDIUM

`create_cagr` computes `total_return ** (1/years) - 1`.

- Terminal equity of −10 over exactly one year returns **−1.1**, a loss of
  110%, which is not a possible growth rate.
- The same case over a fractional number of years takes a fractional power of a
  negative base and returns **nan**, which then renders in the tearsheet title
  as `CAGR: nan%`.
- There is no minimum-length check: a 1% gain across two bars is annualised to
  **+250%** without comment.

**Fix:** return `-1.0` (total loss) for non-positive terminal equity, and
refuse to annualise fewer than roughly one period of data.
Tests: `test_cagr_reports_worse_than_total_loss_when_equity_goes_negative`,
`test_cagr_returns_nan_when_equity_goes_negative_over_a_fractional_year`,
`test_cagr_annualises_two_bars_without_complaint`

### 9. Strategy flag tracks signals rather than fills — MEDIUM

`SMACrossoverStrategy.bought[symbol]` is set to `True` at the moment a signal is
*emitted*. Nothing confirms a fill.

Combined with issue 3, this desyncs permanently: if the order is dropped, the
strategy believes it holds a position, so it never emits `LONG` again, and
`NaivePortfolio._generate_order` will not act on a later `EXIT` because the
position is actually zero. The strategy goes quiet for the rest of the run with
no error.

**Fix:** the portfolio owns position state; the strategy should read it rather
than keep a parallel copy.
Test: `test_internal_flag_tracks_signals_not_fills`

### 3. Unpriceable orders are silently discarded — MEDIUM

`SimulatedExecutionHandler` wraps the bar lookup in `execute_order` and the
price lookup in `on_market` in a bare `except Exception`. A missing bar, an
absent `open`, `close` or `adj close` column, or a typo in a symbol produces no
fill, no warning, and no counter. The backtest
completes and reports results that omit trades the strategy intended.

**Fix:** catch the specific lookup failure, count it, and surface the count in
the summary output.
Test: `test_order_is_silently_dropped_when_no_bar_exists`

### 4. Commission cap overrides the stated minimum — LOW

`FillEvent._ib_commission` ends with `min(cost, 0.005 * fill_cost)`, which
applies a 0.5%-of-notional cap *after* the 1.30 floor. A 100-dollar trade is
charged 0.50, below the documented minimum. Real schedules apply both
constraints, not one overriding the other.

Test: `test_commission_floor_is_breached_on_small_notional`

### 10. EMA windows seeded on inconsistent histories — LOW

In `EMACrossoverStrategy.calculate_signals`, the short EMA is computed on
`bars[-short_window:]` while the long EMA is computed on all `bars`. Each is
seeded from the first element of a *different* slice, so the two series start
from different initial values and the crossover compares inconsistent bases.
The effect decays with more data but is largest right after warm-up, which is
exactly when the first signals fire.

Test: `test_ema_windows_are_seeded_on_different_histories`

### 5. Zero-strength signals create zero-quantity orders — LOW

`qty = int(100 * signal.strength)` with `strength=0.0` produces a 0-share
order that reaches the execution handler and generates a zero-cost fill.
Harmless now; becomes a real problem as soon as a fixed per-order fee exists.

Test: `test_zero_strength_produces_a_zero_quantity_order`

### 6. Sharpe assumes a zero risk-free rate — LOW

`create_sharpe_ratio` has no `risk_free` parameter. This is a documented
assumption rather than a bug, but in a 4–5% rate environment it overstates the
ratio and the README should say so.

Test: `test_sharpe_assumes_a_zero_risk_free_rate`

---

## Fixed

### 2. Signals filled at the same bar's close (fixed)

The engine drained the whole event queue within one bar, so a signal computed
from a bar's close was filled at that same close, a price that was not
tradable when it was observed. `SimulatedExecutionHandler` now holds each order
until the next bar and fills it at that bar's open, scaled to the adjusted
basis, plus slippage. Orders with no next bar are not filled and are logged.
Re-running the strategies changed returns in both directions; see the README.
Tests: `test_order_waits_for_the_next_bar_and_fills_at_its_open`,
`test_signal_is_not_filled_at_the_close_it_was_computed_from`,
`test_open_is_put_on_the_adjusted_basis`,
`test_order_on_the_last_bar_is_not_filled_and_is_logged`,
`test_fill_after_a_split_uses_the_fill_bar_ratio`,
`test_fill_on_an_ex_dividend_bar_uses_the_fill_bar_ratio`,
`test_on_bar_t_plus_1_the_fill_comes_before_new_signals`,
`test_duplicate_final_market_event_cannot_fill_a_last_bar_order`.

### 8. Zero-volatility Sharpe guard never fired (fixed)

`create_sharpe_ratio` compared `returns.std() == 0`, so a constant 1% return
series (std ~1.8e-18) reported a Sharpe of ~8.7e16. It now treats any std below
`1e-12` as zero volatility, and `NaivePortfolio.output_summary_stats` calls the
same function instead of keeping its own guard. Tests:
`test_sharpe_is_zero_on_near_constant_returns`,
`test_portfolio_summary_stats_reuses_the_guarded_sharpe`.
