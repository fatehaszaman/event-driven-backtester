# Lessons

This file records real mistakes in this backtester, what caused them, and what changed. Open defects are tracked separately in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## 1. Zero-volatility Sharpe guard that never triggered

**What happened.** `create_sharpe_ratio` guarded against division by zero with `if returns.std() == 0`. A constant 1% return series has a standard deviation of `1.83e-18`, so the guard was bypassed and the function returned a Sharpe of 8.68e16 instead of 0.0. The same pattern appeared in `NaivePortfolio.output_summary_stats`.

**Why it happened.** Exact floating-point equality was used as a guard. I also predicted a different failure mode before running the test, and the real one was worse.

**What I changed.** Both call sites now use a shared `1e-12` near-zero volatility tolerance.

**How I prevent it now.** Numerical guards use tolerances, not exact equality, and each one has a regression test.

**History.** [`0d9b7da`](https://github.com/fatehaszaman/event-driven-backtester/commit/0d9b7da) (fix and test), [`3d76912`](https://github.com/fatehaszaman/event-driven-backtester/commit/3d76912) (README updated).

## 2. "No look-ahead" claim was broader than the code

**What happened.** The project was described as having no look-ahead bias. That was true of the data handler, which streams bars strictly in order, but not of execution: a signal computed from a bar's close was filled at that same close, a price that was not tradable when it was observed.

**Why it happened.** A property of one component was stated as a property of the whole system.

**What I changed.** The README and profile first narrowed the claim to ordered bar delivery and documented the same-bar execution limit as [KNOWN_ISSUES.md #2](KNOWN_ISSUES.md). The limit is now fixed: each order is held until the next bar and filled at that bar's open, scaled to the adjusted basis, plus slippage. Within each bar, pending fills are applied before the strategy computes new signals, and the repeated final market event (KNOWN_ISSUES.md #1) cannot fill an order. Re-running the four strategies on AAPL from 2018 to 2023 moved returns in both directions, so the README now reports before-and-after figures rather than calling the earlier results optimistic.

**How I prevent it now.** Each defect found by the test suite has a test that pins the current behavior, so a fix or a regression is visible. Claims about the system are checked against every component they cover, and the fix is tested for event order, split and dividend scaling, and the final-bar edge case.

**History.** Profile [`f3926a4`](https://github.com/fatehaszaman/fatehaszaman/commit/f3926a4) and [`5c18078`](https://github.com/fatehaszaman/fatehaszaman/commit/5c18078), limit first documented in [`e7e2260`](https://github.com/fatehaszaman/event-driven-backtester/commit/e7e2260), fixed in [`3602be1`](https://github.com/fatehaszaman/event-driven-backtester/commit/3602be1), profile updated in [`7863970`](https://github.com/fatehaszaman/fatehaszaman/commit/7863970).
