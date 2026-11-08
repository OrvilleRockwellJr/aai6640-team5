# Out-of-sample check, Nov 4-7 2026 (run 2026-11-08)

Models trained on AAA data through Nov 3 only. For each target day D the forecast uses AAA through D-1 and
wholesale through D-2 where available. Scored on the Kalshi daily strike ladder against the 1-hour-after-open
bid/ask midpoint (same definition as Part 2). 4 days, 72 contracts: a smoke test, not a significance test.

| Setup | Mean abs error | Brier (model) | Brier (model+market stack) | Brier (Kalshi) |
|---|---|---|---|---|
| Repo data (wholesale ends Oct 14, 19-22 days stale) | 1.27 c | 0.0996 | n/a | 0.0223 |
| + FRED wholesale/crude through Nov 3 | 0.83 c | 0.0653 | 0.0290 | 0.0223 |
| + 2-parameter calibration fitted on Aug 20-Oct 19 (scale x1.193, bias -0.0149%) | 0.77 c | 0.0600 | 0.0280 | 0.0223 |
| "Tomorrow = today" | 2.81 c | | | |

Actual AAA: Nov 3 4.1493, Nov 4 4.1239, Nov 5 4.1003, Nov 6 4.0688, Nov 7 4.0371.
The calibration was NOT tuned on these days. Stack coefficients are results/stack_coef.json (fitted Aug 20-Sep 19).

Files: `train_scratch.py` (retrain with fresh FRED rows), `calibrate.py` (fit scale/bias), `backtest.py`
(score Nov 4-7), `kalshi_nov_book.csv` (settled strikes + 1h-after-open bid/ask), `fred_fresh/` (FRED pulls through Nov 3).
The scripts expect a scratch copy of data/ at /home/claude/scratch/data with the fresh FRED rows appended to fred.csv.
