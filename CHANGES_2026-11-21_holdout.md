# Changes 2026-11-21: Holdout split and lock (branch `part3-holdout-split`)

## Why
The professor said our test window (Aug 20 – Oct 19 2026) was checked and
revised against many times, so it is no longer an independent test. This
branch adds a new final test period that nobody has scored yet, and a lock
that records every time it is used.

## New data splits (daily model)
| Split | Dates | Use |
|---|---|---|
| Train | to Jun 30 2026 | Training |
| VA (validation) | Jul 1 – Aug 19 2026 | Early stopping, tuning |
| TE (development) | Aug 20 – Oct 19 2026 | Tuning and calibration are now allowed here. It is no longer the final test. |
| HO (holdout) | Oct 20 – Nov 20 2026, minus Nov 4–7 | Final test. Scored once, after we freeze the setup. |

Nov 4–7 is removed because `tools/backtest_nov2026/` already scored those days.

## What changed
- `gaslab/experiments.py`: added `HO` and `holdout_index()`. It raises an
  error unless `results/FROZEN.json` exists, and writes every call to
  `results/holdout_access.log`.
- `gaslab/pipeline.py`: the test mask in `split_masks` now stops at Oct 19,
  so holdout days cannot be used by accident. No scored result changes.
- `gaslab/data.py`: `daily_contracts()` also reads `data/kalshi_holdout_book.csv`.
- `tools/fetch_kalshi_history.py` (new): fills in missing settlements
  (Oct 21–30 were missing, Oct 20 was never updated after it settled) and
  gets the bid/ask one hour after open from Kalshi hourly candlesticks. On
  October data these match `kalshi_open_book.csv` exactly.
- `data/kalshi_markets.csv`: 17 Oct 20 rows updated to settled, 170 rows
  added for Oct 21–30. No other rows changed.
- `data/kalshi_holdout_book.csv` (new): bid/ask one hour after open, Oct 20 – Nov 20.
- `.gitignore`: `results/holdout_access.log` is now tracked.
- `README.md`: added the holdout rules.

## Rules from now on
1. All holdout scoring must go through `holdout_index()`. Do not filter by
   `HO` dates directly.
2. Score the holdout only with pipeline models (trained to Jun 30). Do not
   use `model/`, because it was retrained on data through Nov 3.
3. Do not create `results/FROZEN.json` until the team agrees the Stage 3
   setup is final.
4. All Stage 3 tuning, model selection and calibration use VA and TE only.
5. Before scoring, check that FRED has closes through D-2 for every holdout
   day. If a close is missing, the data code silently treats that day's
   wholesale and crude change as zero.

## Checks
- Holdout: all 28 settlement days and all 500 strikes have a settled result
  and a bid/ask. No ticker appears twice in `kalshi_markets.csv`.
- FRED covers 27 of 28 holdout days. Nov 20 needs the Nov 18 close, but
  FRED's last close is Nov 17. Re-check before scoring.
- On the midterm notebook's own data, all VA and TE results match the
  midterm exactly (weekly test n = 92, GARCH-LSTM-t NLL 2.0315; daily test
  NLL -0.1218, accuracy 0.9151, Brier 0.0629, 1107 contracts, 61 events).
- On today's repo data, daily test NLL is -0.1210 and the weekly test has
  97 weeks (NLL 2.0538). This is not caused by this branch. The midterm
  data had EIA weekly retail only through Oct 12, so the Oct 12 week had no
  target and no weekly forecast, and TE days Oct 14–19 reused the Oct 5
  forecast. With the newer EIA weeks, those days use the Oct 12 forecast.
  **Stage 3 comparisons must re-run all models on the same data version.**

## Known issue (not fixed here)
The weekly model only makes a forecast for weeks whose target is already
known. So the daily model's weekly input depends on whether the next EIA
print has arrived, and the latest days fall back to an older forecast. The
inputs never use future values, but the result changes as data arrives.
This should be fixed before the holdout is scored.

## Needs a team decision
- **Daily bid/ask collection:** `update_daily.py` saves only last price
  and volume, so bid/ask stopped on Oct 19. The bot also only pulls the
  last 10 days, which is why Oct 21–30 was missing. Can we add the 9 a.m.
  book to the daily update?
- **`EVALUATION_PLAN.md`:** we need to agree on the model list, main metric
  (suggested: ATM Brier per settlement day), tests, the holdout end date,
  and a fixed data version before we freeze.
- **Late strikes:** some strikes on Nov 11–13 were added after the 8 a.m.
  open, so their bid/ask is from their first full hour, not 9 a.m. We need
  to decide whether to drop them from the market comparison.
- **`docs/scorecard.json`:** the numbers are typed by hand and look like an
  older Part 4 run (model accuracy 0.886 vs 0.914 in the midterm notebook).
  The public page shows old results.
- **`docs/history.json`:** the website shows daily forecasts next to actual
  prices for some holdout days. No metrics were computed from it and no
  model was changed, but we should say this in the report.
