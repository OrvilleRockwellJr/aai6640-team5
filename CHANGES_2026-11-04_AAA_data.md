# Data update: extended AAA daily series (2026-11-04)

**Part:** 2 (Baselines and deep baseline) · **Owner:** Wenjun Chen

## Summary

The daily AAA national average series was extended from **367 days** (Kalshi settlement values,
Sep 2023 – Oct 2026) to **3,865 days** (Jun 2015 – Nov 2026). The extra days come from Wayback Machine
snapshots of gasprices.aaa.com. As a result, the daily LSTMs now train on **3,181** complete 7-day windows
instead of 83. The daily test window was also given a fixed end date, so that adding new AAA days no longer
changes the test set.

Weekly results are unchanged, because the weekly model uses EIA data.

## Data source and method

1. **Homepage snapshots.** About 2,850 daily Wayback snapshots of `gasprices.aaa.com/` (2016–2026) were downloaded.
   From each one, the script reads the national regular price in the *Current Avg.* and *Yesterday Avg.* rows.
   The date comes from the page's own "Price as of m/d/yy" stamp, not from the snapshot timestamp: snapshots are
   stored in UTC and are often taken on the previous US evening.
2. **Other pages on the domain.** Every page on gasprices.aaa.com (news releases, state averages, archive pages)
   shows the same "National Average $x.xxx / Price as of" header. Snapshots of these pages filled 232 more days.
3. **Week-, month- and year-ago rows.** The homepage table also has *Week Ago*, *Month Ago* and *Year Ago* rows.
   Checked against values read directly from the page, they correspond to D−7 (99.5% exact match), the same day
   of the previous month (98.7%) and the same day of the previous year (87%; mismatches are usually 0.1¢ and
   look like later revisions by AAA). They filled the remaining gaps, in that order of priority, and extended
   the series back to June 2015.

**Coverage.** From 2016-06-01 to 2026-11-03, 3,727 of 3,808 days have a value (97.9%). The 81 missing days
all fall in 2016–2020, and no gap is longer than 2 days. Before June 2016 there are 138 sparse days.

| Source (`data/manual_aaa_sources.csv`) | Days |
|---|---|
| `observed` (read directly from a page) | 3,422 |
| `week_ago` | 116 |
| `month_ago` | 108 |
| `year_ago` | 212 |

**Validation against Kalshi.** On every day that both sources cover, the values are identical once aligned on
AAA's "Price as of" date. This check also found that the **Kalshi settlement dates for Sep 2023 – Apr 2024 are
shifted by one day**: the value Kalshi records for day D is AAA's price for D−1. `load_aaa_daily()` gives
`manual_aaa.csv` precedence over Kalshi, so the price series is now correct for that period. The Kalshi contract
labels themselves (`daily_contracts`) are unchanged; the daily validation and test periods (2026) are not affected.

## Code changes

* `gaslab/experiments.py`: the daily test window `TE` changed from `("2026-08-20", "2027-12-31")` to
  `("2026-08-20", "2026-10-19")`, the last settled Kalshi contract. With an open end date, the extended AAA
  data silently added 15 days (Oct 20 – Nov 3) to the daily NLL/CRPS test set, while the contract metrics stayed on 61 days.
* `gaslab/VERSION` and `VERSION` in the notebook's Setup cell changed from `2026-10-20` to `2026-11-04`, so an
  older local `gaslab/` is backed up and replaced when the Setup cell runs.
* The notebook's Setup cell now embeds `manual_aaa.csv` and `manual_aaa_sources.csv` in `DATA`, so the
  notebook runs on the extended data in a fresh environment such as Colab.

No model code, features or hyperparameters were changed.

## Effect on results (daily test, Aug 20 – Oct 19 2026, 61 days)

| Model | NLL before | NLL after | CRPS before | CRPS after | Brier before | Brier after |
|---|---|---|---|---|---|---|
| GARCH-LSTM-t (x3) | 0.192 | **−0.120** | 0.180 | **0.132** | 0.0853 | **0.0631** |
| Student-t LSTM, no GARCH (x3) | 0.329 | **−0.173** | 0.194 | **0.126** | 0.0914 | **0.0611** |
| Gaussian LSTM (x3) | 0.458 | 0.699 | 0.190 | 0.131 | 0.0910 | 0.0636 |
| Daily ECM | 0.502 | 0.593 | 0.186 | 0.148 | 0.0880 | 0.0712 |
| GARCH(1,1)-t alone | 0.409 | 0.267 | 0.205 | 0.200 | 0.0970 | 0.0946 |
| Momentum (not trained) | 0.493 | 0.493 | 0.178 | 0.178 | 0.0859 | 0.0859 |
| Persistence (not trained) | 0.709 | 0.709 | 0.214 | 0.214 | 0.1023 | 0.1023 |
| Kalshi market (1 h after open) | | | | | 0.0489 | 0.0489 |

Persistence and momentum score exactly as before, which confirms that the test targets did not change.

**Main changes in conclusions**

* **Daily data limit removed.** The deep models now clearly beat momentum and the ECM on CRPS and Brier.
* **Significance.** Diebold–Mariano tests with Holm correction show that GARCH-LSTM-t is now significantly better
  than persistence, momentum and GARCH alone on CRPS and per-day Brier, on both validation and test. It is
  also better than the ECM-based baselines on validation.
* **The GARCH input no longer helps** on daily data: the Student-t LSTM without it is as good or slightly better.
* **The Student-t head matters only through NLL.** The Gaussian LSTM has the same CRPS and Brier.
  The Stage 1 result "Student-t beats Gaussian on daily test NLL" (Holm p = 0.024) no longer holds.
* **Daily intervals are now too narrow.** For GARCH-LSTM-t, 80% coverage is 0.72 on test, against 0.93
  (too wide) before. Scale calibration on daily validation is the natural next step.
* **Revision 5.** Seed variance dropped (per-seed validation NLL sd 0.085 → 0.036) and the regularization
  configurations differ little. The original daily configuration is kept for Parts 3–4.
* **Still behind the market.** The models remain behind the Kalshi market (Brier 0.063 vs 0.049).

The notebook's Markdown readings for the daily Stage 1 results and Revisions 2–5 were rewritten with these
numbers, and a row was added to the revision log.

## Files to update in the repository

**Required**

| File | Status | Change |
|---|---|---|
| `Part2_Baselines.ipynb` | replaced | Rerun on the extended data. Its Setup cell embeds the new `gaslab` version and the two AAA files, and the daily readings and revision log are updated. It replaces the previous Part 2 notebook. |
| `data/manual_aaa.csv` | new | AAA national daily average, 3,865 days (`date,aaa`) |
| `data/manual_aaa_sources.csv` | new | Source of each value (`observed`, `week_ago`, `month_ago`, `year_ago`) |
| `gaslab/experiments.py` | modified | Line 10: the daily test window `TE` now ends on 2026-10-19 |
| `gaslab/VERSION` | modified | `2026-10-20` → `2026-11-04` |

**Recommended**

| File | Status | Purpose |
|---|---|---|
| `CHANGES_2026-11-04_AAA_data.md` | new | This change note |
| `tools/aaa_wayback/scrape.py`, `fill.py`, `ago.py` | new | Scripts that rebuild `manual_aaa.csv` from the Wayback Machine |

**Results files (tracked in the repository)**

| File | Status | Note |
|---|---|---|
| `results/part1.pkl`, `results/part2.pkl`, `results/part2_revisions.pkl` | modified | Rebuilt by the rerun on the extended data (data fingerprint `858e9b7a18fc`) |
| `results/part3.pkl`, `results/stack_coef.json`, `results/weekly_cfg.json` | unchanged | `part3.pkl` was built with the old fingerprint, so Part 3 rebuilds it automatically on its next run |
| `model/*` | unchanged | Saved weights for the demo service, trained on the old daily data; retrain them in Part 4 if the demo should use the extended data |

**Do not commit:** `__pycache__/` and `.ipynb_checkpoints/`.

**Merged notebook cell.** The branch's "Daily raw data: price and volatility over time" cell (commit `ef61634`)
was merged into `Part2_Baselines.ipynb` at the same place. It was adapted to the extended data: it now plots the
AAA series from June 2016 instead of only the unbroken run since March 2026, its title and text name both
sources (Kalshi + Wayback), and its summary table is by year instead of by month.

**Other parts.** If the Part 1, 3 and 4 notebooks embed their own copy of `gaslab` and the data in a Setup cell,
those copies still have the old `TE`, the old `VERSION` and no `manual_aaa.csv`. Update them the same way:

1. Change `TE` in the embedded `experiments.py`.
2. Bump `VERSION` to `2026-11-04`.
3. Add the two AAA files to `DATA`.

Otherwise the parts run on different data and their data fingerprints will not match.

## Reproducing the data

The scripts are in `tools/aaa_wayback/`:

* `scrape.py <workdir> --fetch` downloads the homepage snapshots listed in `<workdir>/cdx.txt` and parses
  them into `aaa_wayback_raw.csv`. The CDX list comes from
  `http://web.archive.org/cdx/search/cdx?url=gasprices.aaa.com/&filter=statuscode:200&collapse=timestamp:8&fl=timestamp,original`.
* `fill.py <workdir> <missing.txt>` fills missing dates from snapshots of other pages on the domain
  (`cdx_all.txt`, the same query with `matchType=domain`).
* `ago.py <workdir>` extracts the week-, month- and year-ago rows from the downloaded homepage snapshots.

Wayback rate-limits heavy use. Three parallel requests with retry backoff took about 1.5 hours for the full set.
