# AAI6640 Team 5 - Applied Deep Learning

Team project: a GARCH-LSTM-t model for US retail gasoline prices (AAA daily and EIA weekly), scored against Kalshi gas-price markets.

## Repository layout

| Path | What it holds |
|---|---|
| `Midterm_Checkpoint.ipynb` | **Midterm checkpoint submission**: all four parts merged into one notebook (Sections 1-7), runs top to bottom |
| `Part2_Baselines.ipynb` | Part 2: baseline models |
| `Part3_Optimization_experiments_revised.ipynb` | Part 3: optimization experiments |
| `Part4_Diagnostics_revised.ipynb` | Part 4: diagnostics and backtest |
| `gaslab/` | Shared Python package (data, models, pipeline, evaluation, experiments, service) |
| `data/` | Input data: `fred.csv`, `kalshi_markets.csv`, `kalshi_open_book.csv` |
| `model/` | Trained daily and weekly weights (3 seeds each), scalers and `meta.json` |
| `results/` | Saved results from earlier parts (`.pkl`, `.json`) |

## Front end (web demo, calendar and dashboard)

| Path | What it is |
|---|---|
| `docs/index.html` | Static web demo: tomorrow's forecast, any-threshold probabilities, a calendar of past forecasts vs actual prices, strike ladder, scorecard |
| `docs/forecast.json` | Tomorrow's forecast (written by `export_demo.py` / `tools/update_daily.py`) |
| `docs/history.json` | One row per past day: predicted price, 80%/95% range, actual price. Held-out and out-of-sample days only, plus live forecasts that are locked in before their price is published |
| `docs/scorecard.json` | Out-of-sample scores from Part 4. Update it when Part 4 is re-run |
| `app.py` | Streamlit dashboard for the team (`pip install -r requirements-app.txt`, then `streamlit run app.py`) |
| `tools/build_history.py` | Rebuilds the held-out and out-of-sample rows of `history.json` (retrains small daily/weekly ensembles, a few minutes) |
| `tools/update_daily.py` | Fetches FRED + AAA data, scores yesterday, locks in tomorrow's forecast, rewrites `docs/*.json` |

Publish: Settings > Pages > Deploy from a branch > `main` / `docs`.

### Automatic updates (GitHub Actions)

| Workflow | When | What |
|---|---|---|
| `daily-update.yml` | every morning, plus manual run | `tools/update_daily.py`, then commits `data/` and `docs/` |
| `weekly-retrain.yml` | Manual run only (the model is pinned to the midterm version) | refits the model on all data (`gaslab.service.train()`), then refreshes the demo. |
| `add-data.yml` | when an "Add a gas price data point" issue is opened | appends the row to `data/manual_aaa.csv` if the author is a project member; otherwise leaves it for review |

The demo's "Add a data point" button opens that issue form. Notebooks read `data/`, so rerunning them after an automatic update gives slightly different numbers than a submitted run; pin a commit when you need fixed results.

## Setup

Use Python 3.12 (TensorFlow has no wheels for newer versions).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open the notebooks in VS Code and select the `.venv` kernel. Run the parts in order: 2, then 3, then 4. Later parts read files written by earlier ones.

## Working together

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: pull before you start, work on a branch, open a pull request, and never commit `.venv`.

## Team

- Orville Rockwell
- Yi-Ting (Jerry)
- Wenjun Chen
- Yixin Tian



## Juroom: model pinning and data refresh
The app (docs/) is named Juroom ("five" in Wolof, for Team 5). It runs the midterm-project model in `model/` (trained through 2026-11-03) and stays on it until the final version is added; the weekly retrain workflow is manual-only. The daily workflow refreshes AAA, FRED and Kalshi data (`tools/update_daily.py`), grades yesterday's forecast and freezes tomorrow's. Chats are saved in the visitor's browser (localStorage).
