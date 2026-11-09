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

