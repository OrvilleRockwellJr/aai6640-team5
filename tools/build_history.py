"""Build docs/history.json: the day-by-day record behind the GasLab calendar.

Every row is a forecast made BEFORE its actual price was known, so the calendar never shows a model grading
its own training data:

  heldout  Aug 20 - Nov 3 2026. Daily model fitted on data through Jun 30 (early stopping to Aug 19), the same
           split Part 4 reports. The weekly drift input comes from a weekly ensemble fitted on the Part 2 split.
  oos      Nov 4 - Nov 7 2026. The shipped model/ (trained through Nov 3) with only the data that existed the
           day before each target (same procedure as tools/backtest_nov2026/backtest.py).
  live     Added by tools/update_daily.py from Nov 9 on: tomorrow's forecast is frozen when it is made and the
           actual is filled in later.

    python tools/build_history.py          # rewrites the heldout + oos rows, keeps any live rows
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gaslab import data as D, pipeline as P, service  # noqa: E402

OUT = ROOT / "docs" / "history.json"
SEEDS = 3
# Actuals for Nov 4-7 are the AAA national averages recorded in tools/backtest_nov2026/RESULTS.md
OOS_ACTUALS = {"2026-11-03": 4.1493, "2026-11-04": 4.1239, "2026-11-05": 4.1003, "2026-11-06": 4.0688, "2026-11-07": 4.0371}


def row_from(date, prev, mu, sc, nu, actual, source):
    """One calendar row. Prices are AAA national average $/gal; the model's mu/scale are in log-percent."""
    dist = stats.t(df=nu, loc=mu, scale=sc) if np.isfinite(nu) else stats.norm(loc=mu, scale=sc)
    px = lambda pct: float(prev * np.exp(pct / 100))
    r = dict(date=str(pd.Timestamp(date).date()), prev=round(float(prev), 4), pred=round(px(mu), 4),
             lo80=round(px(dist.ppf(.10)), 4), hi80=round(px(dist.ppf(.90)), 4),
             lo95=round(px(dist.ppf(.025)), 4), hi95=round(px(dist.ppf(.975)), 4),
             prob_up=round(float(1 - dist.cdf(0.0)), 4), mu_pct=round(float(mu), 5), scale_pct=round(float(sc), 5),
             df=None if not np.isfinite(nu) else round(float(nu), 4),
             actual=None if actual is None else round(float(actual), 4), source=source)
    return r


def heldout():
    f = D.load_fred(); aaa = D.load_aaa_daily()
    import os
    wcfg = json.load(open(ROOT / "results" / "weekly_cfg.json"))
    w, _ = P.weekly_dataset(f)
    wruns = [P.run_weekly(w, head="t", seed=s, **wcfg) for s in range(SEEDS)]
    wk = P.ensemble(wruns)
    df, _ = P.daily_dataset(f, weekly_mu_pct=wk.mu, train_end=P.D_SPLIT["train_end"])
    druns = [P.run_daily(df, head="t", seed=s, lookback=7, split=P.D_SPLIT) for s in range(SEEDS)]
    d = P.ensemble(druns)
    rows = []
    for T in d.index[d.index > P.D_SPLIT["val_end"]]:
        prev_d = T - pd.Timedelta(days=1)
        if T not in aaa.index or prev_d not in aaa.index:
            continue
        r = d.loc[T]
        rows.append(row_from(T, aaa[prev_d], r.mu, r.scale, r.df, aaa[T], "heldout"))
    return rows


def oos():
    base = pd.read_csv(D.DATA / "manual_aaa.csv"); rows = []
    tmp = Path(tempfile.mkdtemp()); shutil.copytree(D.DATA, tmp / "data"); orig = D.DATA; D.DATA = tmp / "data"
    try:
        for T in ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07"]:
            prev = (pd.Timestamp(T) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            add = pd.DataFrame([dict(date=k, aaa=v) for k, v in OOS_ACTUALS.items() if k <= prev])
            pd.concat([base[~base.date.isin(add.date)], add]).sort_values("date").to_csv(D.DATA / "manual_aaa.csv", index=False)
            fc = service.forecast(verbose=False)
            assert fc["target_date"] == T and abs(fc["last_aaa"] - OOS_ACTUALS[prev]) < 1e-9, (fc["target_date"], T)
            rows.append(row_from(T, fc["last_aaa"], fc["mu_pct"], fc["scale_pct"], fc["df"], OOS_ACTUALS[T], "oos"))
    finally:
        D.DATA = orig
    return rows


if __name__ == "__main__":
    keep = []
    if OUT.exists():
        keep = [r for r in json.load(open(OUT))["days"] if r["source"] == "live"]
    days = heldout() + oos() + keep
    days.sort(key=lambda r: r["date"])
    meta = dict(
        about="Every row is a forecast made before its actual was known. heldout = model fitted through Jun 30 2026; "
              "oos = shipped model, data available the day before; live = frozen by the daily update job.",
        price="AAA national average regular, $/gal",
        repo="https://github.com/OrvilleRockwellJr/aai6640-team5")
    OUT.write_text(json.dumps(dict(meta=meta, days=days), indent=1))
    ok = [r for r in days if r["actual"] is not None]
    err = np.array([abs(r["actual"] - r["pred"]) * 100 for r in ok]); nv = np.array([abs(r["actual"] - r["prev"]) * 100 for r in ok])
    print(f"wrote {OUT.name}: {len(days)} days ({sum(r['source']=='heldout' for r in days)} heldout, "
          f"{sum(r['source']=='oos' for r in days)} oos, {len(keep)} live)")
    print(f"mean abs error {err.mean():.2f} c vs 'tomorrow = today' {nv.mean():.2f} c; "
          f"80% interval hit rate {np.mean([r['lo80'] <= r['actual'] <= r['hi80'] for r in ok]):.0%}")
