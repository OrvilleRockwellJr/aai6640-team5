"""Train once, save, then forecast tomorrow's AAA price from new data without retraining.

    from gaslab import service
    service.train()        # fits weekly + daily GARCH-LSTM-t ensembles on all data, saves to model/
    service.forecast()     # loads model/, reads data/, prints tomorrow's price, intervals and confidence
"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from . import data as D, models as M, pipeline as P

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"


def _save(r, path):
    r["model"].save_weights(str(path) + ".weights.h5")
    sc = r["scaler"]
    np.savez(str(path) + "_scaler.npz", m=sc.m, s=sc.s, k=sc.k)


BUILD = {"weekly": dict(units=(32, 16), l2=1e-4, dense=16), "daily": dict(units=(16,), l2=1e-3, dense=8)}


def _load_predict(path, df, feats, lookback, live_rows, kind, dropout):
    X, _, idx = D.sequences(df, feats, lookback, allow_missing_y=live_rows)
    z = np.load(str(path) + "_scaler.npz")
    sc = M.Scaler(keep_last_raw=bool(z["k"])); sc.m, sc.s = z["m"], z["s"]
    m = M.build_lstm(X.shape[1], X.shape[2], head="t", dropout=dropout, **BUILD[kind]); m.load_weights(str(path) + ".weights.h5")
    d = M.predict_dist(m, sc(X), "t"); d.index = idx
    return dict(dist=d)


def train(seeds=3, out=MODEL_DIR):
    """Fit on everything available (validation = latest 104 weeks / 50 days for early stopping) and save."""
    out = Path(out); out.mkdir(exist_ok=True)
    aaa = D.load_aaa_daily(); f = D.load_fred()
    D_date = aaa.index[-1] + pd.Timedelta(days=1)
    w, _ = P.weekly_dataset(f)
    last_w = w.dropna(subset=["y"]).index[-1]
    wsplit = dict(train_start="2016-01-01", train_end=str((last_w - pd.Timedelta(weeks=104)).date()), val_end=str(last_w.date()))
    cfg_f = ROOT / "results" / "weekly_cfg.json"
    wcfg = json.load(open(cfg_f)) if cfg_f.exists() else {}
    wruns = [P.run_weekly(w, head="t", seed=s, split=wsplit, live_rows=[w.index[-1]], **wcfg) for s in range(seeds)]
    for s, r in enumerate(wruns): _save(r, out / f"weekly_s{s}")
    wk = P.ensemble(wruns)
    df, _ = P.daily_dataset(f, weekly_mu_pct=wk.mu, train_end=str(aaa.index[-1].date()), extend_to=D_date)
    last_d = df.dropna(subset=["y"]).index[-1]
    dsplit = dict(train_end=str((last_d - pd.Timedelta(days=50)).date()), val_end=str(last_d.date()))
    druns = [P.run_daily(df, head="t", seed=s, lookback=7, split=dsplit, live_rows=[D_date]) for s in range(seeds)]
    for s, r in enumerate(druns): _save(r, out / f"daily_s{s}")
    meta = dict(seeds=seeds, trained_through=str(aaa.index[-1].date()), weekly_cfg=wcfg, weekly_feats=wruns[0]["feats"],
                weekly_lookback=wruns[0]["lookback"], daily_feats=druns[0]["feats"], daily_lookback=7,
                daily_train_end=str(aaa.index[-1].date()), weekly_epochs=[len(r["hist"]["loss"]) for r in wruns],
                daily_epochs=[len(r["hist"]["loss"]) for r in druns])
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    return meta


def forecast(model_dir=MODEL_DIR, verbose=True):
    """Tomorrow's AAA price from saved models + current data/. No training happens here."""
    mdir = Path(model_dir); meta = json.load(open(mdir / "meta.json"))
    aaa = D.load_aaa_daily(); f = D.load_fred()
    last = aaa.index[-1]; D_date = last + pd.Timedelta(days=1)
    w, _ = P.weekly_dataset(f)
    wruns = [_load_predict(mdir / f"weekly_s{s}", w, meta["weekly_feats"], meta["weekly_lookback"], [w.index[-1]], "weekly", meta["weekly_cfg"].get("dropout", 0.2)) for s in range(meta["seeds"])]
    wk = P.ensemble(wruns)
    df, _ = P.daily_dataset(f, weekly_mu_pct=wk.mu, train_end=meta["daily_train_end"], extend_to=D_date)
    druns = [_load_predict(mdir / f"daily_s{s}", df, meta["daily_feats"], meta["daily_lookback"], [D_date], "daily", 0.3) for s in range(meta["seeds"])]
    d = P.ensemble(druns).loc[D_date]
    A = float(aaa.iloc[-1]); mu, sc, nu = float(d.mu), float(d.scale), float(d.df)
    dist = stats.t(df=nu, loc=mu, scale=sc) if np.isfinite(nu) else stats.norm(loc=mu, scale=sc)
    price = lambda pct: A * np.exp(pct / 100)
    q = lambda p: float(price(dist.ppf(p)))
    half = 0.5   # +/- half a cent, matches Kalshi's $0.005 strike spacing
    lo_pct, hi_pct = 100 * np.log((price(mu) - .005) / A), 100 * np.log((price(mu) + .005) / A)
    out = dict(target_date=str(D_date.date()), last_aaa_date=str(last.date()), last_aaa=A,
               forecast=float(price(mu)), change_cents=float((price(mu) - A) * 100),
               interval80=(q(.10), q(.90)), interval95=(q(.025), q(.975)),
               prob_up=float(1 - dist.cdf(0.0)), prob_down=float(dist.cdf(0.0)),
               confidence_within_half_cent=float(dist.cdf(hi_pct) - dist.cdf(lo_pct)),
               trained_through=meta["trained_through"], stale_days=int((D_date - pd.Timedelta(days=2) - f["whsl_gulf"].dropna().index[-1]).days))
    out.update(mu_pct=mu, scale_pct=sc, df=nu)
    out["direction"] = "UP" if out["prob_up"] > .5 else "DOWN"
    out["direction_confidence"] = max(out["prob_up"], out["prob_down"])
    if verbose:
        print(f"AAA national average forecast for {out['target_date']} (last value {out['last_aaa_date']}: ${A:.4f})")
        print(f"  Forecast price : ${out['forecast']:.4f}  ({out['change_cents']:+.2f} cents)")
        print(f"  80% interval   : ${out['interval80'][0]:.4f} to ${out['interval80'][1]:.4f}")
        print(f"  95% interval   : ${out['interval95'][0]:.4f} to ${out['interval95'][1]:.4f}")
        print(f"  Direction      : {out['direction']} with {out['direction_confidence']:.0%} probability")
        print(f"  Confidence     : {out['confidence_within_half_cent']:.0%} chance the price lands within +/- half a cent of the forecast")
        if out["stale_days"] > 3:
            print(f"  Note: wholesale data is {out['stale_days']} days old; add fresher rows to data/manual_wholesale.csv")
    return out
