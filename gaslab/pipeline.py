"""End-to-end assembly used by the experiments, the notebook and predict.py."""
import numpy as np, pandas as pd
from . import data as D, models as M, evaluate as E

W_SPLIT = dict(train_start="2016-01-01", train_end="2022-12-31", val_end="2024-12-31")   # test = 2025-01-01 ->
D_SPLIT = dict(train_end="2026-06-30", val_end="2026-08-19")                             # test = 2026-08-20 ->
W_CFG = dict(lookback=8, dropout=0.2, lr=1e-3)   # baseline; replaced by the Stage 2 choice (see experiments.weekly_tuned)
D_CFG = dict(lookback=7, head="t")
ECM_W = ["d_retail", "up_gulf", "dn_gulf", "d_nyh", "d_brent", "ect"]
ECM_D = ["d_aaa", "d_aaa_l2", "d_gulf", "d_brent", "ect", "w_drift"]


def split_masks(idx, s, start=None):
    idx = pd.DatetimeIndex(idx)
    tr = (idx <= s["train_end"]) & ((idx >= start) if start else True)
    va = (idx > s["train_end"]) & (idx <= s["val_end"]); te = idx > s["val_end"]
    return tr, va, te


# ----------------------------------------------------------------------------------- weekly
def weekly_dataset(f=None, train_end=None):
    w = D.weekly_frame(f)
    g, gpar = M.garch_feature(w["d_retail"], train_end or W_SPLIT["train_end"])
    w["garch_sd"] = g.reindex(w.index)
    return w, gpar


def run_weekly(w, lookback=8, units=(32, 16), dropout=0.2, l2=1e-4, lr=1e-3, seed=0, head="gauss", feats=None, epochs=400,
               split=None, live_rows=()):
    feats = feats or (D.WEEKLY_FEATS if head == "t" else [c for c in D.WEEKLY_FEATS if c != "garch_sd"])
    X, y, idx = D.sequences(w, feats, lookback, allow_missing_y=live_rows)
    sp = split or W_SPLIT
    tr, va, te = split_masks(idx, sp, sp.get("train_start"))
    tr &= ~np.isnan(y); va &= ~np.isnan(y)
    sc = M.Scaler(keep_last_raw=head == "t").fit(X[tr])
    m, hist = M.train_lstm(sc(X[tr]), y[tr], sc(X[va]), y[va], head=head, lr=lr, seed=seed, epochs=epochs,
                           units=units, dropout=dropout, l2=l2)
    d = M.predict_dist(m, sc(X), head); d.index = idx
    return dict(model=m, hist=hist, dist=d, idx=idx, y=y, masks=(tr, va, te), scaler=sc, feats=feats, lookback=lookback)


def weekly_baselines(w):
    ok = w.dropna(subset=["y"] + ECM_W)
    tr, va, te = split_masks(ok.index, W_SPLIT, W_SPLIT["train_start"])
    ecm, beta = M.ecm_fit_predict(ok, ECM_W, tr)
    pers = M.persistence(ok, 26)
    return {"Persistence": pers, "Asymmetric ECM": ecm}, beta


# ----------------------------------------------------------------------------------- daily
def daily_dataset(f=None, weekly_mu_pct=None, train_end=None, extend_to=None):
    # weekly forecast made with Monday t's EIA survey is published Monday ~5 p.m., so it is usable for
    # bids from Tuesday 8 a.m. (settlement Wednesday) onward: shift by 2 days before forward-filling.
    drift = None if weekly_mu_pct is None else (weekly_mu_pct / 100.0).shift(2, freq="D")
    df = D.daily_frame(f, weekly_drift=drift, extend_to=extend_to)
    ret = df["y"].dropna()
    g, gpar = M.garch_feature(ret, train_end or D_SPLIT["train_end"])
    df["garch_sd"] = g.shift(1, freq="D").reindex(df.index)      # sigma_{D|D-1} uses r_{D-1}
    return df, gpar


def run_daily(df, lookback=7, units=(16,), dropout=0.3, l2=1e-3, lr=3e-3, seed=0, head="gauss", feats=None, epochs=600,
              split=None, live_rows=()):
    feats = feats or (D.DAILY_FEATS if head == "t" else [c for c in D.DAILY_FEATS if c != "garch_sd"])
    X, y, idx = D.sequences(df, feats, lookback, allow_missing_y=live_rows)
    tr, va, te = split_masks(idx, split or D_SPLIT)
    tr &= ~np.isnan(y); va &= ~np.isnan(y)
    sc = M.Scaler(keep_last_raw=head == "t").fit(X[tr])
    m, hist = M.train_lstm(sc(X[tr]), y[tr], sc(X[va]), y[va], head=head, lr=lr, seed=seed, epochs=epochs,
                           units=units, dropout=dropout, l2=l2, dense=8, batch=16, patience=60)
    d = M.predict_dist(m, sc(X), head); d.index = idx
    return dict(model=m, hist=hist, dist=d, idx=idx, y=y, masks=(tr, va, te), scaler=sc, feats=feats, lookback=lookback)


def daily_baselines(df):
    ok = df.dropna(subset=["y"] + ECM_D)
    tr, va, te = split_masks(ok.index, D_SPLIT)
    ecm, beta = M.ecm_fit_predict(ok, ECM_D, tr)
    pers = M.persistence(ok, 14)
    mom = pd.DataFrame({"mu": ok["d_aaa"] * 100, "scale": pers.scale, "df": np.inf}, index=ok.index)
    return {"Persistence": pers, "Momentum": mom, "Daily ECM": ecm}, beta


def daily_contracts_with_prev(aaa=None):
    c = D.daily_contracts("KXAAAGASD")
    aaa = D.load_aaa_daily() if aaa is None else aaa
    c = c.merge(aaa.rename("prev").shift(1, freq="D"), left_on="date", right_index=True, how="left")
    return c.dropna(subset=["prev", "strike"])


def ensemble(runs):
    """Average seeds as an equal-weight mixture, moment-matched to one distribution."""
    mus = np.stack([r["dist"].mu.values for r in runs]); scs = np.stack([r["dist"].scale.values for r in runs])
    dfs = np.stack([r["dist"].df.values for r in runs])
    var_each = np.where(np.isinf(dfs), scs ** 2, scs ** 2 * dfs / np.maximum(dfs - 2, 1e-3))
    mu = mus.mean(0); var = (var_each + mus ** 2).mean(0) - mu ** 2
    nu = np.where(np.isinf(dfs).all(0), np.inf, np.where(np.isinf(dfs), 30, dfs).mean(0))
    scale = np.where(np.isinf(nu), np.sqrt(var), np.sqrt(var * (nu - 2) / nu))
    return pd.DataFrame({"mu": mu, "scale": scale, "df": nu}, index=runs[0]["dist"].index)


def stack_with_market(c_val, c_test):
    """Logistic blend logit(p) = a*logit(p_model) + b*logit(p_market) + c, fitted on an earlier calibration window."""
    from sklearn.linear_model import LogisticRegression
    def feats(c):
        pm = ((c.bid + c.ask) / 2).clip(0.01, 0.99); p = c.p.clip(0.001, 0.999)
        return np.column_stack([np.log(p / (1 - p)), np.log(pm / (1 - pm))])
    v = c_val.dropna(subset=["bid", "ask", "y"]); t = c_test.dropna(subset=["bid", "ask", "y"]).copy()
    lr = LogisticRegression(C=1.0).fit(feats(v), v.y)
    t["p"] = np.clip(lr.predict_proba(feats(t))[:, 1], 1e-4, 1 - 1e-4)
    return t, dict(model=float(lr.coef_[0][0]), market=float(lr.coef_[0][1]), const=float(lr.intercept_[0]))
