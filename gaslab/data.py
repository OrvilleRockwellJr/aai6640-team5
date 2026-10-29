"""Data layer: FRED wholesale/crude/retail + Kalshi AAA settlements -> leak-free feature tensors.

Timing conventions (all decisions are made with information available at the bid time):
  * Weekly model: origin = Monday t (EIA retail R_t surveyed 8 a.m. Monday). Target = log(R_{t+1}/R_t).
    Wholesale/crude inputs use closes through the Friday before Monday t.
  * Daily model: origin = 8:00 a.m. ET on day D-1, when Kalshi opens the contract settling on day D.
    AAA's D-1 average is already posted (Kalshi centres the strike ladder on it). Wholesale/crude
    inputs use closes through business day D-2. Target = log(A_D/A_{D-1}).
"""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
GAL_PER_BBL = 42.0


def load_fred():
    f = pd.read_csv(DATA / "fred.csv", parse_dates=["date"]).set_index("date").sort_index()
    f["wti"] /= GAL_PER_BBL; f["brent"] /= GAL_PER_BBL            # $/bbl -> $/gal
    f.loc[f["wti"] <= 0, "wti"] = np.nan                           # 2020-04-20 negative WTI print
    return f


def load_aaa_daily():
    """AAA national regular average by date, from Kalshi settlement values plus any manual rows."""
    k = pd.read_csv(DATA / "kalshi_markets.csv")
    k = k[k.series.isin(["KXAAAGASD", "KXAAAGASW", "KXAAAGASM"])].copy()
    k["date"] = pd.to_datetime(k.exp, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    k["value"] = pd.to_numeric(k.value, errors="coerce")
    s = k.dropna(subset=["value"]).groupby("date").value.first()
    manual = DATA / "manual_aaa.csv"                               # optional: date,aaa
    if manual.exists():
        m = pd.read_csv(manual, parse_dates=["date"]).set_index("date")["aaa"]
        s = pd.concat([s, m]).groupby(level=0).last()
    return s.sort_index().rename("aaa")


def daily_contracts(series="KXAAAGASD"):
    k = pd.read_csv(DATA / "kalshi_markets.csv")
    k = k[k.series == series].copy()
    k["date"] = pd.to_datetime(k.exp, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    k["y"] = (k.result == "yes").astype(float).where(k.result.isin(["yes", "no"]))
    book = DATA / "kalshi_open_book.csv"
    if book.exists():
        k = k.merge(pd.read_csv(book)[["ticker", "bid", "ask", "hours_after_open"]], on="ticker", how="left")
    return k


# ----------------------------------------------------------------------------------- weekly (10 years)
WEEKLY_FEATS = ["d_retail", "d_gulf", "d_nyh", "d_brent", "up_gulf", "dn_gulf", "ect", "garch_sd"]


def weekly_frame(f=None):
    """One row per EIA Monday t with features known at t and target y = log(R_{t+1}/R_t)."""
    f = load_fred() if f is None else f
    r = f["retail_eia_weekly"].dropna()
    bd = f[["whsl_gulf", "whsl_nyh", "brent"]].ffill()
    rows = []
    for t in r.index:
        fri = bd.loc[: t - pd.Timedelta(days=3)].tail(5)            # Mon-Fri week ending the Friday before t
        if len(fri) < 5: continue
        rows.append(dict(date=t, retail=r.loc[t], gulf=fri.whsl_gulf.mean(), nyh=fri.whsl_nyh.mean(), brent=fri.brent.mean()))
    w = pd.DataFrame(rows).set_index("date")
    lw = np.log(w)
    w["d_retail"] = lw.retail.diff(); w["d_gulf"] = lw.gulf.diff(); w["d_nyh"] = lw.nyh.diff(); w["d_brent"] = lw.brent.diff()
    w["up_gulf"] = w.d_gulf.clip(lower=0); w["dn_gulf"] = w.d_gulf.clip(upper=0)         # rockets & feathers
    spread = lw.retail - lw.gulf                                                          # retail margin (log)
    w["ect"] = spread - spread.rolling(52, min_periods=26).mean()                        # error-correction term
    w["y"] = w.d_retail.shift(-1)
    return w


# ----------------------------------------------------------------------------------- daily (Kalshi)
DAILY_FEATS = ["d_aaa", "d_aaa_l2", "d_gulf", "d_brent", "ect", "w_drift", "garch_sd"]


def daily_frame(f=None, aaa=None, weekly_drift=None, extend_to=None):
    """One row per settlement day D with features known at 8 a.m. ET on D-1.
    extend_to adds future settlement dates (no target yet) so live contracts can be priced."""
    f = load_fred() if f is None else f
    aaa = load_aaa_daily() if aaa is None else aaa
    a = aaa.asfreq("D")
    if extend_to is not None and pd.Timestamp(extend_to) > a.index[-1]:
        a = a.reindex(pd.date_range(a.index[0], extend_to, freq="D"))
    bd = f[["whsl_gulf", "brent"]].ffill().asfreq("D").reindex(a.index.union(f.index)).ffill().reindex(a.index)
    df = pd.DataFrame(index=a.index)
    df["aaa_prev"] = a.shift(1)                                          # A_{D-1}
    df["d_aaa"] = np.log(a).diff().shift(1)                              # change into D-1
    df["d_aaa_l2"] = df["d_aaa"].shift(1)                                # change into D-2
    g2 = np.log(bd.whsl_gulf).shift(2); b2 = np.log(bd.brent).shift(2)   # closes through D-2
    df["d_gulf"] = g2.diff(); df["d_brent"] = b2.diff()
    spread = np.log(df.aaa_prev) - g2.rolling(5).mean()
    df["ect"] = spread - spread.rolling(28, min_periods=10).mean()
    df["w_drift"] = 0.0 if weekly_drift is None else weekly_drift.reindex(df.index, method="ffill").fillna(0) / 7.0
    df["y"] = np.log(a).diff()
    return df


def sequences(df, feats, lookback, rows=None, allow_missing_y=()):
    """Stack rolling windows of features ending at each row. Returns X, y, index.
    Rows listed in allow_missing_y are kept even without a target (used for live forecasts)."""
    vals = df[feats].values.astype("float32"); ys = df["y"].values.astype("float32")
    idx = []; X = []; Y = []
    want = set(df.index if rows is None else rows)
    for i in range(lookback - 1, len(df)):
        if df.index[i] not in want: continue
        win = vals[i - lookback + 1: i + 1]
        if np.isnan(win).any() or (np.isnan(ys[i]) and df.index[i] not in allow_missing_y): continue
        X.append(win); Y.append(ys[i]); idx.append(df.index[i])
    return np.array(X), np.array(Y), pd.DatetimeIndex(idx)
