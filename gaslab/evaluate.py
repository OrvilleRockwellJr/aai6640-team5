"""Scoring: distribution metrics, Kalshi strike-ladder pricing, accuracy variants, and a fee-aware backtest."""
import numpy as np, pandas as pd
from scipy import stats


def cdf(x, mu, scale, df):
    z = (x - mu) / scale
    return np.where(np.isinf(df), stats.norm.cdf(z), stats.t.cdf(z, np.where(np.isinf(df), 30, df)))


def logpdf(y, mu, scale, df):
    z = (y - mu) / scale
    return np.where(np.isinf(df), stats.norm.logpdf(z), stats.t.logpdf(z, np.where(np.isinf(df), 30, df))) - np.log(scale)


def crps_mc(y, mu, scale, df, n=2000, seed=0):
    rng = np.random.default_rng(seed); z = rng.standard_normal((n, len(y)))
    t = np.where(np.isinf(df), z, rng.standard_t(np.where(np.isinf(df), 30, df), size=(n, len(y))))
    s = mu + scale * t; s2 = mu + scale * np.roll(t, 1, axis=0)
    return np.mean(np.abs(s - y), 0) - 0.5 * np.mean(np.abs(s - s2), 0)


def dist_metrics(y_pct, d):
    mu, sc, nu = d.mu.values, d.scale.values, d.df.values
    nz = y_pct != 0
    return {"NLL": float(-logpdf(y_pct, mu, sc, nu).mean()),
            "CRPS": float(crps_mc(y_pct, mu, sc, nu).mean()),
            "MAE": float(np.abs(y_pct - mu).mean()),
            "dir_acc": float((np.sign(mu[nz]) == np.sign(y_pct[nz])).mean()) if np.any(mu != 0) else np.nan,
            "cov80": float(np.mean(np.abs(cdf(y_pct, mu, sc, nu) - 0.5) <= 0.4)),
            "n": int(len(y_pct))}


def price_ladder(contracts, dist, prev_col="prev", extra_sd_pct=None):
    """P(YES) = P(level_D > k) = 1 - F(100*ln(k/prev)). extra_sd_pct widens the scale (e.g. basis noise)."""
    c = contracts.join(dist, on="date", how="inner")
    sc = c.scale if extra_sd_pct is None else np.sqrt(c.scale ** 2 + extra_sd_pct ** 2)
    x = 100 * np.log(c.strike / c[prev_col])
    c["p"] = np.clip(1 - cdf(x, c.mu, sc, c.df), 1e-4, 1 - 1e-4)
    return c


def contract_metrics(c, atm_cents=1.0):
    y, p = c.y.values, c.p.values
    atm = (np.abs(c.strike - c.prev) * 100 <= atm_cents).values
    out = {"contracts": int(len(c)), "events": int(c.date.nunique()),
           "accuracy": float(((p > 0.5) == (y == 1)).mean()),
           "atm_accuracy": float(((p[atm] > 0.5) == (y[atm] == 1)).mean()) if atm.any() else np.nan,
           "atm_n": int(atm.sum()),
           "brier": float(np.mean((p - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    return out


def kalshi_fee(price, rate=0.07):
    """Kalshi taker fee per contract: rate * P * (1-P), rounded up to the centicent ($0.0001), per the Kalshi fee
    schedule effective July 7, 2026. Resting (maker) orders on KXAAAGASD are not in the non-standard fee table,
    so their maker multiplier defaults to 0 (no maker fee); this backtest conservatively assumes taker orders."""
    return np.ceil(rate * price * (1 - price) * 10000 - 1e-9) / 10000


def backtest(c, edge=0.03):
    """Take one contract per signal at the first-hour book after open. YES at ask, NO at (1-bid)."""
    c = c.dropna(subset=["bid", "ask", "y"]).copy()
    c = c[(c.ask > 0) & (c.ask < 1) & (c.bid > 0)]
    buy_yes = c.p - c.ask - kalshi_fee(c.ask) > edge
    no_px = 1 - c.bid
    buy_no = (1 - c.p) - no_px - kalshi_fee(no_px) > edge
    pnl = np.where(buy_yes, c.y - c.ask - kalshi_fee(c.ask), 0.0) + np.where(buy_no, (1 - c.y) - no_px - kalshi_fee(no_px), 0.0)
    trades = buy_yes | buy_no
    c["pnl"] = pnl; c["trade"] = trades
    cost = np.where(buy_yes, c.ask, 0) + np.where(buy_no, no_px, 0)
    return {"trades": int(trades.sum()), "hit_rate": float((pnl[trades] > 0).mean()) if trades.any() else np.nan,
            "pnl_per_contract": float(pnl[trades].mean()) if trades.any() else np.nan,
            "total_pnl_$": float(pnl.sum()), "capital_$": float(cost.sum()),
            "roi": float(pnl.sum() / cost.sum()) if cost.sum() else np.nan}, c


def market_metrics(c):
    m = c.dropna(subset=["bid", "ask", "y"]).copy()
    m["p"] = ((m.bid + m.ask) / 2).clip(0.005, 0.995)
    return contract_metrics(m), m


def event_bootstrap(a, b, n=2000, seed=0):
    """95% CIs that respect the small number of events: resample whole settlement days (all strikes of a day
    move together), then recompute accuracy and Brier for signal a, signal b and the difference a - b.
    a, b: priced contract tables (ticker, date, p, y) on the same contracts."""
    m = a[["ticker", "date", "p", "y", "strike", "prev"]].merge(b[["ticker", "p"]], on="ticker", suffixes=("_a", "_b"))
    m["acc_a"] = ((m.p_a > .5) == (m.y == 1)).astype(float); m["acc_b"] = ((m.p_b > .5) == (m.y == 1)).astype(float)
    m["br_a"] = (m.p_a - m.y) ** 2; m["br_b"] = (m.p_b - m.y) ** 2
    m["atm"] = (np.abs(m.strike - m.prev) * 100 <= 1.0)
    g = m.groupby("date"); days = list(g.groups); rng = np.random.default_rng(seed)
    sums = g[["acc_a", "acc_b", "br_a", "br_b"]].sum(); cnt = g.size()
    atm = m[m.atm].groupby("date")[["acc_a", "acc_b"]].sum().reindex(days).fillna(0); atm_n = m[m.atm].groupby("date").size().reindex(days).fillna(0)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(days), len(days)); S = sums.iloc[idx].sum(); N = cnt.iloc[idx].sum()
        A = atm.iloc[idx].sum(); An = atm_n.iloc[idx].sum()
        stats.append([S.acc_a / N, S.acc_b / N, (S.acc_a - S.acc_b) / N, S.br_a / N, S.br_b / N, (S.br_a - S.br_b) / N,
                      A.acc_a / An, A.acc_b / An, (A.acc_a - A.acc_b) / An])
    stats = np.array(stats); S = sums.sum(); N = cnt.sum(); A = atm.sum(); An = atm_n.sum()
    point = [S.acc_a / N, S.acc_b / N, (S.acc_a - S.acc_b) / N, S.br_a / N, S.br_b / N, (S.br_a - S.br_b) / N,
             A.acc_a / An, A.acc_b / An, (A.acc_a - A.acc_b) / An]
    names = ["accuracy A", "accuracy B", "accuracy A-B", "Brier A", "Brier B", "Brier A-B", "ATM accuracy A", "ATM accuracy B", "ATM accuracy A-B"]
    lo, hi = np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)
    return pd.DataFrame({"metric": names, "estimate": point, "ci95_low": lo, "ci95_high": hi}), dict(contracts=int(N), events=len(days), atm_contracts=int(An))
