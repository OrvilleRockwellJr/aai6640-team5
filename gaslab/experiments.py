"""Reproducible experiment suite used by the midterm notebook. Each function returns tables/figures."""
import os, warnings, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from . import data as D, models as M, evaluate as E, pipeline as P

SEEDS = range(3)
VA = ("2026-07-01", "2026-08-19"); TE = ("2026-08-20", "2026-10-19")   # fixed end: last settled Kalshi contract, so new AAA days do not change the test set
WVA = ("2023-01-01", "2024-12-31"); WTE = ("2025-01-01", "2027-12-31")

# Daily evaluation windows, in order of how untouched they are:
#   VA  Jul 1 - Aug 19 2026: validation, used for early stopping and model choice.
#   TE  Aug 20 - Oct 19 2026: was the test set, but it has now been scored many times (Parts 2-4, revisions, the
#       scorecard) and choices were made after looking at it. Treat TE as DEVELOPMENT data from now on: fine for
#       comparing models, but its scores are no longer an unbiased estimate of out-of-sample skill.
#   HO  Oct 20 - Nov 20 2026, minus Nov 4-7: the ONLY final test. Nov 4-7 are removed because they were already
#       scored in tools/backtest_nov2026 (RESULTS.md, 2026-11-08). HO may be read only through holdout_index().
HO = ("2026-10-20", "2026-11-20")
HO_EXCLUDE = ("2026-11-04", "2026-11-07")                              # inclusive
RESULTS = Path(__file__).resolve().parent.parent / "results"
FROZEN = RESULTS / "FROZEN.json"                                       # written by hand once the final model is fixed
HOLDOUT_LOG = RESULTS / "holdout_access.log"


def holdout_index(idx=None, purpose=""):
    """The settlement dates of the final holdout HO (minus HO_EXCLUDE), restricted to `idx` if given.

    This is the lock on the holdout: ALL holdout scoring must get its dates from here, never from HO directly.
    Every call appends a UTC timestamp (and whether access was granted) to results/holdout_access.log, and the
    call raises unless results/FROZEN.json exists, so the holdout cannot be scored before the model is frozen.

    Score the holdout only with the PIPELINE models (P.run_daily with the default D_SPLIT: trained through
    2026-06-30, early stopping on VA), never with model/ or service.forecast(): model/ is retrained on all data
    available (meta.json: trained through 2026-11-03), so it has seen part of HO.
    """
    RESULTS.mkdir(exist_ok=True)
    ok = FROZEN.exists()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(HOLDOUT_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}\t{'granted' if ok else 'REFUSED (no FROZEN.json)'}\t{purpose}\n")
    if not ok:
        raise PermissionError(f"Holdout is locked: {FROZEN} does not exist. Freeze the final model first.")
    days = pd.date_range(*HO, freq="D")
    days = days[(days < HO_EXCLUDE[0]) | (days > HO_EXCLUDE[1])]
    return days if idx is None else pd.DatetimeIndex(idx)[pd.DatetimeIndex(idx).isin(days)]


def _rows(res, df_y, rngs, contracts=None):
    out = []
    for name, d in res.items():
        for lab, (a, b) in rngs.items():
            dd = d.loc[a:b].dropna(); y = df_y.loc[dd.index].values * 100
            row = {"model": name, "split": lab, **E.dist_metrics(y, dd)}
            if contracts is not None:
                row.update(E.contract_metrics(E.price_ladder(contracts[contracts.date.isin(dd.index)], dd)))
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------- Stage 1: weekly
def weekly_stage1():
    w, gpar = P.weekly_dataset()
    base, beta = P.weekly_baselines(w)
    runs_g = [P.run_weekly(w, head="gauss", seed=s) for s in SEEDS]
    runs_n = [P.run_weekly(w, head="t_nog", seed=s) for s in SEEDS]      # ablation: Student-t, no GARCH input
    runs_t = [P.run_weekly(w, head="t", seed=s) for s in SEEDS]
    nu = float(gpar["nu"]); ok = w.dropna(subset=["y", "garch_sd"]).index      # GARCH(1,1)-t on its own, zero mean
    base["GARCH(1,1)-t alone"] = pd.DataFrame({"mu": 0.0, "scale": w.loc[ok, "garch_sd"] * np.sqrt((nu - 2) / nu), "df": nu}, index=ok)
    res = {**base, "Gaussian LSTM (x3)": P.ensemble(runs_g), "Student-t LSTM, no GARCH (x3)": P.ensemble(runs_n),
           "GARCH-LSTM-t (x3)": P.ensemble(runs_t)}
    tab = _rows(res, w["y"], {"val": WVA, "test": WTE})
    return dict(w=w, garch=gpar, ecm_beta=beta, runs_g=runs_g, runs_n=runs_n, runs_t=runs_t, res=res, table=tab)


# --------------------------------------------------------------------------------------- Stage 2: controlled
def stage2_experiments(w):
    """One factor at a time on the weekly GARCH-LSTM-t; everything else fixed at the baseline config
    (lookback 8, units 32/16, dropout 0.2, L2 1e-4, AdamW lr 1e-3, batch 32, early stop on val NLL)."""
    grid = {"E1 dropout": ("dropout", [0.0, 0.2, 0.4]),
            "E2 learning rate": ("lr", [3e-4, 1e-3, 3e-3])}
    rows, curves = [], {}
    for exp, (k, vals) in grid.items():
        for v in vals:
            nll, ep = [], []
            for s in SEEDS:
                r = P.run_weekly(w, head="t", seed=s, **{k: v})
                tr, va, te = r["masks"]
                nll.append(E.dist_metrics(r["y"][va] * 100, r["dist"][va])["NLL"]); ep.append(len(r["hist"]["loss"]))
                if s == 0: curves[(exp, v)] = r["hist"]
            rows.append({"experiment": exp, "setting": v, "val_NLL_mean": np.mean(nll), "val_NLL_sd": np.std(nll),
                         "epochs_mean": np.mean(ep)})
    return pd.DataFrame(rows), curves


def weekly_tuned(w, s2):
    """Use the Stage 2 findings: combine the best single-factor settings step by step and keep a change only if
    mean validation NLL improves. Returns the chosen config, the comparison table and the ensemble forecast."""
    best = {e: s2[s2.experiment == e].sort_values("val_NLL_mean").iloc[0].setting for e in s2.experiment.unique()}
    steps = [("baseline (lb 8, lr 1e-3, dropout 0.2)", {}),
             ("+ lr %g" % best["E2 learning rate"], {"lr": float(best["E2 learning rate"])}),
             ("+ dropout %g" % best["E1 dropout"], {"dropout": float(best["E1 dropout"])})]
    cfg, rows, keep_runs, best_nll = {}, [], None, np.inf
    for name, add in steps:
        trial = {**cfg, **add}
        runs = [P.run_weekly(w, head="t", seed=s, **trial) for s in SEEDS]
        va = runs[0]["masks"][1]
        nll = [E.dist_metrics(r["y"][va] * 100, r["dist"][va])["NLL"] for r in runs]
        kept = np.mean(nll) < best_nll
        rows.append({"step": name, "config": dict(trial), "val_NLL_mean": np.mean(nll), "val_NLL_sd": np.std(nll), "kept": kept})
        if kept: cfg, best_nll, keep_runs = trial, np.mean(nll), runs
    return cfg, pd.DataFrame(rows), keep_runs


# --------------------------------------------------------------------------------------- daily Kalshi model
def daily_stage(weekly_mu, lookbacks=(7, 14), heads=("gauss", "t")):
    df, gpar = P.daily_dataset(weekly_mu_pct=weekly_mu)
    c = P.daily_contracts_with_prev(); c = c[c.y.notna()]
    base, beta = P.daily_baselines(df)
    nu = float(gpar["nu"]); ok = df.dropna(subset=["y", "garch_sd"]).index    # GARCH(1,1)-t on its own, zero mean
    base["GARCH(1,1)-t alone"] = pd.DataFrame({"mu": 0.0, "scale": df.loc[ok, "garch_sd"] * np.sqrt((nu - 2) / nu), "df": nu}, index=ok)
    runs = {lb: {h: [P.run_daily(df, head=h, seed=s, lookback=lb) for s in SEEDS] for h in heads} for lb in lookbacks}
    res = {**base}
    for lb in runs:
        if "gauss" in runs[lb]: res[f"Gaussian LSTM lb{lb} (x3)"] = P.ensemble(runs[lb]["gauss"])
        if "t_nog" in runs[lb]: res[f"Student-t LSTM, no GARCH lb{lb} (x3)"] = P.ensemble(runs[lb]["t_nog"])
        if "t" in runs[lb]: res[f"GARCH-LSTM-t lb{lb} (x3)"] = P.ensemble(runs[lb]["t"])
    tab = _rows(res, df["y"], {"val": VA, "test": TE}, c)
    mk = [{"model": "Kalshi market (9 a.m., 1 h after open)", "split": lab, **E.market_metrics(c[c.date.isin(df.loc[a:b].index)])[0]}
          for lab, (a, b) in {"val": VA, "test": TE}.items()]
    tab = pd.concat([tab, pd.DataFrame(mk)], ignore_index=True)
    return dict(df=df, garch=gpar, ecm_beta=beta, contracts=c, runs=runs, res=res, table=tab)


def final_daily(ds, name="GARCH-LSTM-t lb7 (x3)"):
    """Final signal = model, plus a market-blended version. The blend is fitted on a calibration window
    (Aug 20 - Sep 19) and scored only on the later, untouched window (Sep 20 - Oct 19)."""
    df, c, d = ds["df"], ds["contracts"], ds["res"][name]
    ct = E.price_ladder(c[c.date.isin(df.loc[TE[0]:TE[1]].index)], d.loc[TE[0]:TE[1]])
    # stack weights fitted on the first half of the test window, scored on the second half (time-ordered, no overlap)
    cut = "2026-09-19"; cfit, ct = ct[ct.date <= cut], ct[ct.date > cut]
    st, coef = P.stack_with_market(cfit, ct)
    bt_model, tr_model = E.backtest(ct); bt_stack, tr_stack = E.backtest(st)
    _, mkt = E.market_metrics(ct)
    return dict(model_test=ct, stacked_test=st, market_test=mkt, stack_coef=coef,
                backtest=pd.DataFrame([{"signal": "model", **bt_model}, {"signal": "model+market stack", **bt_stack}]),
                trades_model=tr_model, trades_stack=tr_stack,
                metrics=pd.DataFrame([{"signal": "model", **E.contract_metrics(ct)},
                                      {"signal": "model+market stack", **E.contract_metrics(st)},
                                      {"signal": "market mid", **E.contract_metrics(mkt)}]))


def weekly_on_kalshi(w, d_weekly, start="2025-01-01"):
    """Score the 10-year weekly model on real Kalshi weekly AAA contracts using the EIA->AAA basis.
    Out of sample: only contracts whose forecast origin is in the weekly TEST period (from `start`), and the
    basis is estimated only on Mondays before `start`."""
    aaa = D.load_aaa_daily(); r = w["retail"]
    both = pd.concat([aaa, r], axis=1, join="inner").dropna(); both = both[both.index < start]
    basis = both["aaa"] - both["retail"]
    k = D.daily_contracts("KXAAAGASW"); k = k[k.y.notna()].copy()
    k["origin"] = k.date - pd.Timedelta(days=7)
    k = k[k.origin.isin(d_weekly.index) & k.date.isin(r.index) & (k.origin >= start)].copy()
    k["prev"] = r.reindex(k.origin).values + basis.mean()
    dd = d_weekly.reindex(k.origin.unique()); dd.index.name = "date"
    kk = k.rename(columns={"date": "settle"}).rename(columns={"origin": "date"})
    dd = dd.dropna()
    priced = E.price_ladder(kk, dd, extra_sd_pct=100 * basis.std() / r.mean()).dropna(subset=["p", "y"])
    return dict(basis_mean=float(basis.mean()), basis_sd=float(basis.std()), basis_n=int(len(basis)),
                metrics=E.contract_metrics(priced), priced=priced)


# --------------------------------------------------------------------------------------- diagnostics
def convergence_table(named_runs):
    """Per run: epochs, best epoch (1-based), train/val loss at the best epoch and at the end."""
    rows = []
    for name, runs in named_runs.items():
        for sd, r in enumerate(runs):
            h = r["hist"]; b = int(np.argmin(h["val_loss"]))
            rows.append(dict(model=name, seed=sd, epochs_run=len(h["loss"]), best_epoch=b + 1, train_loss_at_best=h["loss"][b],
                             val_loss_at_best=h["val_loss"][b], final_train_loss=h["loss"][-1], final_val_loss=h["val_loss"][-1],
                             train_windows=int(r["masks"][0].sum())))
    t = pd.DataFrame(rows); t["val_minus_train_at_best"] = t.val_loss_at_best - t.train_loss_at_best
    return t


def plot_learning_curves(runs_g, runs_t, title="Weekly model learning curves (seed 0)", names=("Gaussian LSTM", "GARCH-LSTM-t")):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, r, nm in [(ax[0], runs_g[0], names[0]), (ax[1], runs_t[0], names[1])]:
        ep = np.arange(1, len(r["hist"]["loss"]) + 1)
        a.plot(ep, r["hist"]["loss"], label="train NLL"); a.plot(ep, r["hist"]["val_loss"], label="validation NLL")
        best = int(np.argmin(r["hist"]["val_loss"])) + 1; a.axvline(best, ls="--", c="gray", lw=1, label=f"best epoch {best}")
        a.set_title(nm); a.set_xlabel("epoch (1-based)"); a.set_ylabel("NLL (percent units)"); a.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); return fig


def plot_stage2(curves):
    exps = sorted({e for e, _ in curves})
    fig, ax = plt.subplots(1, len(exps), figsize=(5 * len(exps), 3.4))
    for a, e in zip(ax, exps):
        for (ee, v), h in curves.items():
            if ee == e: a.plot(h["val_loss"], label=f"{v}")
        a.set_title(e); a.set_xlabel("epoch (1-based)"); a.set_ylabel("validation NLL"); a.legend(fontsize=8)
    fig.tight_layout(); return fig


def plot_reliability(fin, bins=10):
    fig, ax = plt.subplots(figsize=(5, 4.5)); edges = np.linspace(0, 1, bins + 1)
    for nm, c in [("model", fin["model_test"]), ("model+market stack", fin["stacked_test"]), ("market mid", fin["market_test"])]:
        b = np.clip(np.digitize(c.p, edges) - 1, 0, bins - 1)
        g = pd.DataFrame({"b": b, "p": c.p, "y": c.y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        ax.plot(g.p, g.y, "o-", label=nm)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("predicted P(YES)"); ax.set_ylabel("observed YES frequency")
    ax.set_title("Reliability, daily Kalshi test contracts"); ax.legend(fontsize=8); fig.tight_layout(); return fig


def plot_pit(w, d, rng=WTE, title="PIT histogram, weekly test (GARCH-LSTM-t)"):
    dd = d.loc[rng[0]:rng[1]].dropna(); y = w.loc[dd.index, "y"].values * 100
    u = E.cdf(y, dd.mu.values, dd.scale.values, dd.df.values)
    fig, ax = plt.subplots(figsize=(5, 3.4)); ax.hist(u, bins=10, range=(0, 1), edgecolor="k")
    ax.axhline(len(u) / 10, c="r", ls="--", lw=1); ax.set_xlabel("PIT value"); ax.set_ylabel("count"); ax.set_title(title)
    fig.tight_layout(); return fig, stats.kstest(u, "uniform")


def accuracy_by_distance(c):
    dist = (np.abs(c.strike - c.prev) * 100).round(1)
    b = pd.cut(dist, [-0.01, 0.5, 1, 2, 3, 10], labels=["<=0.5c", "0.5-1c", "1-2c", "2-3c", ">3c"])
    return c.assign(bucket=b, correct=((c.p > 0.5) == (c.y == 1))).groupby("bucket", observed=True).agg(
        contracts=("y", "size"), accuracy=("correct", "mean"), brier=("p", lambda p: np.mean((p - c.loc[p.index, "y"]) ** 2)))


def error_by_regime(w, d, rng=WTE):
    dd = d.loc[rng[0]:rng[1]].dropna(); x = w.loc[dd.index]
    reg = np.where(x.d_gulf > 0, "wholesale rising", "wholesale falling")
    y = x.y.values * 100
    return pd.DataFrame({"regime": reg, "abs_err": np.abs(y - dd.mu.values),
                         "dir_ok": np.sign(dd.mu.values) == np.sign(y)}).groupby("regime").agg(
        weeks=("abs_err", "size"), MAE=("abs_err", "mean"), dir_acc=("dir_ok", "mean"))
