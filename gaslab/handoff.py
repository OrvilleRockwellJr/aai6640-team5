"""Connects the four notebook parts into one pipeline.

Part 1 checks and fingerprints the data in data/ and saves results/part1.pkl.
Parts 2, 3 and 4 call part1() first, so they always run on the data Part 1 validated. Each part saves what the next
one needs (results/part2.pkl, part3.pkl). A saved file is used only if it was built from the same data; if you add
or change data in data/, everything downstream is rebuilt automatically. Missing files are rebuilt too, so every part
also works on its own."""
import hashlib, json, pickle
from pathlib import Path
import pandas as pd
from . import data as D, pipeline as P, experiments as X

R = Path("results")
DATA_FILES = ["fred.csv", "kalshi_markets.csv", "kalshi_open_book.csv", "manual_aaa.csv", "manual_wholesale.csv"]
DEFAULT_WEEKLY_CFG = {"lr": 0.003, "dropout": 0.4}   # selected by the Part 3 experiments


def fingerprint():
    h = hashlib.md5()
    for f in DATA_FILES:
        p = D.DATA / f
        if p.exists(): h.update(f.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


def save(name, obj):
    R.mkdir(exist_ok=True); pickle.dump({"fp": fingerprint(), "obj": obj}, open(R / f"{name}.pkl", "wb"))
    print(f"saved results/{name}.pkl for the next part (data fingerprint {fingerprint()})")


def _load(name):
    f = R / f"{name}.pkl"
    if not f.exists(): return None
    d = pickle.load(open(f, "rb"))
    if d["fp"] != fingerprint():
        print(f"results/{name}.pkl was built from different data: rebuilding"); return None
    print(f"loaded results/{name}.pkl (from the previous part, same data)"); return d["obj"]


def _merge_manual_wholesale():
    m = D.DATA / "manual_wholesale.csv"
    if not m.exists(): return 0
    f = pd.read_csv(D.DATA / "fred.csv", parse_dates=["date"]).set_index("date")
    add = pd.read_csv(m, parse_dates=["date"]).set_index("date")
    f = f.combine_first(add); f.update(add); f.sort_index().to_csv(D.DATA / "fred.csv")
    m.rename(D.DATA / "manual_wholesale.merged.csv"); return len(add)     # so rows are merged once


def part1(verbose=True):
    """Validate data/, merge any manual rows, print a data report and save results/part1.pkl."""
    n = _merge_manual_wholesale()
    if n and verbose: print(f"merged {n} manual wholesale rows into data/fred.csv")
    f = pd.read_csv(D.DATA / "fred.csv", parse_dates=["date"])
    need = ["date", "retail_eia_weekly", "whsl_nyh", "whsl_gulf", "wti", "brent"]
    assert list(f.columns[:6]) == need, f"data/fred.csv columns must be {need}, found {list(f.columns)}"
    assert f.date.is_monotonic_increasing and not f.date.duplicated().any(), "data/fred.csv dates must be sorted and unique"
    aaa = D.load_aaa_daily(); k = pd.read_csv(D.DATA / "kalshi_markets.csv")
    rep = dict(fred_rows=len(f), fred_first=str(f.date.min().date()), fred_last=str(f.date.max().date()),
               weekly_retail=int(f.retail_eia_weekly.notna().sum()), aaa_dates=len(aaa), aaa_last=str(aaa.index[-1].date()),
               kalshi_daily_events=int(k[k.series == "KXAAAGASD"].event.nunique()), fingerprint=fingerprint())
    if verbose:
        print("DATA REPORT (this is what Parts 2, 3 and 4 will use)")
        for a, b in rep.items(): print(f"  {a:20s} {b}")
    save("part1", rep); return rep


def require_part1():
    """First cell of Parts 2-4: make sure the data was validated, and say so."""
    rep = _load("part1")
    if rep is None: print("running the Part 1 data check first"); rep = part1(verbose=False)
    print(f"data: EIA/FRED {rep['fred_first']} to {rep['fred_last']}, {rep['aaa_dates']} dated AAA values, last AAA {rep['aaa_last']}")
    return rep


def part2():
    """Weekly frame + Stage 1 forecasts (needed by Parts 3 and 4)."""
    require_part1()
    o = _load("part2")
    if o is None:
        print("rebuilding Part 2 results from data/ (about 2 minutes)")
        s1 = X.weekly_stage1(); o = {"w": s1["w"], "res": s1["res"]}; save("part2", o)
    return o


def part3(s1):
    """Tuned weekly GARCH-LSTM-t forecast (needed by Part 4)."""
    o = _load("part3")
    if o is None:
        f = R / "weekly_cfg.json"
        wcfg = json.load(open(f)) if f.exists() else DEFAULT_WEEKLY_CFG
        print(f"retraining the tuned weekly model with {wcfg} (about 1 minute)")
        runs = [P.run_weekly(s1["w"], head="t", seed=s, **wcfg) for s in X.SEEDS]
        o = {"tuned": P.ensemble(runs), "wcfg": wcfg}; save("part3", o)
    return o
