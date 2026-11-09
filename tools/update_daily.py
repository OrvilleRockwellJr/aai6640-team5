"""Daily refresh for the GasLab demo. Run by .github/workflows/daily-update.yml, or by hand:

    python tools/update_daily.py            # fetch data, grade yesterday, forecast tomorrow, rewrite docs/*.json
    python tools/update_daily.py --offline  # skip the downloads (re-grade and re-forecast from data/ as it is)

What it does, in order
  1. FRED (no API key): appends new wholesale, crude and weekly retail rows to data/fred.csv.
  2. AAA: reads today's and yesterday's national average (regular) from gasprices.aaa.com and appends them to
     data/manual_aaa.csv. Existing rows are never overwritten.
  3. Fills in the actual price on any 'live' calendar row whose day has now been published.
  4. Runs the saved model (model/, no training) and freezes tomorrow's forecast as a new 'live' row in
     docs/history.json, unless that day already has one. A frozen forecast is never rewritten.
  5. Rewrites docs/forecast.json for the demo page.
A failed download is reported and skipped; the run still produces a forecast from the data it has.
"""
import argparse, io, json, re, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
HISTORY = ROOT / "docs" / "history.json"
FRED = {"GASREGW": "retail_eia_weekly", "DGASNYH": "whsl_nyh", "DGASUSGULF": "whsl_gulf",
        "DCOILWTICO": "wti", "DCOILBRENTEU": "brent"}
UA = {"User-Agent": "gaslab-daily-update (+https://github.com/OrvilleRockwellJr/aai6640-team5)"}


def get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def update_fred():
    path = DATA / "fred.csv"
    cur = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    added = 0
    for sid, col in FRED.items():
        try:
            new = pd.read_csv(io.StringIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")))
        except Exception as e:
            print(f"  FRED {sid}: skipped ({type(e).__name__}: {e})"); continue
        new.columns = ["date", "v"]
        new["date"] = pd.to_datetime(new["date"]); new["v"] = pd.to_numeric(new["v"], errors="coerce")
        new = new.dropna().set_index("date")["v"]
        last = cur[col].dropna().index.max()
        new = new[new.index > last] if pd.notna(last) else new
        for d, v in new.items():
            if d not in cur.index:
                cur.loc[d, :] = float("nan")
            cur.loc[d, col] = v; added += 1
    cur = cur.sort_index()
    cur.reset_index().to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  FRED: {added} new values; wholesale now through {cur['whsl_gulf'].dropna().index.max().date()}")


def aaa_prices():
    """(current, yesterday) regular national averages from the AAA home page table."""
    html = re.sub(r"\s+", " ", get("https://gasprices.aaa.com/?state=US"))
    out = []
    for label in ("Current Avg.", "Yesterday Avg."):
        m = re.search(re.escape(label) + r"\s*</td>\s*<td>\s*\$(\d\.\d{3,4})", html)
        if not m:
            raise ValueError(f"could not find '{label}' on the AAA page")
        v = float(m.group(1))
        if not 1.0 < v < 10.0:
            raise ValueError(f"implausible price {v}")
        out.append(v)
    return out


def update_aaa():
    path = DATA / "manual_aaa.csv"
    cur = pd.read_csv(path, parse_dates=["date"])
    try:
        today_px, yday_px = aaa_prices()
    except Exception as e:
        print(f"  AAA: skipped ({type(e).__name__}: {e})"); return
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    have = set(cur.date)
    add = [(d, v) for d, v in ((today - pd.Timedelta(days=1), yday_px), (today, today_px)) if d not in have]
    if add:
        cur = pd.concat([cur, pd.DataFrame(add, columns=["date", "aaa"])]).sort_values("date")
        cur.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  AAA: {today.date()} = ${today_px:.4f}, yesterday ${yday_px:.4f}; {len(add)} new row(s)")


def update_history(fc):
    from gaslab import data as D
    h = json.load(open(HISTORY)) if HISTORY.exists() else {"meta": {}, "days": []}
    aaa = D.load_aaa_daily(); by = {r["date"]: r for r in h["days"]}
    for r in h["days"]:                                      # 3. grade finished days
        t = pd.Timestamp(r["date"])
        if r["actual"] is None and t in aaa.index:
            r["actual"] = round(float(aaa[t]), 4)
    T = fc["target_date"]
    if T not in by:                                          # 4. freeze tomorrow
        from tools.build_history import row_from
        h["days"].append(row_from(T, fc["last_aaa"], fc["mu_pct"], fc["scale_pct"], fc["df"], None, "live"))
        print(f"  froze live forecast for {T}: ${fc['forecast']:.4f}")
    else:
        print(f"  {T} already has a {by[T]['source']} forecast; left as is")
    h["days"].sort(key=lambda r: r["date"])
    h["meta"]["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    HISTORY.write_text(json.dumps(h, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    if not a.offline:
        print("Fetching data"); update_fred(); update_aaa()
    from gaslab import service
    import export_demo
    fc = service.forecast(verbose=False)
    print("Updating calendar"); update_history(fc)
    out = export_demo.build()
    (ROOT / "docs" / "forecast.json").write_text(json.dumps(out, indent=1))
    print(f"Wrote docs/forecast.json: {out['target_date']} = ${out['forecast']:.4f} (data through {out['last_aaa_date']})")
