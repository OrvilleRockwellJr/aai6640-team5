"""Export the model's latest forecast to JSON for the front ends (demo/index.html and app.py).

    python export_demo.py                 # writes demo/forecast.json
    python export_demo.py --out some/dir  # also copies forecast.json + scorecard.json there

Runs the saved model in model/ on the current data/ (no training). When the team retrains (python train_model.py or
gaslab.service.train()) or adds new rows to data/, run this again and the front ends update.
"""
import argparse, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo"


def build():
    from gaslab import service, data as D
    fc = service.forecast(verbose=False)
    A = fc["last_aaa"]
    mu, sc, nu = fc["mu_pct"], fc["scale_pct"], fc["df"]
    dist = stats.t(df=nu, loc=mu, scale=sc) if np.isfinite(nu) else stats.norm(loc=mu, scale=sc)

    # Strike ladder for the target date: the model's P(YES) = P(price > strike), next to Kalshi's last trade.
    k = pd.read_csv(D.DATA / "kalshi_markets.csv")
    k = k[k.series == "KXAAAGASD"].copy()
    k["date"] = pd.to_datetime(k.exp, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    k = k[k.date == pd.Timestamp(fc["target_date"])].sort_values("strike")
    ladder = []
    for r in k.itertuples():
        p = float(1 - dist.cdf(100 * np.log(r.strike / A)))
        ladder.append(dict(strike=float(r.strike), model_p=round(min(max(p, 1e-4), 1 - 1e-4), 4),
                           market_last=None if pd.isna(r.last) else float(r.last),
                           volume=None if pd.isna(r.vol) else float(r.vol)))

    aaa = D.load_aaa_daily().tail(90)
    hist = [dict(date=str(i.date()), aaa=float(v)) for i, v in aaa.items()]
    meta = json.load(open(ROOT / "model" / "meta.json"))
    score = json.load(open(DEMO / "scorecard.json"))
    return dict(
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        model="GARCH-LSTM-t (3-seed ensembles: weekly EIA retail model feeding a daily AAA model)",
        target_date=fc["target_date"], last_aaa_date=fc["last_aaa_date"], last_aaa=A,
        trained_through=meta["trained_through"], wholesale_stale_days=fc["stale_days"],
        forecast=fc["forecast"], change_cents=fc["change_cents"],
        interval80=fc["interval80"], interval95=fc["interval95"],
        prob_up=fc["prob_up"], prob_down=fc["prob_down"],
        within_half_cent=fc["confidence_within_half_cent"],
        dist=dict(mu_pct=mu, scale_pct=sc, df=None if not np.isfinite(nu) else nu),
        ladder=ladder, history=hist, scorecard=score,
        repo="https://github.com/OrvilleRockwellJr/aai6640-team5")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = build()
    DEMO.mkdir(exist_ok=True)
    (DEMO / "forecast.json").write_text(json.dumps(out, indent=1))
    print(f"wrote demo/forecast.json: forecast for {out['target_date']} = ${out['forecast']:.4f}, "
          f"{len(out['ladder'])} strikes, data through {out['last_aaa_date']}")
    if a.out:
        d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
        for f in ("forecast.json", "scorecard.json"): shutil.copy(DEMO / f, d / f)
        print("copied to", d)


if __name__ == "__main__":
    main()
