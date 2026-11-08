import sys, json, shutil, tempfile; sys.path.insert(0,'/home/claude/aai6640-team5')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from gaslab import data as D, service
label, src_data, model_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
scale_mult = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
bias = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
book = pd.read_csv('/home/claude/kalshi_nov_book.csv')
aaa_true = {'2026-11-03':4.1493,'2026-11-04':4.1239,'2026-11-05':4.1003,'2026-11-06':4.0688,'2026-11-07':4.0371}
tmp = Path(tempfile.mkdtemp()); shutil.copytree(src_data, tmp/'data'); D.DATA = tmp/'data'
base = pd.read_csv(D.DATA/'manual_aaa.csv'); rows = []
for T in ['2026-11-04','2026-11-05','2026-11-06','2026-11-07']:
    prev = (pd.Timestamp(T)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    add = pd.DataFrame([dict(date=d, aaa=v) for d, v in aaa_true.items() if d <= prev])
    pd.concat([base[~base.date.isin(add.date)], add]).sort_values('date').to_csv(D.DATA/'manual_aaa.csv', index=False)
    fc = service.forecast(model_dir=model_dir, verbose=False)
    assert fc['target_date'] == T and abs(fc['last_aaa']-aaa_true[prev]) < 1e-9, (fc['target_date'], T)
    A = fc['last_aaa']; sc = fc['scale_pct']*scale_mult; fc['mu_pct'] += bias; fc['forecast'] = A*np.exp(fc['mu_pct']/100)
    dist = stats.t(df=fc['df'], loc=fc['mu_pct'], scale=sc) if np.isfinite(fc['df']) else stats.norm(fc['mu_pct'], sc)
    ev = 'KXAAAGASD-26NOV%s' % T[-2:]; b = book[book.event == ev].copy()
    b['y'] = (b.result == 'yes').astype(float)
    b['p'] = np.clip(1 - dist.cdf(100*np.log(b.strike/A)), 1e-4, 1-1e-4)
    b['pm'] = ((b.bid + b.ask)/2).clip(0.005, 0.995)
    co = json.load(open('/home/claude/aai6640-team5/results/stack_coef.json'))
    lg = lambda p: np.log(p/(1-p))
    b['ps'] = 1/(1+np.exp(-(co['model']*lg(b.p.clip(0.001,0.999)) + co['market']*lg(((b.bid+b.ask)/2).clip(0.01,0.99)) + co['const'])))
    real = aaa_true[T]
    rows.append(dict(date=T, last=A, fc=round(fc['forecast'],4), real=real, err_c=round((real-fc['forecast'])*100,2),
                     mkt_implied_ctr=None, brier_model=np.mean((b.p-b.y)**2), brier_mkt=np.mean((b.pm-b.y)**2), brier_stack=np.mean((b.ps-b.y)**2),
                     stale_days=fc['stale_days'], mu_pct=round(fc['mu_pct'],3), scale_pct=round(sc,3),
                     pit=round(float(dist.cdf(100*np.log(real/A))),3), n=len(b)))
r = pd.DataFrame(rows).drop(columns='mkt_implied_ctr'); pd.set_option('display.width',200)
print(label); print(r.round(4).to_string(index=False))
print('MEAN Brier model %.4f | stack %.4f | Kalshi %.4f | mean abs err %.2f c | naive persistence abs err %.2f c' % (
    r.brier_model.mean(), r.brier_stack.mean(), r.brier_mkt.mean(), r.err_c.abs().mean(),
    np.mean([abs(aaa_true[t]-aaa_true[(pd.Timestamp(t)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')])*100 for t in r.date])))
