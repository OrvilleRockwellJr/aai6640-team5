import sys, pickle; sys.path.insert(0,'/home/claude/aai6640-team5')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
from gaslab import data as D, pipeline as P, experiments as X
D.DATA = Path('/home/claude/scratch/data')
res = pickle.load(open('/home/claude/aai6640-team5/results/part2.pkl','rb'))['obj']['res']
wk = res['GARCH-LSTM-t (x3)']
df, _ = P.daily_dataset(weekly_mu_pct=wk.mu)
runs = [P.run_daily(df, head='t', seed=s, lookback=7) for s in range(3)]
d = P.ensemble(runs); y = df['y']*100
a, b = X.TE; t = d.loc[a:b].join(y.rename('y')).dropna()
print('calibration window', t.index[0].date(), t.index[-1].date(), len(t))
def nll(p, t=t):
    bias, lk = p; s = t.scale*np.exp(lk)
    return -stats.t.logpdf(t.y, df=t.df, loc=t.mu+bias, scale=s).mean()
r = optimize.minimize(nll, [0,0], method='Nelder-Mead'); bias, k = r.x[0], float(np.exp(r.x[1]))
print('raw NLL %.4f  -> calibrated %.4f | bias %+.4f pct  scale x%.3f' % (nll([0,0]), r.fun, bias, k))
# split-half stability: fit on first half, score second half
h = len(t)//2; f1, f2 = t.iloc[:h], t.iloc[h:]
r1 = optimize.minimize(lambda p: nll(p,f1), [0,0], method='Nelder-Mead')
print('first-half fit bias %+.4f scale x%.3f ; 2nd-half NLL raw %.4f vs calibrated %.4f' % (r1.x[0], np.exp(r1.x[1]), nll([0,0],f2), nll(r1.x,f2)))
print('coverage80 raw %.2f calibrated %.2f' % tuple(np.mean([ (lambda D_: (D_.cdf(t.y)>0.1)&(D_.cdf(t.y)<0.9))(stats.t(df=t.df,loc=t.mu+bb,scale=t.scale*kk)).mean() for bb,kk in [(0,1),(bias,k)]]) if False else
      [ ((lambda c:(c>0.1)&(c<0.9))(stats.t(df=t.df,loc=t.mu+bb,scale=t.scale*kk).cdf(t.y))).mean() for bb,kk in [(0,1),(bias,k)]]))
print('mean residual (y-mu) raw: %+.4f pct, sd of y %.3f, mean scale %.3f' % ((t.y-t.mu).mean(), t.y.std(), t.scale.mean()))
pd.Series(dict(bias=bias, k=k)).to_json('/home/claude/bt/calib.json')
