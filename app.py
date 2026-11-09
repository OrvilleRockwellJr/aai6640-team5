"""GasLab dashboard (for the team's own use).

    pip install streamlit
    streamlit run app.py

Runs the saved model in model/ on the current data/ (no training) and shows tomorrow's forecast, any-threshold
probabilities, the Kalshi strike ladder, and the honest scorecard. Add fresh rows to data/ (or run update_data.py)
and press "Re-run model" to refresh.
"""
import json
import numpy as np, pandas as pd, streamlit as st
from scipy import stats
from pathlib import Path

st.set_page_config(page_title="GasLab", page_icon="⛽", layout="wide")
ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner="Running the model...")
def run(_token=0):
    import export_demo
    return export_demo.build()


st.title("GasLab: tomorrow's U.S. gas price")
st.caption("GARCH-LSTM-t forecast of the AAA national average, scored against Kalshi contracts. Research tool, not financial advice.")
if st.sidebar.button("Re-run model"):
    run.clear()
D = run()
A, mu, sc, nu = D["last_aaa"], D["dist"]["mu_pct"], D["dist"]["scale_pct"], D["dist"]["df"]
dist = stats.t(df=nu, loc=mu, scale=sc) if nu else stats.norm(loc=mu, scale=sc)
p_above = lambda k: float(1 - dist.cdf(100 * np.log(k / A)))

st.info(f"Forecast for {D['target_date']} · last AAA {D['last_aaa_date']} ${A:.4f} · model trained through {D['trained_through']}"
        + (f" · wholesale inputs {D['wholesale_stale_days']} days old" if D["wholesale_stale_days"] > 3 else ""))
c = st.columns(4)
c[0].metric("Forecast", f"${D['forecast']:.4f}", f"{D['change_cents']:+.2f} ¢")
c[1].metric("80% interval", f"${D['interval80'][0]:.3f} – ${D['interval80'][1]:.3f}")
c[2].metric("Direction", f"{'UP' if D['prob_up'] > .5 else 'DOWN'} {max(D['prob_up'], D['prob_down']):.0%}")
c[3].metric("Within ±0.5¢", f"{D['within_half_cent']:.0%}")

st.subheader("Ask the model a question")
lo, hi = A * np.exp(dist.ppf(.003) / 100), A * np.exp(dist.ppf(.997) / 100)
k = st.slider("Price threshold ($/gal)", float(round(lo, 3)), float(round(hi, 3)), float(round(A, 3)), 0.001, format="%.3f")
st.markdown(f"**P(price > ${k:.3f}) = {p_above(k):.1%}**")
xs = np.linspace(lo, hi, 300)
dens = dist.pdf(100 * np.log(xs / A)) / sc * 100 / xs
df_plot = pd.DataFrame({"price": xs, "density": dens, "above": np.where(xs >= k, dens, 0)}).set_index("price")
st.area_chart(df_plot, height=260)

st.subheader("Contract by contract")
L = pd.DataFrame(D["ladder"])
L["gap (pts)"] = ((L.model_p - L.market_last) * 100).round(1)
st.dataframe(L.rename(columns={"strike": "Strike", "model_p": "Model P(YES)", "market_last": "Market last trade", "volume": "Volume"}),
             hide_index=True, width="stretch")
st.caption("Market column is Kalshi's last trade at an unknown time, so it can include information after the model's 8 a.m. cutoff.")

st.subheader("Scorecard (out of sample)")
S = D["scorecard"]
st.write(S["source"])
st.dataframe(pd.DataFrame(S["rows"]).rename(columns=str.title), hide_index=True, width="stretch")
b = S["backtest"]
st.write(f"Fee-aware backtest ({b['rule']}): model-only {b['model_only']['roi']:.1%} over {b['model_only']['trades']} trades; "
         f"with the market {b['model_plus_market']['roi']:.1%} over {b['model_plus_market']['trades']} trades.")
st.warning(S["verdict"])
