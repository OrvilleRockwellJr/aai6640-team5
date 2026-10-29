"""Forecasters. Every model returns a predictive distribution (mu, scale, df) for the next log change,
in percent units (y*100). df=inf means Gaussian.

  Persistence     : random walk, mu=0, scale = rolling sd                      (naive baseline)
  ECM             : asymmetric error-correction OLS (Borenstein et al., 1997) (statistical baseline)
  Gaussian LSTM   : 2x LSTM + Gaussian NLL                                     (Stage 1 deep baseline)
  GARCH-LSTM-t    : LSTM fed GARCH(1,1) volatility, Student-t head scaled by GARCH sd (Stage 3 extension;
                    successor of the proposal's VAD-LSTM recursion, now a trainable hybrid in the
                    style of Kim & Won, 2018)
"""
import numpy as np, pandas as pd
from arch import arch_model

# ----------------------------------------------------------------------------- GARCH feature (no look-ahead)
def garch_feature(returns, train_end):
    """Fit GARCH(1,1)-t on returns up to train_end only, then run the variance recursion forward with
    fixed parameters. Value at index t is the one-step-ahead sd sigma_{t+1|t} (percent units)."""
    r = returns.dropna() * 100
    fit = arch_model(r[r.index <= train_end], mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    om, al, be = fit.params["omega"], fit.params["alpha[1]"], fit.params["beta[1]"]
    s2 = np.empty(len(r)); prev = r[r.index <= train_end].var()
    for i, x in enumerate(r.values):
        prev = om + al * x ** 2 + be * prev          # sigma^2_{t+1|t} uses r_t
        s2[i] = prev
    return pd.Series(np.sqrt(s2), index=r.index), fit.params


# ----------------------------------------------------------------------------- statistical baselines
def persistence(df, window):
    sd = (df["y"].shift(1).rolling(window, min_periods=max(4, window // 2)).std() * 100)
    return pd.DataFrame({"mu": 0.0, "scale": sd, "df": np.inf}, index=df.index)


def ecm_fit_predict(df, feats, train_mask):
    """OLS mean on the error-correction feature set; constant residual sd from training residuals."""
    X = np.column_stack([np.ones(len(df)), df[feats].values]); y = df["y"].values * 100
    ok = train_mask & ~np.isnan(X).any(1) & ~np.isnan(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    mu = X @ beta; sd = np.std(y[ok] - mu[ok], ddof=X.shape[1])
    return pd.DataFrame({"mu": mu, "scale": sd, "df": np.inf}, index=df.index), dict(zip(["const"] + feats, beta))


# ----------------------------------------------------------------------------- deep models (TensorFlow/Keras)
def build_lstm(lookback, n_feat, units=(32, 16), dropout=0.2, l2=1e-4, dense=16, head="gauss", garch_col=None, seed=0):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(l2) if l2 else None
    inp = keras.Input((lookback, n_feat))
    h = inp
    for i, u in enumerate(units):
        h = keras.layers.LSTM(u, return_sequences=i < len(units) - 1, kernel_regularizer=reg,
                              recurrent_initializer="orthogonal", unit_forget_bias=True)(h)
        h = keras.layers.Dropout(dropout)(h)
    h = keras.layers.Dense(dense, activation="gelu", kernel_regularizer=reg)(h)
    mu = keras.layers.Dense(1)(h)
    s_raw = keras.layers.Dense(1)(h)
    if head == "gauss":
        scale = keras.layers.Lambda(lambda z: tf.nn.softplus(z) + 1e-3)(s_raw)
        out = keras.layers.Concatenate()([mu, scale])
    elif head == "t_nog":  # Student-t head WITHOUT the GARCH input (ablation: isolates the GARCH effect)
        scale = keras.layers.Lambda(lambda z: tf.nn.softplus(z) + 0.05)(s_raw)
        nu = keras.layers.Lambda(lambda z: 2.5 + tf.nn.softplus(z))(keras.layers.Dense(1)(h))
        out = keras.layers.Concatenate()([mu, scale, nu])
    else:  # Student-t, scale anchored on the (raw, unstandardised) GARCH sd passed as last input column
        g = keras.layers.Lambda(lambda x: x[:, -1, -1:])(inp)
        scale = keras.layers.Lambda(lambda z: (tf.nn.softplus(z[0]) + 0.05) * z[1])([s_raw, g])
        nu = keras.layers.Lambda(lambda z: 2.5 + tf.nn.softplus(z))(keras.layers.Dense(1)(h))
        out = keras.layers.Concatenate()([mu, scale, nu])
    return keras.Model(inp, out)


def nll_gauss(y, p):
    import tensorflow as tf
    mu, s = p[:, 0:1], p[:, 1:2]
    return tf.reduce_mean(tf.math.log(s) + 0.5 * tf.square((y - mu) / s) + 0.9189385)


def nll_t(y, p):
    import tensorflow as tf
    mu, s, nu = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    z = (y - mu) / s
    lp = (tf.math.lgamma((nu + 1) / 2) - tf.math.lgamma(nu / 2) - 0.5 * tf.math.log(nu * np.pi) - tf.math.log(s)
          - (nu + 1) / 2 * tf.math.log1p(tf.square(z) / nu))
    return -tf.reduce_mean(lp)


def train_lstm(Xtr, ytr, Xva, yva, head="gauss", lr=1e-3, wd=1e-4, batch=32, epochs=400, patience=40,
               seed=0, verbose=0, **kw):
    import tensorflow as tf
    from tensorflow import keras
    m = build_lstm(Xtr.shape[1], Xtr.shape[2], head=head, seed=seed, **kw)
    m.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0),
              loss=nll_gauss if head == "gauss" else nll_t)
    cb = [keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True)]
    h = m.fit(Xtr, ytr[:, None] * 100, validation_data=(Xva, yva[:, None] * 100), epochs=epochs,
              batch_size=batch, callbacks=cb, verbose=verbose, shuffle=True)
    return m, h.history


def predict_dist(m, X, head="gauss"):
    p = m.predict(X, verbose=0)
    return pd.DataFrame({"mu": p[:, 0], "scale": p[:, 1], "df": p[:, 2] if head != "gauss" else np.inf})


class Scaler:
    """Per-feature standardisation fitted on training windows only. The GARCH sd column (if last) is
    left raw so the Student-t head can use it as a volatility anchor."""
    def __init__(self, keep_last_raw=False): self.k = keep_last_raw
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1]); self.m = flat.mean(0); self.s = flat.std(0) + 1e-8
        if self.k: self.m[-1], self.s[-1] = 0.0, 1.0
        return self
    def __call__(self, X): return (X - self.m) / self.s
