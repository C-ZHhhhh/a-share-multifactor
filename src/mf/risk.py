"""Simplified Barra-style structural risk model.

r_i,t = sum_k X_ik f_k,t + u_i,t       (daily WLS, weights = sqrt(float mcap))
X     = industry dummies + style exposures (size, beta, momentum, volatility,
        value, liquidity), standardised at each month-end and held for the month.
Sigma = X F X' + diag(D), F and D estimated by EWMA (half-life 90 trading days).
The same daily factor returns are reused for performance attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROC
from .factors import zscore
from .panel import to_wide

STYLE = {"size": "size", "beta": "beta_250", "momentum": "mom_12_1",
         "volatility": "vol_20", "value": "bp", "liquidity": "turn_20"}


def exposures(data: pd.DataFrame, date) -> pd.DataFrame:
    g = data[(data.date == date) & data.mcap.notna()].set_index("code")
    X = pd.get_dummies(g.industry, dtype=float)
    w = np.sqrt(g.mcap)
    for name, col in STYLE.items():
        x = g[col].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median())
        med, mad = x.median(), (x - x.median()).abs().median() * 1.4826
        x = x.clip(med - 5 * mad, med + 5 * mad)
        x = (x - np.average(x, weights=w)) / x.std()   # cap-weighted mean 0 (Barra convention)
        X[name] = x
    return X


def estimate_factor_returns(data: pd.DataFrame, long: pd.DataFrame):
    ret = to_wide(long, "ret")
    traded = to_wide(long, "traded").fillna(False).astype(bool)
    mcap = to_wide(long, "float_mcap").ffill()
    months = sorted(data.date.unique())
    f_list, u_list = [], []
    for i, m in enumerate(months):
        X = exposures(data, m)
        end = months[i + 1] if i + 1 < len(months) else ret.index[-1] + pd.Timedelta(days=1)
        days = ret.index[(ret.index > m) & (ret.index <= end)]
        for d in days:
            ok = traded.loc[d].reindex(X.index).fillna(False) & ret.loc[d].reindex(X.index).notna()
            Xd = X[ok.values]
            Xd = Xd.loc[:, Xd.abs().sum() > 0]
            y = ret.loc[d, Xd.index].clip(-0.21, 0.21)
            w = np.sqrt(mcap.loc[d, Xd.index].fillna(mcap.loc[d, Xd.index].median()))
            sw = np.sqrt(w.values)[:, None]
            beta, *_ = np.linalg.lstsq(Xd.values * sw, y.values * sw[:, 0], rcond=None)
            f_list.append(pd.Series(beta, index=Xd.columns, name=d))
            u_list.append(pd.Series(y.values - Xd.values @ beta, index=Xd.index, name=d))
    fret = pd.DataFrame(f_list).fillna(0.0)
    resid = pd.DataFrame(u_list)
    fret.to_parquet(PROC / "factor_returns.parquet")
    resid.to_parquet(PROC / "specific_returns.parquet")
    return fret, resid


def ewma_cov(x: pd.DataFrame, halflife: int = 90, window: int = 252) -> pd.DataFrame:
    x = x.tail(window)
    w = 0.5 ** (np.arange(len(x))[::-1] / halflife)
    w = w / w.sum()
    xc = x - (x * w[:, None]).sum()
    return pd.DataFrame((xc.values * w[:, None]).T @ xc.values, index=x.columns, columns=x.columns)


def risk_model_at(date, data, fret, resid, horizon: int = 21):
    """Returns (X, F, D) scaled to `horizon` trading days, using data strictly up to `date`."""
    X = exposures(data, date)
    hist = fret.loc[:date]
    F = ewma_cov(hist, 90) * horizon
    F = F.reindex(index=X.columns, columns=X.columns)
    # a factor with too little history (e.g. a new industry) must not look riskless: use the median variance
    # of the other factors of its kind and no covariances
    n_obs = hist.tail(252).reindex(columns=X.columns).notna().sum().reindex(X.columns).fillna(0)
    thin = n_obs < 60
    diag = pd.Series(np.diag(F.fillna(0).values), index=X.columns)
    style = [c for c in X.columns if c in STYLE]
    ind = [c for c in X.columns if c not in STYLE]
    for grp in (style, ind):
        med = diag[[c for c in grp if not thin[c]]].median() if any(not thin[c] for c in grp) else diag.median()
        for c in grp:
            if thin[c] or not np.isfinite(diag[c]) or diag[c] <= 0:
                F.loc[c, :] = 0.0
                F.loc[:, c] = 0.0
                F.loc[c, c] = med
    F = F.fillna(0.0)
    Fv = (F.values + F.values.T) / 2
    ev, evec = np.linalg.eigh(Fv)                       # project onto the PSD cone
    if ev.min() < 0:
        Fv = (evec * np.clip(ev, 0, None)) @ evec.T
    F = pd.DataFrame(Fv, index=X.columns, columns=X.columns)
    u = resid.loc[:date].tail(252)
    w = 0.5 ** (np.arange(len(u))[::-1] / 90)
    spec = ((u ** 2).mul(w, axis=0).sum() / (u.notna().mul(w, axis=0).sum())).reindex(X.index)
    # shrink specific variance toward the cross-sectional median; unknown or tiny values are not optimistic:
    # missing -> median, and a floor at the 10th percentile
    med = spec.median()
    spec = 0.7 * spec.fillna(med) + 0.3 * med
    spec = spec.clip(lower=spec.quantile(0.10))
    D = spec * horizon
    assert np.isfinite(F.values).all() and np.isfinite(D.values).all(), f"non-finite risk inputs on {date}"
    return X, F, D
