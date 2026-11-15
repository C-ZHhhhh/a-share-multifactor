"""Stock-selection models, all trained walk-forward (expanding window).

Look-ahead control: the label of rebalance date t (return from t+1 VWAP to the
next rebalance's t+1 VWAP) is only fully realised one trading day *after* the
next rebalance. So at rebalance month m we may only train on labels from months
<= m-2 (a one-month gap).

Models
------
1. icir   : equal-risk composite, factor weights = rolling 24m mean(IC)/std(IC)
2. ridge  : cross-sectional Ridge regression on selected factors -> rank label
3. gbdt   : HistGradientBoosting on all 19 factors (non-linear + interactions)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .config import CFG
from .factor_test import ic_series
from .factors import FACTORS

GAP = 1  # months between the last usable label and the prediction month


def _months(data: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(data.date.unique())


def predict_icir(data: pd.DataFrame, factors: list[str], signs: dict[str, int], window: int = 24) -> pd.Series:
    ics = pd.DataFrame({f: ic_series(data, f"z_{f}") * signs[f] for f in factors})
    months = _months(data)
    out = []
    for i, d in enumerate(months):
        hist = ics.loc[ics.index < months[max(i - GAP, 0)]].tail(window) if i > GAP else ics.iloc[:0]
        if len(hist) < 12:
            w = pd.Series(1.0, index=factors)          # warm-up: equal weights
        else:
            w = (hist.mean() / hist.std()).clip(lower=0)  # drop factors whose IC flipped sign
            w = w / w.sum() if w.sum() > 0 else pd.Series(1.0, index=factors)
        g = data[(data.date == d) & data.tradable]
        score = sum(g[f"z_{f}"] * signs[f] * w[f] for f in factors)
        out.append(pd.Series(score.values, index=pd.MultiIndex.from_arrays([g.date, g.code])))
    return pd.concat(out).rename("score")


def _walk_forward(data: pd.DataFrame, cols: list[str], make_model, refit_every: int) -> pd.Series:
    months = _months(data)
    d = data[data.tradable]
    min_train = CFG["model"]["min_train_months"]
    out, model = [], None
    for i, m in enumerate(months):
        if i < min_train + GAP:
            continue
        if model is None or (i - min_train - GAP) % refit_every == 0:
            train = d[(d.date <= months[i - 1 - GAP]) & d.fwd_rank.notna()]
            model = make_model().fit(train[cols].values, train.fwd_rank.values)
        g = d[d.date == m]
        pred = model.predict(g[cols].values)
        out.append(pd.Series(pred, index=pd.MultiIndex.from_arrays([g.date, g.code])))
    return pd.concat(out).rename("score")


def predict_ridge(data: pd.DataFrame, factors: list[str]) -> pd.Series:
    return _walk_forward(data, [f"z_{f}" for f in factors], lambda: Ridge(alpha=10.0), refit_every=1)


def predict_gbdt(data: pd.DataFrame) -> pd.Series:
    def make():
        return HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
            l2_regularization=1.0, random_state=42)
    return _walk_forward(data, [f"z_{f}" for f in FACTORS], make, refit_every=3)


def zscore_by_date(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
