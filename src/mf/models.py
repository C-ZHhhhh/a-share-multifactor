"""Stock-selection models, all trained walk-forward (expanding window).

Look-ahead control: the label of rebalance date t (return from t+1 VWAP to the
next rebalance's t+1 VWAP) is only fully realised on `label_date`, one trading day
*after* the next rebalance. At rebalance date m a model may only use rows whose
label_date < m, which in practice means labels from months <= m-2 (a one-month gap).
The same label-availability rule is used for every in-sample decision (see dataset.label_window).

Models
------
1. icir   : equal-risk composite, factor weights = rolling 24m mean(IC)/std(IC)
2. ridge  : cross-sectional Ridge regression on selected factors -> rank label
Both use the factor set selected walk-forward (factor_test.walk_forward_selection): at every month only
factors that pass the screen on labels realised before that month.
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


def predict_icir(data: pd.DataFrame, selection: pd.DataFrame, window: int = 24) -> pd.Series:
    """selection: (date x factor) signs from factor_test.walk_forward_selection (0 = not selected)."""
    factors = list(selection.columns)
    ics = pd.DataFrame({f: ic_series(data, f"z_{f}") for f in factors})
    known = data.groupby("date").label_date.first()           # when each month's IC becomes known
    out = []
    for d in _months(data):
        sg = selection.loc[d] if d in selection.index else pd.Series(0, index=factors)
        chosen = [f for f in factors if sg[f] != 0]
        if not chosen:
            continue
        hist = (ics.loc[ics.index.isin(known[known < d].index), chosen] * sg[chosen]).tail(window)
        if len(hist) < 12:
            w = pd.Series(1.0, index=chosen)                   # warm-up: equal weights
        else:
            w = (hist.mean() / hist.std()).clip(lower=0)       # drop factors whose IC flipped sign
            w = w / w.sum() if w.sum() > 0 else pd.Series(1.0, index=chosen)
        g = data[(data.date == d) & data.tradable]
        score = sum(g[f"z_{f}"] * sg[f] * w[f] for f in chosen)
        out.append(pd.Series(score.values, index=pd.MultiIndex.from_arrays([g.date, g.code])))
    return pd.concat(out).rename("score")


def _walk_forward(data: pd.DataFrame, cols, make_model, refit_every: int) -> pd.Series:
    """cols: a fixed list of columns, or a callable month -> list of columns (walk-forward factor set)."""
    months = _months(data)
    d = data[data.tradable]
    min_train = CFG["model"]["min_train_months"]
    out, model, use = [], None, None
    for i, m in enumerate(months):
        if i < min_train + GAP:
            continue
        if model is None or (i - min_train - GAP) % refit_every == 0:
            use = cols(m) if callable(cols) else cols
            if not use:
                model = None
                continue
            train = d[(d.label_date < m) & d.fwd_rank.notna()]   # labels already realised at m
            model = make_model().fit(train[use].values, train.fwd_rank.values)
        if model is None:
            continue
        g = d[d.date == m]
        pred = model.predict(g[use].values)
        out.append(pd.Series(pred, index=pd.MultiIndex.from_arrays([g.date, g.code])))
    return pd.concat(out).rename("score")


def predict_ridge(data: pd.DataFrame, selection: pd.DataFrame) -> pd.Series:
    """Refit monthly on the factors selected (walk-forward) at that month."""
    cols = lambda m: [f"z_{f}" for f in selection.columns if selection.loc[m, f] != 0] if m in selection.index else []
    return _walk_forward(data, cols, lambda: Ridge(alpha=10.0), refit_every=1)


def predict_gbdt(data: pd.DataFrame) -> pd.Series:
    def make():
        return HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
            l2_regularization=1.0, random_state=42)
    return _walk_forward(data, [f"z_{f}" for f in FACTORS], make, refit_every=3)


def zscore_by_date(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
