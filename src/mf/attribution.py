"""Performance attribution.

1. Brinson-Hood-Beebower industry attribution per holding period:
   allocation  = (wp_s - wb_s) * (rb_s - Rb)
   selection   = wb_s * (rp_s - rb_s)
   interaction = (wp_s - wb_s) * (rp_s - rb_s)
2. Factor (risk-model) attribution of daily active return:
   active_t = sum_k (x_p - x_b)_k * f_k,t  +  residual (stock-specific + trading)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import STYLE, exposures


def bench_weights(data: pd.DataFrame, date) -> pd.Series:
    g = data[(data.date == date) & data.mcap.notna()].set_index("code").mcap
    return g / g.sum()


def brinson(holdings: dict, signal_of_exec: dict, data: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    ind = data.drop_duplicates("code", keep="last").set_index("code").industry
    exec_days = sorted(holdings)
    rows = []
    for k, d in enumerate(exec_days):
        end = exec_days[k + 1] if k + 1 < len(exec_days) else daily_ret.index[-1]
        period = daily_ret.loc[(daily_ret.index > d) & (daily_ret.index <= end)]
        if period.empty:
            continue
        r = (1 + period).prod() - 1
        wp = holdings[d]
        wb = bench_weights(data, signal_of_exec[d])
        df = pd.DataFrame({"wp": wp, "wb": wb}).fillna(0)
        df["r"] = r.reindex(df.index).fillna(0)
        df["ind"] = ind.reindex(df.index).fillna("未分类")
        Rb = (df.wb * df.r).sum()
        for s, g in df.groupby("ind"):
            wps, wbs = g.wp.sum(), g.wb.sum()
            rps = (g.wp * g.r).sum() / wps if wps > 0 else 0.0
            rbs = (g.wb * g.r).sum() / wbs if wbs > 0 else 0.0
            if wps == 0:
                rps = rbs
            rows.append({"date": d, "industry": s, "allocation": (wps - wbs) * (rbs - Rb),
                         "selection": wbs * (rps - rbs), "interaction": (wps - wbs) * (rps - rbs)})
    return pd.DataFrame(rows)


def factor_attribution(holdings: dict, signal_of_exec: dict, data: pd.DataFrame, fret: pd.DataFrame,
                       active_ret: pd.Series) -> pd.DataFrame:
    exec_days = sorted(holdings)
    parts = []
    for k, d in enumerate(exec_days):
        end = exec_days[k + 1] if k + 1 < len(exec_days) else active_ret.index[-1]
        sig = signal_of_exec[d]
        X = exposures(data, sig)
        wb = bench_weights(data, sig).reindex(X.index).fillna(0)
        wp = holdings[d].reindex(X.index).fillna(0)
        act_x = X.T @ (wp - wb)
        f = fret.loc[(fret.index > d) & (fret.index <= end)]
        contrib = f.reindex(columns=act_x.index).fillna(0) * act_x.values
        parts.append(contrib)
    c = pd.concat(parts)
    style_cols = list(STYLE)
    out = pd.DataFrame(index=c.index)
    out["industry"] = c.drop(columns=[s for s in style_cols if s in c.columns]).sum(axis=1)
    for s in style_cols:
        out[s] = c[s] if s in c.columns else 0.0
    out["active"] = active_ret.reindex(out.index).fillna(0)
    out["specific+trading"] = out["active"] - out[["industry"] + style_cols].sum(axis=1)
    return out


def active_exposure_series(holdings: dict, signal_of_exec: dict, data: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for d in sorted(holdings):
        sig = signal_of_exec[d]
        X = exposures(data, sig)
        wb = bench_weights(data, sig).reindex(X.index).fillna(0)
        wp = holdings[d].reindex(X.index).fillna(0)
        rows[d] = (X[list(STYLE)].T @ (wp - wb))
    return pd.DataFrame(rows).T
