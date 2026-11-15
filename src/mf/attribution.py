"""Performance attribution.

1. Brinson-Hood-Beebower industry attribution per holding period (cash is a sector with zero return):
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
        # cash is its own "sector" (return 0, no benchmark weight), so that portfolio weights sum to 1 and
        # allocation + selection + interaction = Rp - Rb holds exactly even when part of the book is cash
        df.loc["CASH"] = {"wp": max(0.0, 1.0 - df.wp.sum()), "wb": 0.0, "r": 0.0, "ind": "现金"}
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
                       active_ret: pd.Series, cost: pd.Series | None = None) -> pd.DataFrame:
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
    out["cost"] = -cost.reindex(out.index).fillna(0) if cost is not None else 0.0
    # what the risk model cannot explain: stock-specific returns, intra-month weight drift (exposures are
    # held at the execution-day weights) and execution-day timing. Costs are split out explicitly.
    out["specific"] = out["active"] - out[["industry"] + style_cols + ["cost"]].sum(axis=1)
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


def reconcile(holdings: dict, signal_of_exec: dict, data: pd.DataFrame, daily_ret: pd.DataFrame,
              strat_ret: pd.Series, bench_ret: pd.Series, cost: pd.Series) -> pd.DataFrame:
    """Per holding period, bridge the Brinson total to the strategy's actual active return.

    Brinson (BHB, incl. interaction) explains EXACTLY  Rp_bh - Rb_bh, where both are buy-and-hold returns of
    the post-trade weights / signal-date benchmark weights over the days (exec_k, exec_k+1]. The actual active
    return over the same days differs by:
      cost        trading costs charged on the next execution day
      execution   the next execution day is split open->VWAP (old weights) / VWAP->close (new weights)
                  instead of a full close-to-close day on the old weights
      benchmark   the daily benchmark re-weights by yesterday's float cap and follows index changes, the
                  buy-and-hold benchmark keeps signal-date weights for the month
    """
    exec_days = sorted(holdings)
    nav = (1 + strat_ret).cumprod()
    rows = []
    for k, d in enumerate(exec_days):
        end = exec_days[k + 1] if k + 1 < len(exec_days) else daily_ret.index[-1]
        days = daily_ret.index[(daily_ret.index > d) & (daily_ret.index <= end)]
        if days.empty:
            continue
        r = (1 + daily_ret.loc[days]).prod() - 1
        wp = holdings[d]
        wb = bench_weights(data, signal_of_exec[d])
        rp_bh = float((wp * r.reindex(wp.index).fillna(0)).sum())
        rb_bh = float((wb * r.reindex(wb.index).fillna(0)).sum())
        rp_act = float(nav.loc[end] / nav.loc[d] - 1)
        rb_act = float((1 + bench_ret.reindex(days).fillna(0)).prod() - 1)
        c = float(cost.reindex(days).fillna(0).sum())
        rows.append({"date": d, "brinson": rp_bh - rb_bh, "cost": -c, "execution": rp_act - rp_bh + c,
                     "benchmark": -(rb_act - rb_bh), "actual": rp_act - rb_act})
    return pd.DataFrame(rows).set_index("date")


def constraint_check(holdings: dict, signal_of_exec: dict, data: pd.DataFrame, turnover: pd.Series,
                     pcfg: dict, tol: float = 1e-3) -> pd.DataFrame:
    """Check the ACTUAL post-trade portfolio of every rebalance (after blocked trades) against the
    optimiser's constraints, measured at the signal date the optimiser used."""
    rows = []
    for d in sorted(holdings):
        sig = signal_of_exec[d]
        X = exposures(data, sig)
        wb = bench_weights(data, sig).reindex(X.index).fillna(0)
        wp = holdings[d].reindex(X.index.union(holdings[d].index)).fillna(0)
        act = wp.reindex(X.index).fillna(0) - wb
        ind_cols = [c for c in X.columns if c not in STYLE]
        ind = (X[ind_cols].T @ act).abs().max()
        size = abs(float(X["size"] @ act))
        cap = (wp - (wb.reindex(wp.index).fillna(0) + pcfg["max_active_weight"])).max()
        rows.append({"date": d, "max_industry_dev": ind, "size_exposure": size, "max_over_cap": max(cap, 0.0),
                     "turnover": float(turnover.get(d, np.nan)),
                     "out_of_universe": float(wp[~wp.index.isin(X.index)].sum())})
    t = pd.DataFrame(rows).set_index("date")
    t["ok_industry"] = t.max_industry_dev <= pcfg["industry_dev"] + tol
    t["ok_size"] = t.size_exposure <= pcfg["size_dev"] + tol
    t["ok_cap"] = t.max_over_cap <= tol
    t["ok_turnover"] = t.turnover <= pcfg["turnover_limit"] + tol
    return t
