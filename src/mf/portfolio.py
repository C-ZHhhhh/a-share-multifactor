"""Portfolio construction shared by the main backtest (step 5) and the robustness tests (step 7)."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .attribution import bench_weights
from .backtest import run_backtest
from .optimizer import grinold_alpha, optimize
from .risk import risk_model_at


def topn_targets(scores: pd.Series, n: int) -> dict:
    out = {}
    for d, s in scores.dropna().groupby(level=0):
        top = s.droplevel(0).nlargest(n)
        out[d] = pd.Series(1.0 / len(top), index=top.index)
    return out


def optimised_backtest(score: pd.Series, data, fret, resid, mkt, pcfg=None, label="opt", end=None,
                       cost_on=True, progress=True):
    """Optimise inside the backtest: each month the optimiser starts from the portfolio's actual
    (drifted, post-blocked-trade) weights, so the turnover limit and no-buy rule bind on real holdings."""
    info = []
    dates = sorted(score.dropna().index.get_level_values(0).unique())
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    bar = tqdm(total=len(dates), desc=label, disable=not progress)

    def target_fn(d, w_actual):
        s = score.loc[d].dropna()
        X, F, D = risk_model_at(d, data, fret, resid)
        g = data[data.date == d].set_index("code")
        alpha = grinold_alpha(s, D.reindex(s.index).fillna(D.median()))
        w, inf = optimize(alpha, bench_weights(data, d), X, F, D,
                          w_actual if len(w_actual) else None, g.tradable, pcfg)
        info.append({"date": d, **inf})
        bar.update()
        return w

    bt = run_backtest(target_fn, mkt, end=end, signal_dates=dates, cost_on=cost_on)
    bar.close()
    return bt, pd.DataFrame(info).set_index("date")


def calibrate_lambda(score: pd.Series, data, fret, resid, mkt, bench: pd.Series, pcfg: dict, end,
                     grid=None, target_te=None, label="calib", progress=True) -> tuple[float, pd.DataFrame]:
    """Pick the risk-aversion lambda by a RISK BUDGET: run the in-sample backtest for every grid value and
    choose the one whose mean ex-ante tracking error is closest to target_te. Returns are reported in the
    table for reference but are not used for the choice (picking the in-sample-IR winner is fragile)."""
    from .metrics import perf
    grid = grid or pcfg["lambda_grid"]
    target_te = target_te if target_te is not None else pcfg["target_te"]
    rows = {}
    for lam in grid:
        bt, inf = optimised_backtest(score, data, fret, resid, mkt, {**pcfg, "risk_aversion": float(lam)},
                                     f"{label} λ={lam}", end=end, progress=progress)
        p = perf(bt["ret"], bench, bt["turnover"], bt["cost"])
        rows[float(lam)] = {"ex_ante_te": inf.ex_ante_te.mean(), "IR": p["信息比率"], "超额年化": p["超额年化"],
                            "跟踪误差": p["跟踪误差"], "持仓数": inf.n_holdings.mean()}
    calib = pd.DataFrame(rows).T
    lam = float((calib.ex_ante_te - target_te).abs().idxmin())
    return lam, calib
