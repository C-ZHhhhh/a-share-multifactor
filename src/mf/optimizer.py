"""Mean-variance active portfolio optimisation (long-only, benchmark-relative).

max_w   alpha'w - lambda * (w-b)' (X F X' + D) (w-b) - c * ||w - w0||_1
s.t.    sum(w) = 1, 0 <= w <= b + max_active   (active-weight cap: an absolute cap would force
                                               underweights in mega-caps such as Moutai, ~6% of CSI800)
        |industry active weight| <= industry_dev
        |active size exposure|   <= size_dev
        0.5 * ||w - w0||_1       <= turnover_limit
        w_i <= w0_i  for stocks that cannot be bought (not tradable at signal)

w0 is the portfolio's ACTUAL weight vector at the signal close (drifted, after any blocked trades),
supplied by the backtest. Holdings that have left the universe must be sold; their weight counts
towards the turnover budget.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import CFG


def grinold_alpha(score: pd.Series, spec_vol: pd.Series, ic: float = 0.05) -> pd.Series:
    """alpha = IC * sigma * z  (Grinold's 'fundamental law' refinement)."""
    z = (score - score.mean()) / score.std()
    return ic * np.sqrt(spec_vol) * z


def optimize(alpha: pd.Series, bench: pd.Series, X: pd.DataFrame, F: pd.DataFrame, D: pd.Series,
             w0: pd.Series | None, buyable: pd.Series, pcfg: dict | None = None,
             cost: float = 0.002) -> tuple[pd.Series, dict]:
    pcfg = pcfg or CFG["portfolio"]
    names = X.index
    a = alpha.reindex(names).fillna(alpha.min() if len(alpha) else 0).values
    b = bench.reindex(names).fillna(0).values
    b = b / b.sum()
    w_prev = (w0.reindex(names).fillna(0).values if w0 is not None else np.zeros(len(names)))
    out_w = float(w0[~w0.index.isin(names)].sum()) if w0 is not None else 0.0   # forced sells (left universe)
    can_buy = buyable.reindex(names).fillna(False).values
    has_alpha = alpha.reindex(names).notna().values
    ind_cols = [c for c in X.columns if c not in ("size", "beta", "momentum", "volatility", "value", "liquidity")]
    Xi = X[ind_cols].values
    xs = X["size"].values
    Fm = F.loc[X.columns, X.columns].values
    Fm = (Fm + Fm.T) / 2 + 1e-10 * np.eye(len(Fm))
    Lf = np.linalg.cholesky(Fm)

    # Turnover policy: planned one-way turnover = 0.5 * (sum|target - w0| + forced exits), where w0 is the
    # actual book at the signal close and cash is the remainder; deploying cash therefore counts as buying.
    # The only exemption is a book that is mostly cash (initial build, or cash >= turnover_exempt_cash),
    # because otherwise the limit would forbid investing; every exemption is reported (info["turnover_exempt"]).
    cash0 = 1.0 - (w_prev.sum() + out_w)
    exempt = w0 is None or cash0 >= pcfg.get("turnover_exempt_cash", 0.5)
    w = cp.Variable(len(names))
    act = w - b
    risk = cp.sum_squares(Lf.T @ (X.values.T @ act)) + cp.sum_squares(cp.multiply(np.sqrt(D.values), act))
    trade = cp.norm1(w - w_prev)
    obj = a @ w - pcfg["risk_aversion"] * risk - (cost * trade if w0 is not None else 0)

    def cons(relax: float):
        cap = b + pcfg["max_active_weight"]
        c = [cp.sum(w) == 1, w >= 0, w <= cap,
             w <= np.where(can_buy & has_alpha, cap, w_prev),
             cp.abs(Xi.T @ act) <= pcfg["industry_dev"] * relax,
             cp.abs(xs @ act) <= pcfg["size_dev"] * relax]
        for name, lim in (pcfg.get("style_limits") or {}).items():   # optional extra style-exposure limits
            c.append(cp.abs(X[name].values @ act) <= lim * relax)
        if not exempt:
            c.append(0.5 * (trade + out_w) <= pcfg["turnover_limit"] * relax)
        return c

    status = None
    for relax in (1.0, 1.5, 3.0, 100.0):
        prob = cp.Problem(cp.Maximize(obj), cons(relax))
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            continue
        status = prob.status
        if status in ("optimal", "optimal_inaccurate") and w.value is not None:
            break
    wv = np.clip(w.value, 0, None)
    wv[wv < 1e-4] = 0
    wv = wv / wv.sum()
    res = pd.Series(wv, index=names)
    av = wv - b
    info = {"status": status, "relax": relax, "n_holdings": int((wv > 0).sum()),
            "ex_ante_te": float(np.sqrt(np.sum((Lf.T @ (X.values.T @ av)) ** 2) + np.sum(D.values * av ** 2)) * np.sqrt(12)),
            "planned_turnover": float(0.5 * (np.abs(wv - w_prev).sum() + out_w)),
            "cash_at_signal": float(cash0), "turnover_exempt": bool(exempt)}   # target vs actual at signal
    return res[res > 0], info
