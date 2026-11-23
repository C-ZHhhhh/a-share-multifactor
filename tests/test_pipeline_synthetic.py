"""Synthetic end-to-end run of the real pipeline functions (no downloads): walk-forward factor screen ->
Ridge -> risk model -> optimiser inside the account backtest -> holdings table -> attribution identities.
Runs in CI on Python 3.9 / 3.11 in a few seconds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from mf import holdings as H
from mf.attribution import brinson, constraint_check, reconcile
from mf.backtest import Market
from mf.config import CFG
from mf.factor_test import walk_forward_selection
from mf.metrics import perf
from mf.models import predict_ridge
from mf.portfolio import optimised_backtest


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2016-01-01", "2019-12-31")
    codes = [f"s{i:02d}" for i in range(40)]
    n = len(codes)
    signal = rng.normal(size=n)                                   # persistent stock "quality"
    ret = pd.DataFrame(rng.normal(0.0003, 0.015, (len(days), n)) + 0.0004 * signal, index=days, columns=codes)
    months = pd.Series(days, index=days).groupby(days.to_period("M")).max().values
    months = pd.DatetimeIndex(months)
    pos = {d: i for i, d in enumerate(days)}
    exec_d = [days[pos[m] + 1] if pos[m] + 1 < len(days) else pd.NaT for m in months]
    nav = (1 + ret).cumprod()
    rows = []
    for k, (m, e) in enumerate(zip(months, exec_d)):
        nxt = exec_d[k + 1] if k + 1 < len(exec_d) else pd.NaT
        fwd = (nav.loc[nxt] / nav.loc[e] - 1) if pd.notna(nxt) and pd.notna(e) else pd.Series(np.nan, index=codes)
        noise = rng.normal(size=n)
        for i, c in enumerate(codes):
            rows.append({"date": m, "code": c, "exec_date": e, "label_date": nxt, "tradable": True,
                         "industry": "A" if i < 20 else "B", "mcap": 10 + i, "fwd_ret": fwd[c],
                         "z_ep": signal[i] + 0.5 * noise[i], "z_bp": rng.normal(), "z_size": rng.normal(),
                         "size": np.log(10 + i), "beta_250": rng.normal(), "mom_12_1": rng.normal(),
                         "vol_20": rng.normal(), "bp": rng.normal(), "turn_20": rng.normal()})
    data = pd.DataFrame(rows)
    data["fwd_rank"] = data.groupby("date").fwd_ret.rank(pct=True) - 0.5
    z = ret * 0
    f = z.astype(bool)
    mkt = Market(ret, z, ret, ~f, f, f)
    fcols = ["A", "B", "size", "beta", "momentum", "volatility", "value", "liquidity"]
    fret = pd.DataFrame(rng.normal(0, 0.01, (len(days), len(fcols))), index=days, columns=fcols)
    resid = pd.DataFrame(rng.normal(0, 0.015, (len(days), n)), index=days, columns=codes)
    return data, ret, mkt, fret, resid


def test_end_to_end_on_synthetic_data(world, monkeypatch):
    data, ret, mkt, fret, resid = world
    monkeypatch.setitem(CFG["split"], "is_start", "2016-01-01")
    monkeypatch.setitem(CFG["model"], "min_train_months", 12)
    sel = walk_forward_selection(data, ["ep", "bp", "size"], min_months=12)
    assert (sel["ep"] == 1).mean() > 0.5                           # the planted factor is found walk-forward
    assert (sel.iloc[:12] == 0).all().all()                        # nothing selected before enough history
    score = predict_ridge(data, sel)
    pc = {**CFG["portfolio"], "risk_aversion": 5.0, "industry_dev": 0.2, "size_dev": 5.0}
    bt, info = optimised_backtest(score, data, fret, resid, mkt, pc, progress=False)
    # account identities on every execution
    for d, h in bt["holdings"].items():
        assert h.sum() + bt["cash"][d] == pytest.approx(1.0, abs=1e-9)
    L = bt["ledger"]
    assert np.allclose(L.nav_pre - L.cost, L.nav_post) and (L.cash_after >= -1e-12).all()
    assert info.turnover_exempt.iloc[0] and not info.turnover_exempt.iloc[1:].any()
    # holdings table <-> backtest log, one to one
    monkeypatch.setattr(H, "_names", lambda: pd.Series(dtype=str))
    t = H.holdings_table(bt, data, score, ret.index)
    ex = t[t.executed_weight.notna()]
    for d, h in bt["holdings"].items():
        sub = ex[(ex.exec_date == d.date()) & (ex.code != "CASH")].set_index("code").executed_weight
        assert np.allclose(sub.reindex(h.index).values, h.values)
    # attribution identities
    sig_of = {e: ret.index[ret.index.searchsorted(e) - 1] for e in bt["holdings"]}
    bench = pd.Series(0.0004, index=ret.index)
    rec = reconcile(bt["holdings"], sig_of, data, ret, bt["ret"], bench, bt["cost"])
    assert np.allclose(rec[["brinson", "cost", "execution", "benchmark"]].sum(axis=1), rec.actual)
    br = brinson(bt["holdings"], sig_of, data, ret)
    assert np.allclose(br.groupby("date")[["allocation", "selection", "interaction"]].sum().sum(axis=1).values,
                       rec.brinson.values)
    cc = constraint_check(bt["holdings"], sig_of, data, bt["turnover"], pc)
    assert cc.ok_cap.all()
    p = perf(bt["ret"], bench.reindex(bt["ret"].index))
    assert np.isfinite(p["信息比率"]) and p["最大回撤"] <= 0
