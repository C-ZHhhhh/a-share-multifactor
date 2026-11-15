"""Execution and accounting edge cases: first build, blocked buys / sells, residual cash, index exits,
books that are not fully invested, and the attribution identities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from mf import holdings as H
from mf.attribution import brinson, reconcile
from mf.backtest import Market, run_backtest
from mf.config import CFG
from mf.optimizer import optimize

CB = CFG["costs"]["commission"] + CFG["costs"]["slippage"]


def market(ret: pd.DataFrame, no_buy=None, no_sell=None, r1=None, r2=None) -> Market:
    z = ret * 0
    f = z.astype(bool)
    r1 = z if r1 is None else r1
    r2 = ret if r2 is None else r2          # exec day: open->VWAP = r1, VWAP->close = r2
    return Market(ret, r1, r2, ~f, f if no_buy is None else no_buy, f if no_sell is None else no_sell)


def ident(bt):
    """holdings + cash == 1 after every execution; cash never negative; nav consistent with returns."""
    for d, h in bt["holdings"].items():
        assert h.sum() + bt["cash"][d] == pytest.approx(1.0, abs=1e-12)
        assert bt["cash"][d] >= -1e-12
    assert (1 + bt["ret"]).prod() == pytest.approx(bt["nav"].iloc[-1])


def test_first_build_from_cash_uses_all_cash_after_costs():
    idx = pd.bdate_range("2021-01-01", periods=4)
    mkt = market(pd.DataFrame(0.0, index=idx, columns=["a", "b"]))
    seen = {}

    def fn(d, w):
        seen[d] = w
        return pd.Series({"a": 0.5, "b": 0.5})

    bt = run_backtest(fn, mkt, signal_dates=[idx[0]])
    assert seen[idx[0]].empty                                   # optimiser starts from an empty book
    e = idx[1]
    assert bt["cash"][e] == pytest.approx(0.0, abs=1e-12)        # buys scaled so that cash pays the costs
    assert bt["holdings"][e].sum() == pytest.approx(1.0)        # fully invested (fractions of post-trade NAV)
    assert bt["turnover"][e] == pytest.approx(0.5 / (1 + CB))   # executed buys = cash / (1 + cost rate)
    assert bt["cost"][e] == pytest.approx(CB / (1 + CB))
    assert bt["nav"][e] == pytest.approx(1 / (1 + CB))
    ident(bt)


def test_blocked_buy_stays_in_cash_and_is_deployed_next_time():
    idx = pd.bdate_range("2021-01-01", periods=6)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    nb = ret.astype(bool)
    nb.loc[idx[1], "b"] = True                                  # b one-price limit-up on the first exec day
    mkt = market(ret, no_buy=nb)
    tg = pd.Series({"a": 0.5, "b": 0.5})
    bt = run_backtest({idx[0]: tg, idx[3]: tg}, mkt, cost_on=False)
    e1, e2 = idx[1], idx[4]
    assert "b" not in bt["holdings"][e1].index
    assert bt["cash"][e1] == pytest.approx(0.5)                 # the unspent half is cash, a is NOT scaled up
    assert bt["holdings"][e1]["a"] == pytest.approx(0.5)
    assert bt["cash"][e2] == pytest.approx(0.0, abs=1e-12)       # next rebalance deploys it
    assert bt["at_signal"][idx[3]].sum() == pytest.approx(0.5)   # optimiser saw the 50% book (cash implied)
    ident(bt)


def test_blocked_sell_keeps_position_and_shrinks_buys_without_borrowing():
    idx = pd.bdate_range("2021-01-01", periods=6)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b", "c"])
    ns = ret.astype(bool)
    ns.loc[idx[4], "a"] = True                                  # a one-price limit-down on the second exec day
    mkt = market(ret, no_sell=ns)
    bt = run_backtest({idx[0]: pd.Series({"a": 1.0}), idx[3]: pd.Series({"b": 0.5, "c": 0.5})}, mkt)
    h = bt["holdings"][idx[4]]
    assert h["a"] == pytest.approx(bt["pre_trade"][idx[4]]["a"])  # could not sell
    assert h.get("b", 0) == pytest.approx(0.0, abs=1e-12)          # nothing to buy with: no borrowing
    assert bt["cash"][idx[4]] == pytest.approx(0.0, abs=1e-12)
    ident(bt)


def test_index_exit_is_sold_unless_suspended():
    idx = pd.bdate_range("2021-01-01", periods=6)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    traded_ns = ret.astype(bool)
    traded_ns.loc[idx[4], "a"] = True                           # 'a' left the index and is suspended
    tg1, tg2 = pd.Series({"a": 0.5, "b": 0.5}), pd.Series({"b": 1.0})
    sold = run_backtest({idx[0]: tg1, idx[3]: tg2}, market(ret), cost_on=False)
    stuck = run_backtest({idx[0]: tg1, idx[3]: tg2}, market(ret, no_sell=traded_ns), cost_on=False)
    assert "a" not in sold["holdings"][idx[4]].index
    assert stuck["holdings"][idx[4]]["a"] == pytest.approx(0.5)  # stuck, kept at its weight
    assert stuck["holdings"][idx[4]]["b"] == pytest.approx(0.5)  # buy of b shrunk to the cash available
    ident(stuck)


def test_optimiser_counts_forced_index_exit_in_turnover():
    names = pd.Index(["a", "b", "c"])
    X = pd.DataFrame({"ind1": 1.0, "size": [0.5, 0.0, -0.5], "beta": 0.0, "momentum": 0.0, "volatility": 0.0,
                      "value": 0.0, "liquidity": 0.0}, index=names)
    F = pd.DataFrame(np.eye(X.shape[1]) * 1e-4, index=X.columns, columns=X.columns)
    D = pd.Series(1e-3, index=names)
    w0 = pd.Series({"a": 0.3, "b": 0.3, "z": 0.4})              # 'z' has left the universe
    pc = {**CFG["portfolio"], "risk_aversion": 5.0, "turnover_limit": 1.0, "size_dev": 10, "industry_dev": 1}
    w, info = optimize(pd.Series(0.0, index=names), pd.Series(1 / 3, index=names), X, F, D, w0,
                       pd.Series(True, index=names), pc)
    assert w.sum() == pytest.approx(1.0)
    assert info["planned_turnover"] >= 0.2 - 1e-6               # selling z (40%) alone is 20% one-way


def test_brinson_identity_with_cash():
    d0, d1, d2 = pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-01"), pd.Timestamp("2021-02-02")
    data = pd.DataFrame({"date": d0, "code": ["a", "b", "c"], "industry": ["X", "X", "Y"],
                         "mcap": [2.0, 1.0, 1.0]})
    daily = pd.DataFrame({"a": [0.0, 0.03], "b": [0.0, -0.01], "c": [0.0, 0.02]}, index=[d1, d2])
    hold = {d1: pd.Series({"a": 0.5, "c": 0.3})}                  # 20% cash
    br = brinson(hold, {d1: d0}, data, daily)
    wb = pd.Series({"a": 0.5, "b": 0.25, "c": 0.25})
    r = daily.loc[d2]
    expected = (hold[d1] * r.reindex(hold[d1].index)).sum() - (wb * r).sum()
    assert br[["allocation", "selection", "interaction"]].sum().sum() == pytest.approx(expected)
    assert "现金" in set(br.industry)


def test_reconcile_components_add_up_to_actual():
    idx = pd.bdate_range("2021-01-01", periods=8)
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.01, (8, 3)), index=idx, columns=["a", "b", "c"])
    nb = ret.astype(bool)
    nb.loc[idx[1], "c"] = True                                   # some cash appears
    mkt = market(ret, no_buy=nb)
    bt = run_backtest({idx[0]: pd.Series({"a": 0.4, "b": 0.3, "c": 0.3}),
                       idx[4]: pd.Series({"a": 0.2, "b": 0.4, "c": 0.4})}, mkt)
    data = pd.DataFrame([{"date": s, "code": k, "industry": "X" if k != "c" else "Y", "mcap": 1.0}
                         for s in (idx[0], idx[4]) for k in "abc"])
    sig = {idx[1]: idx[0], idx[5]: idx[4]}
    bench = (ret * (1 / 3)).sum(axis=1)
    rec = reconcile(bt["holdings"], sig, data, ret, bt["ret"], bench, bt["cost"])
    parts = rec[["brinson", "cost", "execution", "benchmark"]].sum(axis=1)
    assert np.allclose(parts, rec.actual)
    br = brinson(bt["holdings"], sig, data, ret)
    assert np.allclose(br.groupby("date")[["allocation", "selection", "interaction"]].sum().sum(axis=1).values,
                       rec.brinson.values)


def test_holdings_table_separates_position_concepts(monkeypatch):
    idx = pd.bdate_range("2021-01-01", periods=6)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    nb = ret.astype(bool)
    nb.loc[idx[1], "b"] = True
    tg = pd.Series({"a": 0.5, "b": 0.5})
    bt = run_backtest({idx[0]: tg, idx[5]: tg}, market(ret, no_buy=nb), cost_on=False)
    data = pd.DataFrame([{"date": s, "code": k, "industry": "X", "mcap": 1.0} for s in (idx[0], idx[5]) for k in "ab"])
    scores = pd.Series(1.0, index=pd.MultiIndex.from_product([[idx[0], idx[5]], ["a", "b"]]))
    monkeypatch.setattr(H, "_names", lambda: pd.Series({"a": "A", "b": "B"}))
    t = H.holdings_table(bt, data, scores, idx)
    first = t[t.signal_date == idx[0].date()].set_index("code")
    assert first.loc["b", "status"] == "买入受阻" and first.loc["b", "executed_weight"] == 0
    assert first.loc["CASH", "executed_weight"] == pytest.approx(0.5)
    assert first.loc["a", "target_weight"] == pytest.approx(0.5)
    last = t[t.signal_date == idx[5].date()].set_index("code")
    assert (last.drop("CASH").status == "待执行").all()           # latest signal not executed yet
    assert last.loc["a", "weight_at_signal"] == pytest.approx(0.5)
    assert last.loc["CASH", "weight_at_signal"] == pytest.approx(0.5)


def test_forward_registry_is_append_only_and_flags_changes(tmp_path, monkeypatch):
    from mf import forward as F
    monkeypatch.setattr(F, "FWD", tmp_path)
    monkeypatch.setattr(F, "REGISTRY", tmp_path / "registry.csv")
    s0, s1, s2 = pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-30")
    tg = {s0: pd.Series({"a": 1.0}), s1: pd.Series({"a": 0.5, "b": 0.5})}
    reg = F.register(tg, "2026-10-30", compliant=True)
    assert list(reg.signal_date) == ["2026-10-30"]              # first holding period after history_end
    tg[s2] = pd.Series({"b": 1.0})
    tg[s1] = pd.Series({"a": 0.6, "b": 0.4})                    # recomputed differently later (data revision)
    reg = F.register(tg, "2026-10-30", compliant=True)
    assert list(reg.signal_date) == ["2026-10-30", "2026-11-30"]
    assert reg.attrs["mismatch"] == ["2026-10-30"]              # original row kept, change flagged
    reg2 = F.register({**tg, pd.Timestamp("2026-12-31"): pd.Series({"a": 1.0})}, "2026-10-30", compliant=False)
    assert len(reg2) == 2                                       # nothing registered by a non-compliant run
