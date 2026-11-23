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


def _fwd(tmp_path, monkeypatch):
    from mf import forward as F
    monkeypatch.setattr(F, "FWD", tmp_path)
    monkeypatch.setattr(F, "REGISTRY", tmp_path / "registry.csv")
    return F


RULES = {"version": "v2", "rules_sha": "x", "history_end": "2026-10-30", "frozen_at_utc": "2026-11-20T00:00:00+00:00"}


def test_registration_after_execution_is_late_and_backfill_cannot_fix_it(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    F = _fwd(tmp_path, monkeypatch)
    s0, s1 = pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-30")
    tg = {s0: pd.Series({"a": 1.0}), s1: pd.Series({"a": 0.5, "b": 0.5})}
    # registered on 2026-11-25: s0 (executes 2026-11-02) is late, s1 (executes 2026-12-01 09:30 CST) is valid
    reg = F.register(tg, RULES, True, now=datetime(2026, 11, 25, tzinfo=timezone.utc))
    st = dict(zip(reg.signal_date, reg.status))
    assert st == {"2026-10-30": "late_registration", "2026-11-30": "valid_preregistered"}
    # a later run cannot rewrite a row; a changed target is flagged
    tg[s1] = pd.Series({"a": 0.6, "b": 0.4})
    reg = F.register(tg, RULES, True, now=datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert len(reg) == 2 and reg.attrs["mismatch"] == ["2026-11-30"]
    # rules frozen AFTER the registration -> late even if before execution
    F2 = _fwd(tmp_path / "b", monkeypatch); (tmp_path / "b").mkdir()
    reg2 = F2.register({s1: pd.Series({"a": 1.0})}, {**RULES, "history_end": "2026-11-30", "frozen_at_utc": "2026-11-28T00:00:00+00:00"},
                       True, now=datetime(2026, 11, 27, tzinfo=timezone.utc))
    assert reg2.status.tolist() == ["late_registration"]


def test_strict_forward_days_need_preregistration_hash_and_git_time(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    F = _fwd(tmp_path, monkeypatch)
    days = pd.bdate_range("2026-10-26", "2026-12-04")
    ret = pd.Series(0.001, index=days)
    s0, s1 = pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-30")
    tg = {s0: pd.Series({"a": 1.0}), s1: pd.Series({"b": 1.0})}
    reg = F.register(tg, RULES, True, now=datetime(2026, 11, 25, tzinfo=timezone.utc))
    early = lambda h: datetime(2026, 11, 25, 1, tzinfo=timezone.utc)
    sd = F.strict_days(ret, tg, reg, True, True, "2026-10-30", commit_time=early)
    assert not sd.loc[:"2026-11-30", "strict"].any()                # driven by the late-registered s0
    assert not sd.loc["2026-12-01", "strict"]                       # execution day still mixes s0 (open->VWAP)
    assert sd.loc["2026-12-02":, "strict"].all()                    # only s1 now: strict
    no_git = F.strict_days(ret, tg, reg, True, True, "2026-10-30", commit_time=lambda h: None)
    assert not no_git.strict.any()                                  # no independent timestamp -> not strict
    bad_rules = F.strict_days(ret, tg, reg, False, True, "2026-10-30", commit_time=early)
    assert not bad_rules.strict.any()
    fs = F.forward_summary(ret.loc[:"2026-10-30"], ret * 0, tg, {**RULES}, reg, [], True, commit_time=early)
    assert fs["strict_days"] == 0 and fs["performance_status"].startswith("尚未开始")


def test_freeze_refuses_dirty_tree(monkeypatch, tmp_path):
    from mf import forward as F
    monkeypatch.setattr(F, "_git", lambda *a: " M src/mf/backtest.py" if a[0] == "status" else "")
    monkeypatch.setattr(F, "RULES_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        F.freeze_new_version("Ridge", 5.0, "2026-10-30", "test", {}, None)


def test_history_window_is_unchanged_by_later_data():
    from mf.metrics import perf
    idx = pd.bdate_range("2022-01-03", "2026-12-31")
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    b = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    trunc, full = r.loc[:"2026-10-30"], r
    a1 = perf(trunc.loc["2022-01-01":"2026-10-31"], b)
    a2 = perf(full.loc["2022-01-01":"2026-10-31"], b)
    assert a1 == a2


# ---------------------------------------------------------------- benchmark & drawdown
def test_benchmark_uses_previous_trading_day_cap_on_reentry(tmp_path, monkeypatch):
    from mf import panel
    d = pd.bdate_range("2021-01-28", periods=30)              # spans Jan, Feb, Mar
    rows = []
    for code, cap0 in [("a", 100.0), ("b", 100.0)]:
        cap = cap0
        for x in d:
            cap = cap * (1.5 if (code == "b" and x.month == 2) else 1.0)   # b's cap jumps while out of the index
            rows.append({"code": code, "date": x, "ret": 0.01 if code == "a" else 0.02, "float_mcap": cap})
    long = pd.DataFrame(rows)
    uni = pd.DataFrame({"month": ["2021-01", "2021-01", "2021-02", "2021-03", "2021-03"],
                        "code": ["a", "b", "a", "a", "b"]})              # b leaves in Feb, re-enters in Mar
    (tmp_path / "universe_monthly.csv").write_text(uni.to_csv(index=False))
    monkeypatch.setattr(panel, "RAW", tmp_path)
    bench, cov = panel.reconstruct_benchmark(long, return_coverage=True)
    first_mar = d[d.month == 3][0]
    prev = d[d < first_mar][-1]
    capb = long[(long.code == "b") & (long.date == prev)].float_mcap.iloc[0]
    capa = long[(long.code == "a") & (long.date == prev)].float_mcap.iloc[0]
    expected = (capa * 0.01 + capb * 0.02) / (capa + capb)     # b weighted by its cap on the previous trading day
    assert bench[first_mar] == pytest.approx(expected)
    assert (cov.coverage <= 1).all()


def test_max_drawdown_counts_a_loss_on_the_first_day():
    from mf.metrics import perf
    r = pd.Series([-0.20, 0.05, 0.05], index=pd.bdate_range("2021-01-04", periods=3))
    p = perf(r, r * 0)
    assert p["最大回撤"] == pytest.approx(-0.20)
    assert p["超额最大回撤"] == pytest.approx(-0.20)


# ---------------------------------------------------------------- stale positions, ledger, turnover policy, risk
def test_permanently_untraded_holding_is_flagged_and_can_be_haircut():
    idx = pd.bdate_range("2021-01-01", periods=80)
    ret = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    mkt = market(ret)
    mkt.traded.loc[idx[5]:, "a"] = False                           # 'a' stops trading for good (delisting)
    mkt.no_sell.loc[idx[5]:, "a"] = True
    mkt.no_buy.loc[idx[5]:, "a"] = True
    bt = run_backtest({idx[0]: pd.Series({"a": 0.5, "b": 0.5})}, mkt, cost_on=False, stale_days=20)
    assert len(bt["stale"]) == 1 and bt["stale"].code.iloc[0] == "a"   # audit exception, logged once
    assert bt["nav"].iloc[-1] == pytest.approx(1.0)                      # no haircut by default ...
    hc = run_backtest({idx[0]: pd.Series({"a": 0.5, "b": 0.5})}, mkt, cost_on=False, stale_days=20, stale_haircut=0.5)
    assert hc["nav"].iloc[-1] == pytest.approx(0.75)                    # ... conservative scenario writes it down


def test_every_execution_reconciles_in_the_ledger():
    idx = pd.bdate_range("2021-01-01", periods=10)
    rng = np.random.default_rng(7)
    ret = pd.DataFrame(rng.normal(0, 0.02, (10, 3)), index=idx, columns=list("abc"))
    nb = ret.astype(bool); nb.loc[idx[4], "c"] = True
    bt = run_backtest({idx[0]: pd.Series({"a": .4, "b": .3, "c": .3}), idx[3]: pd.Series({"a": .1, "b": .2, "c": .7})},
                      market(ret, no_buy=nb, r1=ret * 0.3, r2=ret * 0.7))
    L = bt["ledger"]
    assert np.allclose(L.nav_pre - L.cost, L.nav_post)
    assert np.allclose(L.cash_before + L.sells - L.buys - L.cost, L.cash_after)
    assert (L.cash_after >= -1e-12).all()
    ident(bt)


def _toy_opt():
    names = pd.Index(["a", "b", "c"])
    X = pd.DataFrame({"ind1": 1.0, "size": [0.5, 0.0, -0.5], "beta": 0.0, "momentum": 0.0, "volatility": 0.0,
                      "value": 0.0, "liquidity": 0.0}, index=names)
    F = pd.DataFrame(np.eye(X.shape[1]) * 1e-4, index=X.columns, columns=X.columns)
    return names, X, F, pd.Series(1e-3, index=names)


def test_turnover_limit_applies_unless_book_is_mostly_cash():
    names, X, F, D = _toy_opt()
    pc = {**CFG["portfolio"], "risk_aversion": 0.01, "turnover_limit": 0.1, "size_dev": 10, "industry_dev": 1}
    alpha = pd.Series([0.05, 0.0, -0.05], index=names)
    b = pd.Series(1 / 3, index=names)
    ok = pd.Series(True, index=names)
    w, info = optimize(alpha, b, X, F, D, None, ok, pc)                       # initial build: exempt
    assert info["turnover_exempt"]
    w, info = optimize(alpha, b, X, F, D, pd.Series({"a": .2, "b": .2}), ok, pc)  # 60% cash: exempt, reported
    assert info["turnover_exempt"] and info["cash_at_signal"] == pytest.approx(0.6)
    w, info = optimize(alpha, b, X, F, D, pd.Series({"a": .30, "b": .35, "c": .35}), ok, pc)
    assert not info["turnover_exempt"] and info["relax"] == 1.0
    assert info["planned_turnover"] <= 0.1 + 1e-4


def test_risk_model_inputs_are_finite_psd_and_new_factors_not_riskless():
    from mf.risk import risk_model_at
    d = pd.Timestamp("2021-06-30")
    days = pd.bdate_range("2020-06-01", d)
    codes = [f"s{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"date": d, "code": codes, "industry": ["X"] * 15 + ["NEW"] * 15, "mcap": rng.uniform(1, 5, 30),
                         **{c: rng.normal(size=30) for c in ["size", "beta_250", "mom_12_1", "vol_20", "bp", "turn_20"]}})
    cols = ["X", "size", "beta", "momentum", "volatility", "value", "liquidity"]
    fret = pd.DataFrame(rng.normal(0, 0.01, (len(days), len(cols))), index=days, columns=cols)
    fret["NEW"] = np.nan
    fret.loc[days[-10:], "NEW"] = 0.02                                       # only 10 days of history
    resid = pd.DataFrame(rng.normal(0, 0.02, (len(days), 30)), index=days, columns=codes)
    X, F, D = risk_model_at(d, data, fret, resid)
    assert np.isfinite(F.values).all() and np.linalg.eigvalsh(F.values).min() >= -1e-12
    assert F.loc["NEW", "NEW"] > 0                                           # not treated as riskless
    assert (D > 0).all()


def test_mid_month_data_end_is_not_a_rebalance_date():
    from mf.factors import rebalance_dates
    d = pd.bdate_range("2026-09-01", "2026-11-11")
    assert rebalance_dates(d, "2026-11-11")[-1] == pd.Timestamp("2026-10-30")   # November incomplete
    d2 = pd.bdate_range("2026-09-01", "2026-11-30")
    assert rebalance_dates(d2, "2026-11-30")[-1] == pd.Timestamp("2026-11-30")
    d3 = pd.bdate_range("2026-09-01", "2026-11-27")                             # holiday-shortened end
    assert rebalance_dates(d3, "2026-11-30")[-1] == pd.Timestamp("2026-11-27")
