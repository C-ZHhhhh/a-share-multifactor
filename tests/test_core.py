import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from mf import models
from mf.backtest import Market, run_backtest
from mf.factors import mad_winsorize, neutralize, rebalance_dates


def test_mad_winsorize_clips_outliers():
    x = pd.Series([1, 2, 3, 4, 1000.0])
    assert mad_winsorize(x).max() < 1000


def test_neutralize_removes_industry_and_size():
    rng = np.random.default_rng(0)
    n = 500
    ind = pd.Series(rng.choice(list("ABCD"), n))
    size = pd.Series(rng.normal(size=n))
    y = pd.Series(2 * size + ind.map({"A": 1, "B": -1, "C": 0, "D": 3}) + rng.normal(size=n))
    r = neutralize(y, ind, size)
    assert abs(np.corrcoef(r, size)[0, 1]) < 1e-8
    assert r.groupby(ind).mean().abs().max() < 1e-8


def test_rebalance_dates_are_month_ends():
    d = pd.bdate_range("2020-01-01", "2020-03-31")
    r = rebalance_dates(d)
    assert list(r.strftime("%m-%d")) == ["01-31", "02-28", "03-31"]


def _toy_market(days, codes, ret=0.01, no_buy=None):
    idx = pd.bdate_range("2021-01-01", periods=days)
    R = pd.DataFrame(ret, index=idx, columns=codes)
    zero = R * 0
    nb = zero.astype(bool) if no_buy is None else no_buy
    return Market(R, zero, R, ~zero.astype(bool) | True, nb, zero.astype(bool))


def test_backtest_costs_and_returns():
    mkt = _toy_market(5, ["a", "b"])
    sig = mkt.ret.index[0]
    tgt = {sig: pd.Series({"a": 0.5, "b": 0.5})}
    with_cost = run_backtest(tgt, mkt)
    no_cost = run_backtest(tgt, mkt, cost_on=False)
    assert no_cost["nav"].iloc[-1] == pytest.approx(1.01 ** 4)        # 4 days held after exec
    assert with_cost["nav"].iloc[-1] < no_cost["nav"].iloc[-1]
    assert with_cost["turnover"].iloc[0] == pytest.approx(0.5)


def test_backtest_blocks_limit_up_buys():
    idx = pd.bdate_range("2021-01-01", periods=3)
    nb = pd.DataFrame(False, index=idx, columns=["a", "b"])
    nb.loc[idx[1], "a"] = True                                      # 'a' one-price limit-up on exec day
    mkt = _toy_market(3, ["a", "b"], no_buy=nb)
    bt = run_backtest({idx[0]: pd.Series({"a": 0.5, "b": 0.5})}, mkt, cost_on=False)
    h = bt["holdings"][idx[1]]
    assert "a" not in h.index and h["b"] == pytest.approx(0.5)   # the blocked half stays in cash


def test_optimiser_callback_sees_actual_drifted_weights():
    # 'a' rises 10%/day, 'b' flat: after the first rebalance to 50/50 the actual weights drift towards 'a'
    idx = pd.bdate_range("2021-01-01", periods=4)
    R = pd.DataFrame({"a": 0.10, "b": 0.0}, index=idx)
    zero = R * 0
    mkt = Market(R, zero, zero, ~zero.astype(bool) | True, zero.astype(bool), zero.astype(bool))
    seen = {}

    def target_fn(d, w_actual):
        seen[d] = w_actual.copy()
        return pd.Series({"a": 0.5, "b": 0.5})

    run_backtest(target_fn, mkt, signal_dates=[idx[0], idx[2]], cost_on=False)
    assert seen[idx[0]].empty                                    # first decision: no holdings yet
    # exec on idx[1] (open->VWAP and VWAP->close returns are 0): 50/50 at the close; on idx[2] 'a' gains 10%
    w = seen[idx[2]]
    assert w["a"] == pytest.approx(0.55 / 1.05) and w["b"] == pytest.approx(0.5 / 1.05)


def test_walk_forward_has_no_lookahead(monkeypatch):
    months = pd.date_range("2016-01-31", periods=40, freq="ME")
    # label of month k is realised the day after month k+1's rebalance
    label = {m: (months[k + 1] + pd.Timedelta(days=1)) if k + 1 < len(months) else pd.NaT
             for k, m in enumerate(months)}
    rows = [{"date": m, "code": f"s{i}", "tradable": True, "z_x": np.random.rand(), "fwd_rank": 0.0,
             "label_date": label[m]} for m in months for i in range(30)]
    data = pd.DataFrame(rows)
    seen = {}

    class Spy:
        def fit(self, X, y):
            seen["n_train"] = len(X)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setitem(models.CFG["model"], "min_train_months", 24)
    out = models._walk_forward(data, ["z_x"], Spy, refit_every=1)
    first = out.index.get_level_values(0).min()
    # first prediction month index = 24 + GAP; training uses months 0..(24+GAP-1-GAP) = 24 months
    assert first == months[24 + models.GAP]
    assert seen["n_train"] == 30 * (len(months) - 1 - models.GAP)


def test_label_window_excludes_labels_realised_after_period_end():
    from mf.dataset import label_window
    d = pd.DataFrame({"date": pd.to_datetime(["2021-11-30", "2021-12-31"]),
                      "label_date": pd.to_datetime(["2022-01-04", "2022-02-07"])})
    d.loc[0, "label_date"] = pd.Timestamp("2021-12-31")
    assert label_window(d, "2016-01-01", "2021-12-31").tolist() == [True, False]


# ---------------------------------------------------------------- point-in-time fundamentals
from mf.panel import attach_fundamentals


def _px(f_after_split=1.0):
    d = pd.bdate_range("2021-04-26", periods=6)
    close = [10, 10, 10, 5, 5, 5] if f_after_split != 1.0 else [10] * 6
    f = [1, 1, 1, f_after_split, f_after_split, f_after_split]
    return pd.DataFrame({"code": "sh.600000", "date": d, "close": close, "f": f})


def _fund(ann):
    return pd.DataFrame({"code": ["sh.600000"], "period": [pd.Timestamp("2021-03-31")],
                         "ann_date": [ann], "eps_ttm": [1.0], "sps_ttm": [5.0], "bvps": [8.0],
                         "roe": [0.1], "gpm": [30.0], "np_yoy": [10.0], "rev_yoy": [5.0]})


def test_report_not_used_on_announcement_day():
    px = _px()
    ann = px.date[2]
    out = attach_fundamentals(px, _fund(ann)).set_index("date")
    assert np.isnan(out.loc[px.date[2], "ep"])            # same day: not yet available
    assert out.loc[px.date[3], "ep"] == pytest.approx(0.1)  # next day: EP = 1 / 10


def test_per_share_values_adjusted_for_bonus_shares():
    px = _px(f_after_split=2.0)                             # 10-for-10 bonus issue on day 3
    out = attach_fundamentals(px, _fund(px.date[0])).set_index("date")
    # price halves, share count doubles: EP must be unchanged
    assert out.loc[px.date[2], "ep"] == pytest.approx(out.loc[px.date[4], "ep"])


# ---------------------------------------------------------------- optimiser
def test_optimizer_cap_is_active_not_absolute():
    """A 10% benchmark weight must be holdable: cap is b_i + 3%, not an absolute 3%."""
    from mf.optimizer import optimize
    n = 20
    names = [f"s{i}" for i in range(n)]
    b = pd.Series([0.10] + [0.9 / (n - 1)] * (n - 1), index=names)
    cols = ["ind_A", "size", "beta", "momentum", "volatility", "value", "liquidity"]
    X = pd.DataFrame(0.0, index=names, columns=cols)
    X["ind_A"] = 1.0
    F = pd.DataFrame(np.eye(len(cols)) * 1e-3, index=cols, columns=cols)
    D = pd.Series(0.01, index=names)
    alpha = pd.Series(0.0, index=names)
    w, info = optimize(alpha, b, X, F, D, None, pd.Series(True, index=names))
    assert w["s0"] == pytest.approx(0.10, abs=1e-3)
    assert info["ex_ante_te"] < 1e-3


# ---------------------------------------------------------------- blocked trades & walk-forward selection
from mf.backtest import _fill


def test_blocked_buy_leaves_cash_and_never_flips_a_sale():
    w = np.array([0.5, 0.5, 0.0])
    tgt = np.array([0.2, 0.3, 0.5])           # sell a, sell b, buy c ... but c is limit-up
    nb = np.array([False, False, True]); ns = np.zeros(3, bool)
    new = _fill(w, tgt, nb, ns)
    assert new[2] == 0.0                       # blocked buy not executed
    assert new[0] <= w[0] and new[1] <= w[1]   # the planned sales stay sales (no scaling up)
    assert new.sum() == pytest.approx(0.5)     # the rest is cash


def test_blocked_sell_scales_buys_down_not_into_sales():
    w = np.array([0.6, 0.2, 0.2])
    tgt = np.array([0.2, 0.4, 0.4])           # sell a (limit-down), buy b and c
    nb = np.zeros(3, bool); ns = np.array([True, False, False])
    new = _fill(w, tgt, nb, ns)
    assert new[0] == pytest.approx(0.6)        # could not sell
    assert new.sum() == pytest.approx(1.0)
    assert (new[1:] >= w[1:] - 1e-12).all()   # buys shrink, never turn into sells


def test_walk_forward_selection_ignores_unrealised_labels(monkeypatch):
    from mf import factor_test
    rng = np.random.default_rng(0)
    months = pd.date_range("2016-01-31", periods=30, freq="ME")
    label = {m: months[k + 1] + pd.Timedelta(days=1) if k + 1 < len(months) else pd.NaT for k, m in enumerate(months)}
    rows = []
    for m in months:
        x = rng.normal(size=60)
        rows += [{"date": m, "code": f"s{i}", "tradable": True, "z_size": x[i], "z_ep": rng.normal(),
                  "fwd_ret": 0.5 * x[i] + rng.normal(), "label_date": label[m]} for i in range(60)]
    data = pd.DataFrame(rows)
    monkeypatch.setitem(factor_test.CFG["split"], "is_start", "2016-01-01")
    sel = factor_test.walk_forward_selection(data, ["size", "ep"], min_months=6)
    cut = months[24]
    bad = data.copy()
    bad.loc[bad.label_date >= cut, "fwd_ret"] = -bad.loc[bad.label_date >= cut, "z_size"] * 5   # flip the future
    sel2 = factor_test.walk_forward_selection(bad, ["size", "ep"], min_months=6)
    assert sel.loc[:cut].equals(sel2.loc[:cut])          # decisions up to `cut` cannot see labels realised later
    assert sel.loc[months[10], "size"] == 1
