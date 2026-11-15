"""Daily event-driven backtest with realistic A-share frictions.

Timeline for a rebalance signal generated at the close of day t:
  t+1 open..VWAP : old portfolio earns close(t) -> VWAP(t+1)
  t+1 VWAP       : trade to target; buys blocked if suspended / one-price limit-up,
                   sells blocked if suspended / one-price limit-down (weight kept);
                   commission + slippage both sides, stamp duty on sells
  t+1 VWAP..close: new portfolio earns VWAP(t+1) -> close(t+1)
  afterwards     : weights drift with daily total returns

Targets can be a fixed dict {signal_date: weights} or a callable target_fn(signal_date, w_actual) that is
called at the close of each signal date with the portfolio's ACTUAL drifted weights (after blocked trades),
so a path-dependent optimiser (turnover limit, no-buy constraint) sees the real current holdings.
The account holds position values and cash (see run_backtest). When buys are blocked the unspent weight stays
in cash (0 return) until the next rebalance; when sells are blocked the planned buys are scaled down. A trade
is never turned around: a stock the target wants to sell is never bought, and blocked names are never traded
(see _fill). Trading costs are paid from cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG
from .panel import to_wide


@dataclass
class Market:
    ret: pd.DataFrame
    r_open_to_vwap: pd.DataFrame
    r_vwap_to_close: pd.DataFrame
    traded: pd.DataFrame
    no_buy: pd.DataFrame
    no_sell: pd.DataFrame

    @classmethod
    def from_long(cls, long: pd.DataFrame) -> "Market":
        W = {c: to_wide(long, c) for c in ["ret", "adj_close", "close", "vwap", "traded",
                                           "one_price_up", "one_price_down"]}
        traded = W["traded"].fillna(False).astype(bool)
        adj_vwap = W["vwap"] * W["adj_close"] / W["close"]
        prev_adj = W["adj_close"].shift()
        r1 = (adj_vwap / prev_adj - 1).where(traded, 0.0).fillna(0.0)
        r2 = (W["adj_close"] / adj_vwap - 1).where(traded, 0.0).fillna(0.0)
        ret = W["ret"].fillna(0.0)
        return cls(ret, r1, r2, traded,
                   ~traded | W["one_price_up"].fillna(False).astype(bool),
                   ~traded | W["one_price_down"].fillna(False).astype(bool))


def _fill(w: np.ndarray, tgt: np.ndarray, no_buy: np.ndarray, no_sell: np.ndarray) -> np.ndarray:
    """Post-trade weights given current weights w, target tgt and per-stock trading blocks.
    Blocked names keep w. If the blocked names now take more room than planned (a sell was blocked), the
    remaining BUYS are scaled down (never below w, so no buy becomes a sell); if that is not enough, other
    free names are scaled down, but limit-down / suspended names never below w. If they take less room
    (a buy was blocked) the leftover is held as cash instead of being spread over other names."""
    blocked = ((tgt > w) & no_buy) | ((tgt < w) & no_sell)
    new = np.where(blocked, w, tgt)
    excess = new.sum() - 1.0
    if excess > 1e-12:
        buy = np.where(~blocked & (new > w), new - w, 0.0)
        cut = min(excess, buy.sum())
        if cut > 0:
            new = new - buy * (cut / buy.sum())
            excess -= cut
        if excess > 1e-12:
            floor = np.where(no_sell, np.minimum(w, new), 0.0)
            slack = np.where(~blocked, new - floor, 0.0)
            if slack.sum() > 0:
                new = new - slack * min(1.0, excess / slack.sum())
    return np.clip(new, 0.0, None)


def stamp_rate(d: pd.Timestamp) -> float:
    c = CFG["costs"]
    return c["stamp_before_2023_08_28"] if d < pd.Timestamp("2023-08-28") else c["stamp_after_2023_08_28"]


def run_backtest(targets, mkt: Market, start=None, end=None, cost_on: bool = True,
                 signal_dates=None, cost_mult: float = 1.0) -> dict:
    """Account-based simulation: a vector of position VALUES plus a CASH balance (NAV = sum(values) + cash,
    starting at 1). Four position concepts are kept apart in the output:

      at_signal  {signal date: weights}     actual weights at the signal close = the optimiser's starting point
      targets    {signal date: weights}     what the optimiser asked for (fractions of NAV, sum 1)
      pre_trade  {exec date: weights}       actual drifted weights at the execution VWAP, before trading
      holdings   {exec date: weights}       executed post-trade weights (fractions of post-trade NAV)
      cash       {exec date: weight}        post-trade cash weight; holdings + cash = 1

    Execution: blocked names keep their weight (_fill); sells are credited and buys paid from cash, and
    commission, slippage and stamp duty are paid from cash. If cash would go negative, all buys are scaled
    down by the same factor until it is exactly zero, so the account never borrows. Cash earns 0.
    turnover = 0.5 * (buys + sells) / pre-trade NAV (executed, not planned); cost = costs / pre-trade NAV.
    """
    c = CFG["costs"]
    dates = mkt.ret.index
    codes = mkt.ret.columns
    col = {k: i for i, k in enumerate(codes)}
    target_fn = targets if callable(targets) else None
    sig_dates = sorted(signal_dates if target_fn else targets)
    targets = {} if target_fn else dict(targets)
    exec_map = {}
    for s in sig_dates:
        j = dates.searchsorted(s, side="right")
        if j < len(dates):
            exec_map[dates[j]] = s
    first = min(exec_map)
    start = max(pd.Timestamp(start), first) if start else first
    days = dates[(dates >= start) & ((dates <= pd.Timestamp(end)) if end else True)]

    R, R1, R2 = mkt.ret.values, mkt.r_open_to_vwap.values, mkt.r_vwap_to_close.values
    NB, NS = mkt.no_buy.values, mkt.no_sell.values
    pos_of = {d: i for i, d in enumerate(dates)}

    V = np.zeros(len(codes))          # position values
    C = 1.0                           # cash
    navs, turn_log, cost_log, hold_log, trade_log, pre_log, cash_log, sig_log = [], {}, {}, {}, {}, {}, {}, {}
    wser = lambda v, n: pd.Series(v[v > 0] / n, index=codes[v > 0])
    cur = lambda: wser(V, V.sum() + C)
    sig_set = set(sig_dates)
    if target_fn:                        # signals decided before the first simulated day: start from cash
        for s_ in sig_dates:
            if s_ < start:
                sig_log[s_] = cur()
                targets[s_] = target_fn(s_, cur())
    for d in days:
        i = pos_of[d]
        if d in exec_map and exec_map[d] in targets:
            V = V * (1 + R1[i])                                  # open -> VWAP on the old book
            nav0 = V.sum() + C
            w_pre = V / nav0
            tgt = np.zeros(len(codes))
            t = targets[exec_map[d]]
            tgt[[col[k] for k in t.index if k in col]] = t[[k for k in t.index if k in col]].values
            new = _fill(w_pre, tgt, NB[i], NS[i])
            cb = (c["commission"] + c["slippage"]) * cost_mult if cost_on else 0.0
            cs = (c["commission"] + c["slippage"] + stamp_rate(d)) * cost_mult if cost_on else 0.0
            buy = np.clip(new - w_pre, 0, None)
            sell = np.clip(w_pre - new, 0, None)
            c0 = C / nav0
            cash_w = c0 + sell.sum() * (1 - cs) - buy.sum() * (1 + cb)
            if cash_w < 0 and buy.sum() > 0:                     # not enough cash: scale all buys down
                f = max(0.0, (c0 + sell.sum() * (1 - cs)) / (buy.sum() * (1 + cb)))
                buy = buy * f
                new = w_pre - sell + buy
                cash_w = c0 + sell.sum() * (1 - cs) - buy.sum() * (1 + cb)
            cost = cb * buy.sum() + cs * sell.sum()
            V = new * nav0
            C = max(cash_w, 0.0) * nav0
            nav1 = V.sum() + C
            turn_log[d], cost_log[d] = 0.5 * (buy.sum() + sell.sum()), cost
            dw = new - w_pre
            trade_log[d] = pd.Series(dw[dw != 0], index=codes[dw != 0])   # traded weight, fraction of pre-trade NAV
            pre_log[d] = wser(w_pre, 1.0)
            hold_log[d] = wser(V, nav1)
            cash_log[d] = C / nav1
            V = V * (1 + R2[i])                                  # VWAP -> close on the new book
        else:
            V = V * (1 + R[i])
        navs.append(V.sum() + C)
        if d in sig_set:                 # close of a signal date: record actual weights, decide next targets
            sig_log[d] = cur()
            if target_fn:
                targets[d] = target_fn(d, cur())
    nav_s = pd.Series(navs, index=days, name="nav")
    return {"nav": nav_s, "ret": nav_s.pct_change().fillna(nav_s.iloc[0] - 1),
            "turnover": pd.Series(turn_log, dtype=float), "cost": pd.Series(cost_log, dtype=float),
            "holdings": hold_log, "cash": pd.Series(cash_log, dtype=float), "trades": trade_log,
            "pre_trade": pre_log, "targets": targets, "at_signal": sig_log}
