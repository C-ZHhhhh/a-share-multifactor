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
The portfolio is always fully invested: costs are paid out of NAV and weights always sum to 1 (cash = 0).
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


def stamp_rate(d: pd.Timestamp) -> float:
    c = CFG["costs"]
    return c["stamp_before_2023_08_28"] if d < pd.Timestamp("2023-08-28") else c["stamp_after_2023_08_28"]


def run_backtest(targets, mkt: Market, start=None, end=None, cost_on: bool = True,
                 signal_dates=None, cost_mult: float = 1.0) -> dict:
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

    w = np.zeros(len(codes))
    nav, navs, turn_log, cost_log, hold_log, trade_log, pre_log = 1.0, [], {}, {}, {}, {}, {}
    cur = lambda: pd.Series(w[w > 0], index=codes[w > 0])
    sig_set = set(sig_dates)
    if target_fn:                        # signals decided before the first simulated day: start from cash
        for s_ in sig_dates:
            if s_ < start:
                targets[s_] = target_fn(s_, cur())
    for d in days:
        i = pos_of[d]
        if d in exec_map and exec_map[d] in targets:
            r1 = R1[i]
            g1 = 1 + w @ r1
            w = w * (1 + r1) / g1 if g1 > 0 else w
            nav *= g1
            tgt = np.zeros(len(codes))
            t = targets[exec_map[d]]
            tgt[[col[k] for k in t.index if k in col]] = t[[k for k in t.index if k in col]].values
            blocked = ((tgt > w) & NB[i]) | ((tgt < w) & NS[i])
            new = np.where(blocked, w, tgt)
            free = ~blocked
            room = 1 - new[blocked].sum()
            if new[free].sum() > 0:
                new[free] = new[free] * room / new[free].sum()
            buy = np.clip(new - w, 0, None).sum()
            sell = np.clip(w - new, 0, None).sum()
            cost = (buy * (c["commission"] + c["slippage"]) +
                    sell * (c["commission"] + c["slippage"] + stamp_rate(d))) * cost_mult if cost_on else 0.0
            nav *= (1 - cost)
            turn_log[d], cost_log[d] = 0.5 * (buy + sell), cost
            dw = new - w
            trade_log[d] = pd.Series(dw[dw != 0] * nav, index=codes[dw != 0])   # traded value / initial capital
            pre_log[d] = cur()
            w = new
            hold_log[d] = pd.Series(w[w > 0], index=codes[w > 0])
            r2 = R2[i]
            g2 = 1 + w @ r2
            w = w * (1 + r2) / g2 if g2 > 0 else w
            nav *= g2
        else:
            r = R[i]
            g = 1 + w @ r
            w = w * (1 + r) / g if g > 0 else w
            nav *= g
        navs.append(nav)
        if target_fn and d in sig_set:   # close of a signal date: decide next targets from actual weights
            targets[d] = target_fn(d, cur())
    nav_s = pd.Series(navs, index=days, name="nav")
    return {"nav": nav_s, "ret": nav_s.pct_change().fillna(nav_s.iloc[0] - 1),
            "turnover": pd.Series(turn_log), "cost": pd.Series(cost_log), "holdings": hold_log,
            "trades": trade_log, "pre_trade": pre_log, "targets": targets}
