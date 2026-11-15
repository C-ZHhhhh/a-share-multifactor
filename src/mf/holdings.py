"""Export the portfolios with explicit, separate position concepts, plus a readable latest-portfolio report.

One row per (signal date, stock) and one CASH row per signal date. Weight columns (fractions of NAV):

  weight_at_signal   actual holdings at the signal close (after a month of drift and any blocked trades);
                     this is the optimiser's starting point w0
  target_weight      the optimiser's target, decided at the signal close
  pre_trade_weight   actual holdings at the next day's VWAP, just before trading
  executed_weight    actual holdings right after trading (blocked trades, cost-scaled buys, cash)

For the CASH row the weights are the cash share (target cash is 0: targets are fully invested).
`status` explains a gap between target and executed weight. The latest signal is not executed yet
(`待执行`), so its executed columns are empty. Stock names are today's names from baostock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .attribution import bench_weights
from .config import RAW, REPORTS

OUT = REPORTS / "holdings"
TOL = 1e-4
COLS = ["signal_date", "exec_date", "code", "name", "industry", "weight_at_signal", "target_weight",
        "pre_trade_weight", "executed_weight", "bench_weight", "target_active", "executed_active",
        "model_rank", "n_ranked", "status"]


def _names() -> pd.Series:
    return pd.read_csv(RAW / "stock_basic.csv").set_index("code").code_name


def _status(w0, tgt, pre, ex) -> str:
    if np.isnan(ex):
        return "待执行"
    if tgt > pre + TOL and ex < tgt - TOL:
        return "买入受阻" if ex <= pre + TOL else "买入缩减"
    if tgt < pre - TOL and ex > tgt + TOL:
        return "卖出受阻"
    return "正常"


def holdings_table(bt: dict, data: pd.DataFrame, scores: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """bt: a run_backtest result; scores: model score indexed by (date, code); dates: trading calendar."""
    names = _names()
    exec_of = {s: dates[dates.searchsorted(s, side="right")] if dates.searchsorted(s, side="right") < len(dates)
               else pd.NaT for s in bt["targets"]}
    rows = []
    for d in sorted(bt["targets"]):
        e = exec_of[d]
        executed = e in bt["holdings"] if pd.notna(e) else False
        w0 = bt["at_signal"].get(d, pd.Series(dtype=float))
        tg = bt["targets"][d]
        tg = tg[tg > 0]
        pre = bt["pre_trade"][e] if executed else pd.Series(dtype=float)
        ex = bt["holdings"][e] if executed else pd.Series(dtype=float)
        codes = w0.index.union(tg.index).union(pre.index).union(ex.index)
        g = data[data.date == d].set_index("code")
        b = bench_weights(data, d).reindex(codes).fillna(0)
        s = scores.loc[d].dropna() if d in scores.index.get_level_values(0) else pd.Series(dtype=float)
        rank = s.rank(ascending=False, method="min")
        f = lambda x: x.reindex(codes).fillna(0).values
        t = pd.DataFrame({
            "signal_date": d.date(), "exec_date": e.date() if pd.notna(e) else None, "code": codes,
            "name": names.reindex(codes).values, "industry": g.industry.reindex(codes).values,
            "weight_at_signal": f(w0), "target_weight": f(tg),
            "pre_trade_weight": f(pre) if executed else np.nan, "executed_weight": f(ex) if executed else np.nan,
            "bench_weight": b.values, "model_rank": rank.reindex(codes).values, "n_ranked": len(s)})
        cash = pd.DataFrame([{
            "signal_date": d.date(), "exec_date": t.exec_date.iloc[0] if len(t) else None, "code": "CASH",
            "name": "现金", "industry": "", "weight_at_signal": 1 - w0.sum(), "target_weight": 1 - tg.sum(),
            "pre_trade_weight": 1 - pre.sum() if executed else np.nan,
            "executed_weight": float(bt["cash"].get(e, np.nan)) if executed else np.nan,
            "bench_weight": 0.0, "model_rank": np.nan, "n_ranked": len(s)}])
        rows += [t, cash]
    t = pd.concat(rows, ignore_index=True)
    t["target_active"] = t.target_weight - t.bench_weight
    t["executed_active"] = t.executed_weight - t.bench_weight
    t["status"] = [_status(a, b_, c, d_) for a, b_, c, d_ in
                   zip(t.weight_at_signal, t.target_weight, t.pre_trade_weight.fillna(np.nan), t.executed_weight)]
    t.loc[(t.code == "CASH") & t.executed_weight.notna(), "status"] = "现金"
    t["_cash"] = t.code == "CASH"
    t = t.sort_values(["signal_date", "_cash", "target_weight"], ascending=[True, True, False]).drop(columns="_cash")
    keep = (t[["weight_at_signal", "target_weight", "pre_trade_weight", "executed_weight"]].fillna(0).abs() > 1e-8).any(axis=1)
    return t[keep | (t.code == "CASH")][COLS]


def write_holdings(portfolios: dict, data: pd.DataFrame, opt_info: pd.DataFrame, main: str,
                   dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """portfolios: {name: (backtest result, scores)}. Writes one CSV per portfolio and latest_portfolio.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"组合优化": "optimized_monthly.csv", "Top50": "top50_equal_weight_monthly.csv"}
    tables = {}
    for name, (bt, scores) in portfolios.items():
        t = holdings_table(bt, data, scores, dates)
        fn = next(v for k, v in files.items() if name.startswith(k))
        t.to_csv(OUT / fn, index=False, float_format="%.6f", encoding="utf-8-sig")  # BOM: opens in Excel
        tables[name] = t
    _latest_report(tables[main], data, opt_info, main)
    return tables


def execution_summary(t: pd.DataFrame) -> pd.DataFrame:
    """Per executed rebalance: planned vs executed, blocked trades and cash (from the holdings table)."""
    ex = t[t.executed_weight.notna()]
    st = ex[ex.code != "CASH"]
    g = st.groupby("signal_date")
    out = pd.DataFrame({
        "exec_date": g.exec_date.first(),
        "planned_turnover": g.apply(lambda x: 0.5 * (x.target_weight - x.weight_at_signal).abs().sum()),
        "executed_turnover": g.apply(lambda x: 0.5 * (x.executed_weight - x.pre_trade_weight).abs().sum()),
        "n_buy_blocked": g.status.apply(lambda s: int((s == "买入受阻").sum())),
        "n_buy_scaled": g.status.apply(lambda s: int((s == "买入缩减").sum())),
        "n_sell_blocked": g.status.apply(lambda s: int((s == "卖出受阻").sum())),
        "target_minus_executed": g.apply(lambda x: 0.5 * (x.target_weight - x.executed_weight).abs().sum()),
    })
    out["cash_after"] = ex[ex.code == "CASH"].set_index("signal_date").executed_weight
    return out


def latest_snapshot(t: pd.DataFrame, data: pd.DataFrame) -> dict:
    """The latest TARGET portfolio (decided at the last signal close, executed the next day) vs the actual
    holdings at that close, plus benchmark top-10 and industry active weights. Shared by
    latest_portfolio.md and the HTML report."""
    d = t.signal_date.max()
    all_ = t[(t.signal_date == d) & (t.code != "CASH")].copy()
    cur = all_[all_.target_weight > 0].copy()
    cur["weight"] = cur.target_weight
    cur["active_weight"] = cur.target_active
    cur["change"] = (cur.target_weight - cur.weight_at_signal).round(4) + 0.0   # +0.0: no "-0.00%"
    cur["is_new"] = cur.weight_at_signal <= 1e-8
    sold = all_[(all_.target_weight <= 0) & (all_.weight_at_signal > 0)].code
    b = bench_weights(data, pd.Timestamp(d))
    g = data[data.date == pd.Timestamp(d)].set_index("code")
    bind = b.groupby(g.industry).sum()
    ind = cur.groupby("industry").weight.sum().reindex(bind.index.union(cur.industry.unique())).fillna(0) \
        .sub(bind, fill_value=0).sort_values()
    cash_now = t[(t.signal_date == d) & (t.code == "CASH")].weight_at_signal.iloc[0]
    return {"date": d, "cur": cur.sort_values("weight", ascending=False), "sold": pd.Index(sold), "bench": b,
            "industry_active": ind, "n_ranked": int(cur.n_ranked.iloc[0]), "cash_at_signal": float(cash_now)}


def _latest_report(t: pd.DataFrame, data: pd.DataFrame, opt_info: pd.DataFrame, main: str):
    snap = latest_snapshot(t, data)
    d, cur, sold, b = snap["date"], snap["cur"], snap["sold"], snap["bench"]
    info = opt_info.loc[pd.Timestamp(d)]
    names = _names()
    n_ranked = snap["n_ranked"]
    ex = execution_summary(t)
    last = ex.iloc[-1]

    pct = lambda x: f"{x:.2%}"
    show = cur.assign(**{
        "目标权重": cur.weight.map(pct), "信号日实际持仓": cur.weight_at_signal.map(pct),
        "基准权重": cur.bench_weight.map(pct), "目标主动权重": cur.active_weight.map(lambda x: f"{x:+.2%}"),
        "计划调整": cur.change.map(lambda x: f"{x:+.2%}"),
        "模型排名": cur.model_rank.map(lambda x: f"{int(x)} / {n_ranked}" if pd.notna(x) else "-")})
    show.loc[cur.is_new.values, "计划调整"] = "新买入"
    show = show.rename(columns={"code": "代码", "name": "名称", "industry": "行业"})
    show = show[["代码", "名称", "行业", "目标权重", "信号日实际持仓", "基准权重", "目标主动权重", "计划调整", "模型排名"]]

    top_b = b.sort_values(ascending=False).head(10)
    wmap = cur.set_index("code").weight
    big = pd.DataFrame({"名称": names.reindex(top_b.index), "基准权重": top_b.map(pct),
                        "目标权重": wmap.reindex(top_b.index).fillna(0).map(pct)})
    big.index.name = "代码"

    ind = snap["industry_active"]
    ind = ind[ind.abs() >= 0.005]
    ind.index.name = "行业"
    md = [
        "# 最新一期组合 (自动生成)", "",
        "> 研究回测的输出，用于说明模型如何工作，**不构成投资建议**。", "",
        "## 仓位口径", "",
        "- **目标仓位**: 优化器在信号日收盘后给出的目标, 满仓 (现金 0)。",
        "- **信号日实际持仓**: 上次调仓成交后经过一个月涨跌的实际仓位, 也是优化器的起点。",
        "- **成交后仓位**: 下一交易日按 VWAP 执行后的实际仓位。涨停 / 停牌买不进的部分留作现金, 跌停 / 停牌卖不掉的保留原仓位, 交易成本从现金支付。",
        "- 最新一期还没有执行, 所以下表是**目标仓位**; 所有历史月份的四种仓位和现金见 CSV。", "",
        "## 最新目标仓位", "",
        f"- 组合: **{main}**; 信号日 **{d}** 收盘后生成, 下一交易日按 VWAP 执行, 持有至下一个月末",
        f"- 股票池中可打分股票 {n_ranked} 只 → 目标持有 **{len(cur)}** 只; 前 10 大目标权重合计 {pct(cur.weight.head(10).sum())}",
        f"- 计划单边换手 (目标 vs 信号日实际持仓) {pct(info.planned_turnover)}; 事前跟踪误差 (年化) {pct(info.ex_ante_te)}; "
        f"约束放宽倍数 {info.relax:g} (1 = 未放宽); 信号日现金 {pct(snap['cash_at_signal'])}",
        f"- 目标持仓模型排名中位数 {int(cur.model_rank.median())}; 排名前 100 的 {(cur.model_rank <= 100).sum()} 只; "
        f"计划新买入 {int(cur.is_new.sum())} 只, 计划清仓 {len(sold)} 只", "",
        "## 上一次调仓的执行情况", "",
        f"- 信号日 {last.name}, 执行日 {last.exec_date}: 计划单边换手 {pct(last.planned_turnover)}, 实际成交 {pct(last.executed_turnover)}",
        f"- 买入受阻 {int(last.n_buy_blocked)} 只, 买入因成本或卖出受阻而缩减 {int(last.n_buy_scaled)} 只, 卖出受阻 {int(last.n_sell_blocked)} 只; "
        f"目标与成交后仓位的差异 (单边) {pct(last.target_minus_executed)}; 成交后现金 {pct(last.cash_after)}", "",
        "**怎么读目标仓位**: 模型排名靠前的股票通常被超配到上限 (基准权重 + 3%), 这是赚取超额的仓位; "
        "基准权重大但排名一般的股票 (如大型银行) 也可能被持有, 它们用来把行业和市值暴露拉回基准附近, "
        "这是控制风险的仓位。这正是转移系数 (模型信号转化为实际仓位的程度) 偏低的直观体现。", "",
        "## 全部目标持仓", "", show.to_markdown(index=False), "",
        "## 基准前 10 大权重股的处理", "", big.to_markdown(), "",
        "## 行业主动权重 (目标 − 基准, |偏离| ≥ 0.5% 的行业; 约束上限 ±5%)", "",
        ind.map(lambda x: f"{x:+.2%}").rename("主动权重").to_frame().to_markdown(), "",
        f"计划清仓: {', '.join(f'{names.get(c, c)}({c})' for c in sold) or '无'}", "",
        "完整历史 (每月一份): `optimized_monthly.csv` (组合优化), `top50_equal_weight_monthly.csv` (对照组)。",
        "列说明: weight_at_signal 信号日实际持仓; target_weight 目标; pre_trade_weight 执行日交易前; "
        "executed_weight 成交后; bench_weight 基准权重; target_active / executed_active 主动权重; "
        "status 正常 / 买入受阻 / 买入缩减 / 卖出受阻 / 待执行 / 现金; code = CASH 的行是现金。",
    ]
    (OUT / "latest_portfolio.md").write_text("\n".join(md), encoding="utf-8")
