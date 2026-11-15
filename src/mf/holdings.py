"""Export the actual portfolios: every month's target weights plus a readable latest-portfolio report.

Each row is one stock held on one signal date (the month-end close on which the portfolio is decided;
trades execute at the next day's VWAP). Stock names are today's names from baostock, not point-in-time.
"""
from __future__ import annotations

import pandas as pd

from .attribution import bench_weights
from .config import RAW, REPORTS

OUT = REPORTS / "holdings"


def _names() -> pd.Series:
    return pd.read_csv(RAW / "stock_basic.csv").set_index("code").code_name


def holdings_table(targets: dict, data: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """targets: {signal_date: weights}; scores: model score indexed by (date, code)."""
    names = _names()
    rows = []
    for d in sorted(targets):
        w = targets[d]
        w = w[w > 0]
        g = data[data.date == d].set_index("code")
        b = bench_weights(data, d)
        s = scores.loc[d].dropna()
        rank = s.rank(ascending=False, method="min")
        rows.append(pd.DataFrame({
            "signal_date": d.date(), "code": w.index, "name": names.reindex(w.index).values,
            "industry": g.industry.reindex(w.index).values,
            "weight": w.values, "bench_weight": b.reindex(w.index).fillna(0).values,
            "model_rank": rank.reindex(w.index).values, "n_ranked": len(s)}))
    t = pd.concat(rows, ignore_index=True)
    t["active_weight"] = t.weight - t.bench_weight
    cols = ["signal_date", "code", "name", "industry", "weight", "bench_weight", "active_weight",
            "model_rank", "n_ranked"]
    return t[cols].sort_values(["signal_date", "weight"], ascending=[True, False])


def write_holdings(portfolios: dict[str, tuple[dict, pd.Series]], data: pd.DataFrame,
                   opt_info: pd.DataFrame, main: str) -> dict[str, pd.DataFrame]:
    """portfolios: {name: (targets, scores)}. Writes one CSV per portfolio and latest_portfolio.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"组合优化": "optimized_monthly.csv", "Top50": "top50_equal_weight_monthly.csv"}
    tables = {}
    for name, (targets, scores) in portfolios.items():
        t = holdings_table(targets, data, scores)
        fn = next(v for k, v in files.items() if name.startswith(k))
        t.to_csv(OUT / fn, index=False, float_format="%.6f", encoding="utf-8-sig")  # BOM: opens in Excel
        tables[name] = t
    _latest_report(tables[main], data, opt_info, main)
    return tables


def latest_snapshot(t: pd.DataFrame, data: pd.DataFrame) -> dict:
    """The latest month's holdings with month-on-month changes, sold names, benchmark top-10 and
    industry active weights. Shared by latest_portfolio.md and the HTML report."""
    d = t.signal_date.max()
    cur = t[t.signal_date == d].copy()
    prev_d = sorted(t.signal_date.unique())[-2]
    prev = t[t.signal_date == prev_d].set_index("code").weight
    cur["change"] = (cur.weight - prev.reindex(cur.code).fillna(0).values).round(4) + 0.0  # +0.0: no "-0.00%"
    cur["is_new"] = ~cur.code.isin(prev.index)
    b = bench_weights(data, pd.Timestamp(d))
    g = data[data.date == pd.Timestamp(d)].set_index("code")
    bind = b.groupby(g.industry).sum()
    ind = cur.groupby("industry").weight.sum().reindex(bind.index.union(cur.industry.unique())).fillna(0) \
        .sub(bind, fill_value=0).sort_values()
    return {"date": d, "cur": cur, "prev": prev, "sold": prev.index.difference(cur.code), "bench": b,
            "industry_active": ind, "n_ranked": int(cur.n_ranked.iloc[0])}


def _latest_report(t: pd.DataFrame, data: pd.DataFrame, opt_info: pd.DataFrame, main: str):
    snap = latest_snapshot(t, data)
    d, cur, prev, sold, b = snap["date"], snap["cur"], snap["prev"], snap["sold"], snap["bench"]
    info = opt_info.loc[pd.Timestamp(d)]
    names = _names()
    n_ranked = snap["n_ranked"]

    pct = lambda x: f"{x:.2%}"
    show = cur.assign(**{
        "权重": cur.weight.map(pct), "基准权重": cur.bench_weight.map(pct),
        "主动权重": cur.active_weight.map(lambda x: f"{x:+.2%}"),
        "较上月": cur.change.map(lambda x: f"{x:+.2%}"),
        "模型排名": cur.model_rank.map(lambda x: f"{int(x)} / {n_ranked}")})
    new_codes = set(cur.code[cur.is_new])
    show.loc[show.code.isin(new_codes), "较上月"] = "新买入"
    show = show.rename(columns={"code": "代码", "name": "名称", "industry": "行业"})
    show = show[["代码", "名称", "行业", "权重", "基准权重", "主动权重", "较上月", "模型排名"]]

    top_b = b.sort_values(ascending=False).head(10)
    wmap = cur.set_index("code").weight
    big = pd.DataFrame({"名称": names.reindex(top_b.index), "基准权重": top_b.map(pct),
                        "组合权重": wmap.reindex(top_b.index).fillna(0).map(pct)})
    big.index.name = "代码"

    ind = snap["industry_active"]
    ind = ind[ind.abs() >= 0.005]
    ind.index.name = "行业"
    md = [
        "# 最新一期组合 (自动生成)", "",
        "> 研究回测的输出，用于说明模型如何工作，**不构成投资建议**。", "",
        f"- 组合: **{main}**; 信号日 **{d}** 收盘后生成, 下一交易日按 VWAP 调仓, 持有至下一个月末",
        f"- 股票池中可打分股票 {n_ranked} 只 → 持有 **{len(cur)}** 只; 前 10 大持仓合计 {pct(cur.weight.head(10).sum())}",
        f"- 本期单边换手 {pct(info.turnover)}; 事前跟踪误差 (年化) {pct(info.ex_ante_te)}; "
        f"约束放宽倍数 {info.relax:g} (1 = 未放宽)",
        f"- 持仓模型排名中位数 {int(cur.model_rank.median())}; 排名前 100 的持仓 {(cur.model_rank <= 100).sum()} 只; "
        f"较上月新买入 {len(new_codes)} 只, 清仓 {len(sold)} 只", "",
        "**怎么读这张表**: 模型排名靠前的股票通常被超配到上限 (基准权重 + 3%), 这是赚取超额的仓位; "
        "基准权重大但排名一般的股票 (如大型银行) 也可能被持有, 它们用来把行业和市值暴露拉回基准附近, "
        "这是控制风险的仓位。这正是转移系数只有约 0.3 的直观体现。", "",
        "## 全部持仓", "", show.to_markdown(index=False), "",
        "## 基准前 10 大权重股的处理", "", big.to_markdown(), "",
        "## 行业主动权重 (组合 − 基准, |偏离| ≥ 0.5% 的行业; 约束上限 ±5%)", "",
        ind.map(lambda x: f"{x:+.2%}").rename("主动权重").to_frame().to_markdown(), "",
        f"清仓股票: {', '.join(f'{names.get(c, c)}({c})' for c in sold) or '无'}", "",
        "完整历史持仓 (每月一份): `optimized_monthly.csv` (组合优化), `top50_equal_weight_monthly.csv` (对照组)。",
        "列说明: signal_date 信号日; weight 组合权重; bench_weight 基准 (流通市值) 权重; "
        "active_weight 主动权重; model_rank 模型排名 (1 = 最看好); n_ranked 当期可打分股票数。",
    ]
    (OUT / "latest_portfolio.md").write_text("\n".join(md), encoding="utf-8")
