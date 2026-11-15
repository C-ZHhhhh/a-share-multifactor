"""Step 6: IS vs OOS evaluation, Brinson + factor attribution with reconciliation, constraint check,
figures, results tables, holdings export and the data for the HTML report (built in step 8)."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mf.attribution import active_exposure_series, brinson, constraint_check, factor_attribution, reconcile
from mf.config import CFG, FIG, PROC, REPORTS
from mf.dataset import load_dataset
from mf.holdings import write_holdings
from mf.html_report import build_html_report
from mf.metrics import fmt_table, perf, yearly_table
from mf.panel import load_benchmarks, load_index, load_long, to_wide
from mf.plotting import setup_style

if __name__ == "__main__":
    setup_style()
    S = CFG["split"]
    bt = pickle.load(open(PROC / "backtests.pkl", "rb"))
    res, best, opt_info = bt["results"], bt["best"], bt["opt_info"]
    first_pred = pd.Timestamp(json.load(open(PROC / "model_choice.json"))["first_pred"])
    B = load_benchmarks()
    bench, tri = B.rec, B.tri
    hs300 = load_index("H00300").ret
    zz500 = load_index("H00905").ret
    main = f"组合优化({best})"
    periods = {"样本内 IS": (first_pred, S["is_end"]), "样本外 OOS": (S["oos_start"], S["oos_end"])}

    # ---------------- performance tables
    perf_tabs = {}
    for pname, (a, b) in periods.items():
        rows = {}
        for name, r in res.items():
            p = perf(r["ret"].loc[a:b], bench, r["turnover"].loc[a:b], r["cost"].loc[a:b])
            p["超额年化(vs官方全收益)"] = perf(r["ret"].loc[a:b], tri)["超额年化"]
            rows[name] = p
        span = r["ret"].loc[a:b].index
        rows["自定义基准(成分股流通市值加权)"] = perf(bench.reindex(span).fillna(0))
        rows["中证800全收益(官方)"] = perf(tri.reindex(span).fillna(0))
        perf_tabs[pname] = fmt_table(rows)
    yearly = yearly_table(res[main]["ret"], bench)

    # ---------------- NAV figure
    r = res[main]["ret"]
    b = bench.reindex(r.index).fillna(0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    for name, rr in res.items():
        if "无成本" in name:
            continue
        axes[0].plot((1 + rr["ret"]).cumprod(), lw=1.6 if name == main else 1, label=name)
    axes[0].plot((1 + b).cumprod(), color="k", lw=1.4, label="自定义基准 (成分股流通市值加权)")
    axes[0].plot((1 + tri.reindex(r.index).fillna(0)).cumprod(), color="k", lw=0.8, ls=":", label="中证800全收益 (官方)")
    axes[0].plot((1 + hs300.reindex(r.index).fillna(0)).cumprod(), color="gray", lw=0.8, ls="--", label="沪深300全收益")
    axes[0].set_title("净值曲线 (含交易成本)"); axes[0].legend(fontsize=8, ncol=2)
    ex = (1 + r).cumprod() / (1 + b).cumprod()
    axes[1].plot(ex, color="C3", label=f"{main} 相对基准")
    axes[1].plot((1 + res[main + " 无成本"]["ret"]).cumprod() / (1 + b).cumprod(), color="C3", ls=":",
                 label="同上 (无成本)")
    axes[1].set_title("超额净值"); axes[1].legend(fontsize=8)
    axes[2].fill_between(ex.index, ex / ex.cummax() - 1, 0, color="C3", alpha=0.4)
    axes[2].set_title("超额回撤")
    for ax in axes:
        ax.axvspan(pd.Timestamp(S["oos_start"]), r.index[-1], color="gold", alpha=0.12)
    axes[0].text(pd.Timestamp(S["oos_start"]), axes[0].get_ylim()[1] * 0.95, "  样本外 →", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "nav.png", dpi=130); plt.close()

    # ---------------- optimiser diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    opt_info["ex_ante_te"].plot(ax=axes[0], title="事前跟踪误差 (年化)")
    opt_info["n_holdings"].plot(ax=axes[1], title="持仓股票数")
    res[main]["turnover"].plot(ax=axes[2], title="单边换手 (每次调仓)", kind="line")
    plt.tight_layout(); plt.savefig(FIG / "optimizer_diag.png", dpi=120); plt.close()

    # ---------------- attribution
    data = load_dataset()
    long = load_long()
    daily_ret = to_wide(long, "ret").fillna(0)
    fret = pd.read_parquet(PROC / "factor_returns.parquet")
    hold = res[main]["holdings"]
    dates = daily_ret.index
    sig_of = {d: dates[dates.searchsorted(d) - 1] for d in hold}
    br = brinson(hold, sig_of, data, daily_ret)
    active = (r - b)
    fa = factor_attribution(hold, sig_of, data, fret, active, res[main]["cost"])
    rec = reconcile(hold, sig_of, data, daily_ret, r, b, res[main]["cost"])
    pcfg = {**CFG["portfolio"], "risk_aversion": bt["lam"]}
    cons = constraint_check(hold, sig_of, data, res[main]["turnover"], pcfg)
    expo = active_exposure_series(hold, sig_of, data)

    br_tabs, fa_tabs, rec_tabs, cons_tabs = {}, {}, {}, {}
    for pname, (a, bb) in periods.items():
        sub = br[(br.date >= a) & (br.date <= bb)]
        yrs = len(r.loc[a:bb]) / 252
        br_tabs[pname] = (sub[["allocation", "selection", "interaction"]].sum() / yrs)
        f = fa.loc[a:bb]
        fa_tabs[pname] = f.sum() / yrs
        rc = rec.loc[a:bb].sum() / yrs
        p_ = perf(r.loc[a:bb], bench)
        rc["复利与日度/期间口径差"] = p_["超额年化"] - rc["actual"]
        rc["超额年化(几何, 绩效表口径)"] = p_["超额年化"]
        rec_tabs[pname] = rc
        cs = cons.loc[a:bb]
        cons_tabs[pname] = {"调仓次数": len(cs), "行业偏离满足率": cs.ok_industry.mean(), "市值暴露满足率": cs.ok_size.mean(),
                            "单票上限满足率": cs.ok_cap.mean(), "换手上限满足率": cs.ok_turnover.mean(),
                            "最大行业偏离": cs.max_industry_dev.max(), "最大市值暴露(σ)": cs.size_exposure.max(),
                            "最大单边换手": cs.turnover.max(), "平均单边换手": cs.turnover.mean(),
                            "被迫持有(已出股票池)平均权重": cs.out_of_universe.mean()}
    br_df = pd.DataFrame(br_tabs).T
    br_df["合计"] = br_df.sum(axis=1)
    fa_df = pd.DataFrame(fa_tabs).T
    rec_df = pd.DataFrame(rec_tabs).T.rename(columns={
        "brinson": "Brinson合计(买入持有)", "cost": "交易成本", "execution": "调仓日执行时点差",
        "benchmark": "基准口径差(日度再平衡 vs 月初权重)", "actual": "实际超额(期间算术加总)"})
    cons_df = pd.DataFrame(cons_tabs).T
    oos_br = br[br.date >= S["oos_start"]].groupby("industry")[["allocation", "selection", "interaction"]].sum()
    oos_br = oos_br.assign(total=oos_br.sum(axis=1)).sort_values("total")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fa.drop(columns=["active"]).cumsum().plot(ax=axes[0], lw=1.2)
    axes[0].plot(fa["active"].cumsum(), color="k", lw=1.8, label="主动收益合计")
    axes[0].axvline(pd.Timestamp(S["oos_start"]), color="gray", ls="--")
    axes[0].legend(fontsize=7); axes[0].set_title("因子收益归因 (累计, 算术)")
    expo.plot(ax=axes[1], lw=1); axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("主动风格暴露 (σ)"); axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "attribution.png", dpi=130); plt.close()

    pct = lambda df: df.map(lambda v: f"{v:.2%}")
    calib = bt["calib"].copy()
    for c in ["ex_ante_te", "超额年化", "跟踪误差"]:
        calib[c] = calib[c].map(lambda v: f"{v:.2%}")
    model_eval = pd.read_csv(PROC / "model_eval.csv")

    md = [
        "# 回测与归因结果 (自动生成)", "",
        f"- 选股模型 (样本内 ICIR 最优): **{best}**; 风险厌恶系数 (样本内 IR 最优): **λ={bt['lam']}**",
        "- **基准口径**: 本项目是**自定义流通市值基准增强**, 不是官方中证800指数增强。主基准 = 时点中证800成分股按 (前一日) 流通市值加权的全收益, "
        "即优化器跟踪的组合; 官方指数用自由流通市值、分级靠档和权重上限, 历史成分权重无法免费获取。官方中证800全收益 (H00906) 只作副口径, 见 '超额年化(vs官方全收益)' 一列和下方基准对比表。",
        f"- 夏普比率 = (日收益均值 − 无风险利率/252) × 252 / 年化波动, 无风险利率 {CFG.get('risk_free', 0):.1%}; "
        "信息比率 = 日主动收益均值 × 252 / 跟踪误差 (算术口径); 年化收益与超额为几何口径",
        f"- 预测区间: {first_pred.date()} 起 (前 24 个月仅用于训练); 样本外: {S['oos_start']} ~ {S['oos_end']}",
        "- 成本: 佣金万2.5 双边 + 滑点千1 双边 + 印花税 (2023-08-28 前千1, 之后万5, 仅卖出)", "",
        "## 模型预测能力", "", model_eval.round(4).to_markdown(index=False), "",
        "## 风险厌恶系数校准 (仅样本内)", "", calib.round(3).to_markdown(), "",
        *sum([[f"## 绩效 — {k}", "", v.to_markdown(), ""] for k, v in perf_tabs.items()], []),
        f"## 分年度 — {main}", "", pct(yearly).to_markdown(), "",
        "![nav](figures/nav.png)", "", "![diag](figures/optimizer_diag.png)", "",
        "## 约束满足率 (实际成交后的持仓, 而非优化器目标)", "",
        "优化器以实际持仓 (经过一个月涨跌、以及涨跌停/停牌导致的未成交) 为起点; 下表按实际成交后的持仓逐月检查约束。",
        "不满足的月份来自: 一字跌停/停牌卖不出、已调出股票池但无法卖出的股票, 以及优化器为求可行而放宽约束的月份。", "",
        pct(cons_df[[c for c in cons_df.columns if c not in ("调仓次数", "最大市值暴露(σ)")]]).assign(
            调仓次数=cons_df["调仓次数"].astype(int), **{"最大市值暴露(σ)": cons_df["最大市值暴露(σ)"].map(lambda v: f"{v:.3f}")}).to_markdown(), "",
        "## Brinson 行业归因 (年化, 算术加总)", "", pct(br_df).to_markdown(), "",
        "样本外分行业 (累计, 含交互项):", "", pct(pd.concat([oos_br.head(5), oos_br.tail(5)])).to_markdown(), "",
        "## 归因对账: 从 Brinson 到实际超额 (年化)", "",
        "Brinson (含交互项) 精确解释的是 '调仓后权重买入持有一个月' 的组合相对 '信号日基准权重买入持有' 的超额。"
        "它与绩效表中的超额之间的差异逐项列出: Brinson合计 + 交易成本 + 调仓日执行时点差 + 基准口径差 = 实际超额 (期间算术加总); "
        "再加上复利口径差 = 几何超额年化。", "",
        pct(rec_df).to_markdown(), "",
        "## 因子收益归因 (年化主动收益贡献)", "",
        "industry + 6 个风格 + cost (交易成本) + specific = active (按日精确对账)。specific 是风险模型解释不了的部分: "
        "个股特异收益、月内权重漂移 (暴露按调仓日权重计算) 和调仓日执行时点。", "",
        pct(fa_df).to_markdown(), "",
        "![attr](figures/attribution.png)", "",
        "## 实际持仓", "",
        "每月组合的完整持仓见 [holdings/](holdings/): 最新一期的可读版本 [latest_portfolio.md](holdings/latest_portfolio.md), "
        "全部历史 `optimized_monthly.csv` (组合优化) 和 `top50_equal_weight_monthly.csv` (Top50 等权对照组)。", "",
    ]
    (REPORTS / "results.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- holdings export (the actual stock lists behind every backtest number)
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    scores = scores[scores.index.get_level_values(0) >= first_pred]
    n = CFG["portfolio"]["topn"]
    top = {d: pd.Series(1.0 / min(n, len(s)), index=s.droplevel(0).nlargest(n).index)
           for d, s in scores[best].dropna().groupby(level=0)}
    top_name = f"Top{n}等权({best})"
    tables = write_holdings({main: (bt["opt_targets"], scores[best]), top_name: (top, scores[best])},
                            data, opt_info, main)
    print("holdings written to", REPORTS / "holdings")

    # ---------------- interactive HTML report (one per run)
    payload = build_html_report(
        res=res, main=main, top_name=top_name, bench=bench, tri=tri, opt_info=opt_info, perf_tabs=perf_tabs,
        yearly=yearly, model_eval=model_eval, calib=calib, lam=bt["lam"], best=best, first_pred=first_pred,
        scores=scores[best], data=data, br_df=br_df, fa_df=fa_df, oos_br=oos_br, holdings_main=tables[main],
        factor_tests_is=pd.read_csv(PROC / "factor_tests_is.csv"),
        selected=json.load(open(PROC / "selected_factors.json")))
    payload["reconcile"] = {"columns": list(rec_df.columns), "rows": {k: [float(v) for v in row] for k, row in rec_df.iterrows()}}
    payload["constraints"] = {k: {kk: float(vv) for kk, vv in row.items()} for k, row in cons_df.iterrows()}
    json.dump(payload, open(PROC / "report_payload.json", "w"), ensure_ascii=False)
    print("report payload saved; run scripts/08_html_report.py to build reports/report.html")
    for k, v in perf_tabs.items():
        print(k); print(v.to_string())
    print(br_df); print(fa_df)
