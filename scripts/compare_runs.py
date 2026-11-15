"""Compare the current run with a saved snapshot of an earlier run and write reports/revision_log.md.

usage: python scripts/compare_runs.py data/snapshots/v1

Besides old-vs-new tables it isolates the effect of each methodology fix on the headline numbers:
  metric definitions   the same old returns scored with the old (geometric excess / TE) and new
                       (arithmetic mean active x 252 / TE) information ratio
  portfolio state      the current model and lambda run with the OLD optimiser input (previous month's
                       TARGET weights) vs the current one (actual drifted weights from the backtest)
  label availability   whatever remains: different factor set, model and lambda chosen in-sample
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mf.attribution import bench_weights, constraint_check
from mf.backtest import Market, run_backtest
from mf.config import CFG, PROC, REPORTS
from mf.dataset import load_dataset
from mf.metrics import perf
from mf.optimizer import grinold_alpha, optimize
from mf.panel import load_benchmarks, load_long
from mf.risk import risk_model_at


def stale_targets(score, data, fret, resid, pcfg):
    """The OLD construction: next month's optimiser starts from this month's target, not actual holdings."""
    targets, w0 = {}, None
    for d in sorted(score.dropna().index.get_level_values(0).unique()):
        s = score.loc[d].dropna()
        X, F, D = risk_model_at(d, data, fret, resid)
        g = data[data.date == d].set_index("code")
        w, _ = optimize(grinold_alpha(s, D.reindex(s.index).fillna(D.median())), bench_weights(data, d),
                        X, F, D, w0, g.tradable, pcfg)
        targets[d], w0 = w, w
    return targets


def overlap(h1: dict, h2: dict) -> pd.Series:
    """sum_i min(w1_i, w2_i) per common execution date (1 = identical portfolios)."""
    out = {}
    for d in sorted(set(h1) & set(h2)):
        a, b = h1[d], h2[d]
        out[d] = float(np.minimum(a.reindex(a.index.union(b.index)).fillna(0),
                                  b.reindex(a.index.union(b.index)).fillna(0)).sum())
    return pd.Series(out)


def old_ir(ret, bench):
    p = perf(ret, bench)
    return p["超额年化"] / p["跟踪误差"]


if __name__ == "__main__":
    old_dir = Path(sys.argv[1] if len(sys.argv) > 1 else PROC.parent / "snapshots" / "v1")
    S = CFG["split"]
    B = load_benchmarks()
    bench = B.rec
    old = pickle.load(open(old_dir / "backtests.pkl", "rb"))
    new = pickle.load(open(PROC / "backtests.pkl", "rb"))
    old_main, new_main = f"组合优化({old['best']})", f"组合优化({new['best']})"
    oc, nc = json.load(open(old_dir / "model_choice.json")), json.load(open(PROC / "model_choice.json"))
    fp = pd.Timestamp(nc["first_pred"])
    per = {"样本内": (fp, S["is_end"]), "样本外": (S["oos_start"], S["oos_end"])}
    pct = lambda v: f"{v:.2%}"

    # ---- decisions
    of, nf = json.load(open(old_dir / "selected_factors.json"))["selected"], json.load(open(PROC / "selected_factors.json"))["selected"]
    dec = pd.DataFrame({"旧": [len(of), ", ".join(of), old["best"], f"{old['lam']:g}"],
                        "新": [len(nf), ", ".join(nf), new["best"], f"{new['lam']:g}"]},
                       index=["入选因子数", "入选因子", "选中的模型", "λ"])
    dec.loc["新增 / 移除的因子"] = ["", f"+{sorted(set(nf) - set(of))} −{sorted(set(of) - set(nf))}"]

    # ---- model RankIC
    me_o, me_n = pd.read_csv(old_dir / "model_eval.csv"), pd.read_csv(PROC / "model_eval.csv")
    me = me_o.merge(me_n, on=["模型", "区间"], suffixes=("_旧", "_新"))
    me = me[["模型", "区间", "RankIC_旧", "RankIC_新", "ICIR(年化)_旧", "ICIR(年化)_新"]].round(4)

    # ---- performance (new metric definitions for both, old IR definition shown too)
    rows = []
    for name, bt, main in [("旧", old, old_main), ("新", new, new_main)]:
        r = bt["results"][main]
        for pn, (a, b) in per.items():
            p = perf(r["ret"].loc[a:b], bench, r["turnover"].loc[a:b], r["cost"].loc[a:b])
            rows.append({"版本": name, "区间": pn, "组合": main, "超额年化": pct(p["超额年化"]), "IR(新口径)": f"{p['信息比率']:.2f}",
                         "IR(旧口径)": f"{old_ir(r['ret'].loc[a:b], bench):.2f}", "夏普(新口径)": f"{p['夏普比率']:.2f}",
                         "跟踪误差": pct(p["跟踪误差"]), "年化换手": pct(p["年化单边换手"]), "成本拖累": pct(p["年化成本拖累"])})
    perf_df = pd.DataFrame(rows)

    # ---- isolate the portfolio-state fix: current model & lambda, old optimiser input
    data = load_dataset()
    fret = pd.read_parquet(PROC / "factor_returns.parquet")
    resid = pd.read_parquet(PROC / "specific_returns.parquet")
    mkt = Market.from_long(load_long())
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    score = scores[new["best"]][scores.index.get_level_values(0) >= fp]
    pcfg = {**CFG["portfolio"], "risk_aversion": new["lam"]}
    st = run_backtest(stale_targets(score, data, fret, resid, pcfg), mkt)
    nr = new["results"][new_main]
    dates = mkt.ret.index
    state_rows = []
    for label, r in [("旧做法: 以上月目标权重为起点", st), ("新做法: 以实际持仓为起点", nr)]:
        sig_of = {d: dates[dates.searchsorted(d) - 1] for d in r["holdings"]}
        cc = constraint_check(r["holdings"], sig_of, data, r["turnover"], pcfg)
        for pn, (a, b) in per.items():
            p = perf(r["ret"].loc[a:b], bench, r["turnover"].loc[a:b], r["cost"].loc[a:b])
            c = cc.loc[a:b]
            state_rows.append({"做法": label, "区间": pn, "超额年化": pct(p["超额年化"]), "IR": f"{p['信息比率']:.2f}",
                               "年化换手": pct(p["年化单边换手"]), "成本拖累": pct(p["年化成本拖累"]),
                               "换手上限满足率": pct(c.ok_turnover.mean()), "最大单边换手": pct(c.turnover.max()),
                               "行业偏离满足率": pct(c.ok_industry.mean()), "市值暴露满足率": pct(c.ok_size.mean())})
    state_df = pd.DataFrame(state_rows)

    # ---- effect decomposition of the OOS IR
    a, b = per["样本外"]
    ir = lambda r: perf(r["ret"].loc[a:b], bench)["信息比率"]
    ir_old_def = old_ir(old["results"][old_main]["ret"].loc[a:b], bench)
    ir_old_new_def = ir(old["results"][old_main])
    ir_stale = ir(st)
    ir_new = ir(nr)
    steps = pd.DataFrame([
        {"步骤": "旧版本 (旧 IR 口径)", "样本外 IR": ir_old_def, "变化": np.nan},
        {"步骤": "① 改为新 IR 口径 (同一条收益曲线)", "样本外 IR": ir_old_new_def, "变化": ir_old_new_def - ir_old_def},
        {"步骤": "② 修复时间泄漏: 重新筛因子、选模型、选 λ (仍用旧的组合状态做法)", "样本外 IR": ir_stale, "变化": ir_stale - ir_old_new_def},
        {"步骤": "③ 修复组合状态: 优化器以实际持仓为起点 = 新版本", "样本外 IR": ir_new, "变化": ir_new - ir_stale},
    ])
    steps["样本外 IR"] = steps["样本外 IR"].map(lambda v: f"{v:.2f}")
    steps["变化"] = steps["变化"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")

    # ---- holdings change
    ov = overlap(old["results"][old_main]["holdings"], nr["holdings"])
    hold_df = pd.DataFrame({pn: {"新旧持仓权重重合度 (平均)": pct(ov.loc[a_:b_].mean()),
                                  "最低": pct(ov.loc[a_:b_].min()), "最高": pct(ov.loc[a_:b_].max())}
                            for pn, (a_, b_) in per.items()}).T
    oh = old["results"][old_main]["holdings"]; last = max(set(oh) & set(nr["holdings"]))
    n_old, n_new = len(oh[last]), len(nr["holdings"][last])
    common = len(set(oh[last].index) & set(nr["holdings"][last].index))

    md = [
        "# 方法修订记录 (自动生成)", "",
        f"对比: 旧版本 = `{old_dir}` (快照), 新版本 = 当前运行。两个版本使用同一份下载数据, 差异只来自方法修订。", "",
        "## 修订内容", "",
        "1. **时间泄漏**: 每个信号日的标签只有在完全实现 (下一次调仓成交) 之后才能被使用。因子筛选、模型比较、λ 校准统一按"
        "'标签实现日 ≤ 样本内终点' 取样本; 旧版本按 '信号日 ≤ 样本内终点', 使 2021-11、2021-12 两个信号日 (标签在 2022 年才实现) 进入了样本内。",
        "2. **组合状态**: 优化器的起点 (换手约束、不能买入约束) 改为回测中经过涨跌、未成交之后的**实际持仓**; 旧版本用的是上个月的目标权重。",
        "3. **指标与归因**: 夏普改为 (日均收益 − 无风险利率) 年化 / 波动; IR 改为日均主动收益 × 252 / 跟踪误差; Brinson 分行业表补上交互项; "
        "新增 Brinson → 实际超额的对账和因子归因中交易成本的拆分。",
        "4. **基准口径**: 明确为'自定义流通市值基准增强', 官方中证800全收益只作副口径。", "",
        "## 样本内决策的变化", "", dec.to_markdown(), "",
        "## 选股模型 RankIC / ICIR", "", me.to_markdown(index=False), "",
        "## 主组合绩效", "", perf_df.to_markdown(index=False), "",
        "## 每项修订对样本外 IR 的影响", "", steps.to_markdown(index=False), "",
        "## 组合状态修复的单独效果 (同一模型、同一 λ)", "",
        "旧做法下, 优化器以为当前持仓就是上个月的目标, 实际持仓已经随行情漂移, 部分股票也没卖掉。"
        "所以优化器算出来的换手低于实际成交的换手, 约束在实际持仓上不一定满足。", "",
        state_df.to_markdown(index=False), "",
        "## 持仓变化", "", hold_df.to_markdown(), "",
        f"最新一期 ({last.date()}): 旧组合 {n_old} 只, 新组合 {n_new} 只, 共同持有 {common} 只。", "",
    ]
    (REPORTS / "revision_log.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
