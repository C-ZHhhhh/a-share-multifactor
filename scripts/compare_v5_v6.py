"""v5 -> v6 difference table (acceptance 6): headline numbers, benchmark change, and a counterfactual that keeps
the v5 lambda (5) under the v6 code, so the effect of the code fixes is separated from the lambda change.
Writes reports/revision_v6.md. Usage: python scripts/compare_v5_v6.py data/snapshots/v5"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mf.backtest import Market
from mf.config import CFG, PROC, REPORTS
from mf.dataset import load_dataset
from mf.metrics import perf
from mf.panel import load_benchmarks, load_long
from mf.portfolio import optimised_backtest

if __name__ == "__main__":
    old_dir = Path(sys.argv[1] if len(sys.argv) > 1 else PROC.parent / "snapshots" / "v5")
    S = CFG["split"]
    oos = (S["oos_start"], S["oos_end"])
    old, new = pickle.load(open(old_dir / "backtests.pkl", "rb")), pickle.load(open(PROC / "backtests.pkl", "rb"))
    om, nm = f"组合优化({old['best']})", f"组合优化({new['best']})"
    b_old = pd.read_parquet(old_dir / "bench_rec.parquet")["bench_rec"]
    B = load_benchmarks()
    b_new = B.rec
    pct = lambda v: f"{v:.2%}"

    def row(label, ret, bench, r):
        rows = {}
        for pn, (a, b) in {"样本内": (pd.Timestamp(json.load(open(PROC / "model_choice.json"))["first_pred"]), S["is_end"]),
                           "历史验证期": oos}.items():
            p = perf(ret.loc[a:b], bench, r["turnover"].loc[a:b], r["cost"].loc[a:b])
            rows[pn] = p
        return {"方案": label, "样本内超额": pct(rows["样本内"]["超额年化"]), "样本内 IR": f"{rows['样本内']['信息比率']:.2f}",
                "验证期超额": pct(rows["历史验证期"]["超额年化"]), "验证期 IR": f"{rows['历史验证期']['信息比率']:.2f}",
                "验证期跟踪误差": pct(rows["历史验证期"]["跟踪误差"]), "验证期最大回撤": pct(rows["历史验证期"]["最大回撤"]),
                "验证期换手": pct(rows["历史验证期"]["年化单边换手"])}

    ro, rn = old["results"][om], new["results"][nm]
    # counterfactual: v6 code and data, v5 lambda
    data = load_dataset()
    fret = pd.read_parquet(PROC / "factor_returns.parquet")
    resid = pd.read_parquet(PROC / "specific_returns.parquet")
    mkt = Market.from_long(load_long())
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    fp = pd.Timestamp(json.load(open(PROC / "model_choice.json"))["first_pred"])
    sc = scores[new["best"]][scores.index.get_level_values(0) >= fp]
    cf, _ = optimised_backtest(sc, data, fret, resid, mkt, {**CFG["portfolio"], "risk_aversion": float(old["lam"])},
                               progress=False)
    table = pd.DataFrame([
        row(f"v5 ({om}, λ={old['lam']:g}), v5 基准", ro["ret"], b_old, ro),
        row(f"v5 收益, 改用 v6 基准", ro["ret"], b_new, ro),
        row(f"v6 代码, 保持 λ={old['lam']:g}", cf["ret"], b_new, cf),
        row(f"v6 ({nm}, λ={new['lam']:g}, 同一规则重新校准)", rn["ret"], b_new, rn),
    ])
    yr = lambda x: (1 + x.loc[:S["oos_end"]]).groupby(x.loc[:S["oos_end"]].index.year).prod() - 1
    bdiff = pd.DataFrame({"v5 基准": yr(b_old), "v6 基准": yr(b_new)})
    bdiff["差异"] = bdiff["v6 基准"] - bdiff["v5 基准"]
    hold_o, hold_n = ro["holdings"], rn["holdings"]
    ov = []
    for d in sorted(set(hold_o) & set(hold_n)):
        a, b = hold_o[d], hold_n[d]
        idx = a.index.union(b.index)
        ov.append(np.minimum(a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)).sum())
    calib_new = new["calib"]
    md = ["# v5 → v6 修订差异 (自动生成)", "",
          f"对比: v5 快照 `{old_dir}` 与当前运行。两者使用同一份原始数据。", "",
          "## 主要绩效 (逐步拆分)", "",
          "第 2 行只换基准口径, 第 3 行再换成 v6 代码但保持 v5 的 λ, 第 4 行是 v6 按同一规则 (样本内事前跟踪误差最接近 4%) 重新校准 λ 的正式结果。", "",
          table.to_markdown(index=False), "",
          f"λ 变化的原因: 规则是'样本内平均事前跟踪误差最接近 4%'。v5 时 λ=5 和 λ=7 的距离分别是 "
          f"{abs(old['calib'].loc[5.0, 'ex_ante_te'] - 0.04):.4f} 和 {abs(old['calib'].loc[7.0, 'ex_ante_te'] - 0.04):.4f}, "
          f"v6 是 {abs(new['calib'].loc[5.0, 'ex_ante_te'] - 0.04):.4f} 和 {abs(new['calib'].loc[7.0, 'ex_ante_te'] - 0.04):.4f}: "
          "两者几乎打平, 风险模型的输入检查 (新行业不按零风险处理、特异方差下限、半正定投影) 让事前跟踪误差略有变化, 就翻转了选择。"
          "这是又一个边缘决策; λ 的选择没有使用历史验证期的结果。", "",
          "v6 样本内校准表 (仅参考):", "", calib_new.round(4).to_markdown(), "",
          "## 基准修正 (前一交易日流通市值) 的年度影响", "", bdiff.map(pct).to_markdown(), "",
          f"2016 年的差异来自 v5 漏掉了基准第一天: v5 先按成分股过滤再取'上一行'市值, 每只股票在指数中的第一天都没有权重, "
          f"所以 v5 的基准从 2016-01-05 开始, 漏掉了 2016-01-04 (熔断日, v6 基准当天 {b_new.iloc[0]:.2%})。"
          "其他年份的差异来自股票重新进入指数的当天。组合回测从 2018 年开始, 这项修正对策略绩效的影响很小 (见上表第 2 行)。", "",
          f"## 持仓变化", "", f"v5 与 v6 每期成交后持仓的权重重合度: 平均 {np.mean(ov):.1%}, 最低 {np.min(ov):.1%}。", ""]
    (REPORTS / "revision_v6.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
