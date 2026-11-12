"""Step 3: factor construction, preprocessing, single-factor tests, factor selection (IS only)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.config import CFG, PROC, REPORTS
from mf.dataset import build_dataset
from mf.factor_test import (ic_decay, plot_factor_tests, plot_ic_decay, run_factor_tests,
                            select_factors)
from mf.factors import FACTOR_INFO
from mf.panel import load_long

if __name__ == "__main__":
    data = build_dataset(load_long())
    print(f"dataset: {len(data):,} stock-months, {data.date.nunique()} months, "
          f"tradable share {data.tradable.mean():.1%}")
    S = CFG["split"]
    is_tab, is_extra = run_factor_tests(data, (S["is_start"], S["is_end"]))
    chosen, signs, corr = select_factors(is_tab, data)
    plot_factor_tests(is_tab, is_extra, corr, "is")
    is_data = data[(data.date >= S["is_start"]) & (data.date <= S["is_end"])]
    decay = {f: ic_decay(is_data, f"z_{f}") for f in chosen}
    plot_ic_decay(decay, "is")
    # OOS single-factor results: computed for the report only, never used for selection
    oos_tab, oos_extra = run_factor_tests(data, (S["oos_start"], S["oos_end"]))
    plot_factor_tests(oos_tab, oos_extra, corr, "oos")

    is_tab.to_csv(PROC / "factor_tests_is.csv")
    oos_tab.to_csv(PROC / "factor_tests_oos.csv")
    corr.to_csv(PROC / "factor_corr_is.csv")
    json.dump({"selected": chosen, "signs": signs}, open(PROC / "selected_factors.json", "w"), indent=2)

    show = ["类别", "IC均值", "ICIR(年化)", "t值", "IC>0占比", "多空年化", "多空夏普", "单调性", "因子自相关"]
    fmt = lambda t: t[show].round(3).to_markdown()
    md = ["# 单因子检验 (Single-Factor Tests)", "",
          "预处理: 行业中位数填充 → MAD(5倍) 去极值 → Z-score → 对证监会行业哑变量 + ln流通市值 回归取残差 (市值因子仅行业中性) → Z-score。",
          "检验: 月末截面 RankIC (Spearman), 标签为 T+1 日 VWAP 买入、持有至下月末 T+1 VWAP 的收益; 五分组等权月度收益。", "",
          "## 因子定义", "", "| 因子 | 类别 | 定义 |", "|---|---|---|",
          *[f"| {k} | {v[0]} | {v[1]} |" for k, v in FACTOR_INFO.items()], "",
          f"## 样本内 {S['is_start']} ~ {S['is_end']}", "", fmt(is_tab), "",
          f"**入选因子** (|t|≥2, |ICIR|≥0.5, 两两相关<0.6, 按 |ICIR| 贪心): "
          + ", ".join(f"{f}({'+' if signs[f] > 0 else '−'})" for f in chosen), "",
          "![q](figures/quantile_nav_is.png)", "", "![c](figures/cum_ic_is.png)", "",
          "![corr](figures/factor_corr_is.png)", "", "![decay](figures/ic_decay_is.png)", "",
          f"## 样本外 {S['oos_start']} ~ {S['oos_end']} (仅展示, 未参与筛选)", "", fmt(oos_tab), "",
          "![q](figures/quantile_nav_oos.png)", ""]
    (REPORTS / "factor_tests.md").write_text("\n".join(md), encoding="utf-8")
    print(is_tab[show].round(3).to_string())
    print("selected:", signs)
