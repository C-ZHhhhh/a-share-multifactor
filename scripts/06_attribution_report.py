"""Step 6: IS vs OOS evaluation, Brinson + factor attribution, figures and results tables."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mf.attribution import active_exposure_series, brinson, factor_attribution
from mf.config import CFG, FIG, PROC, REPORTS
from mf.dataset import load_dataset
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
        rows["基准: 复原中证800"] = perf(bench.reindex(span).fillna(0))
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
    axes[0].plot((1 + b).cumprod(), color="k", lw=1.4, label="基准 (复原中证800)")
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
    fa = factor_attribution(hold, sig_of, data, fret, active)
    expo = active_exposure_series(hold, sig_of, data)

    br_tabs, fa_tabs = {}, {}
    for pname, (a, bb) in periods.items():
        sub = br[(br.date >= a) & (br.date <= bb)]
        yrs = len(r.loc[a:bb]) / 252
        br_tabs[pname] = (sub[["allocation", "selection", "interaction"]].sum() / yrs)
        f = fa.loc[a:bb]
        fa_tabs[pname] = f.sum() / yrs
    br_df = pd.DataFrame(br_tabs).T
    br_df["合计"] = br_df.sum(axis=1)
    fa_df = pd.DataFrame(fa_tabs).T
    oos_br = br[br.date >= S["oos_start"]].groupby("industry")[["allocation", "selection"]].sum()
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
        "- 基准 (主口径): 时点中证800成分股按流通市值加权的全收益 (= 优化器跟踪的基准); 副口径: 官方中证800全收益指数 H00906, 见 '超额年化(vs官方全收益)' 一行",
        f"- 预测区间: {first_pred.date()} 起 (前 24 个月仅用于训练); 样本外: {S['oos_start']} ~ {S['oos_end']}",
        "- 成本: 佣金万2.5 双边 + 滑点千1 双边 + 印花税 (2023-08-28 前千1, 之后万5, 仅卖出)", "",
        "## 模型预测能力", "", model_eval.round(4).to_markdown(index=False), "",
        "## 风险厌恶系数校准 (仅样本内)", "", calib.round(3).to_markdown(), "",
        *sum([[f"## 绩效 — {k}", "", v.to_markdown(), ""] for k, v in perf_tabs.items()], []),
        f"## 分年度 — {main}", "", pct(yearly).to_markdown(), "",
        "![nav](figures/nav.png)", "", "![diag](figures/optimizer_diag.png)", "",
        "## Brinson 行业归因 (年化, 算术加总)", "", pct(br_df).to_markdown(), "",
        "样本外分行业 (累计):", "", pct(pd.concat([oos_br.head(5), oos_br.tail(5)])).to_markdown(), "",
        "## 因子收益归因 (年化主动收益贡献)", "", pct(fa_df).to_markdown(), "",
        "![attr](figures/attribution.png)", "",
    ]
    (REPORTS / "results.md").write_text("\n".join(md), encoding="utf-8")
    for k, v in perf_tabs.items():
        print(k); print(v.to_string())
    print(br_df); print(fa_df)
