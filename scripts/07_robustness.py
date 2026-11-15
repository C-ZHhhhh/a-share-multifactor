"""Step 7: robustness, data-sensitivity and capacity tests of the main strategy.

  A. cost scenarios          same trades, cost multiplier 0 / 0.5 / 1 / 2 / 3
  -  parameter sensitivity   the other two lambdas and the other two models (all chosen on IS only)
  B. exposure constraints    extra limits on beta & volatility, all styles, tighter industry bands
  C. sample splits           in-sample ends 2020-12 / 2022-12: factor screen, model choice and lambda redone
  D. data sensitivity        fundamentals one month staler, fundamentals dropped, 10% of fundamentals blanked;
                             benchmark coverage (missing delisted stocks) and custom vs official benchmark
  E. block bootstrap         confidence intervals of the out-of-sample active return and IR
  F. capacity                participation in daily traded value and square-root market impact by AUM

Optimisation variants run in parallel processes and are cached in data/processed/robustness/; a cached
variant is reused only if it is newer than the main backtest (pass --force to recompute anyway). Writes reports/robustness.md and
data/processed/robustness.json (read by the HTML report).
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mf.config import CFG, PROC, REPORTS
from mf.dataset import label_window, load_dataset
from mf.factor_test import ic_series, ic_stats, run_factor_tests, select_factors
from mf.metrics import perf
from mf.panel import load_benchmarks, load_long, to_wide

CACHE = PROC / "robustness"
FUND = ["ep", "bp", "sp", "roe", "gpm", "np_yoy", "rev_yoy"]
_W = {}  # per-process shared state


def _init():
    from mf.backtest import Market
    _W["data"] = load_dataset()
    _W["fret"] = pd.read_parquet(PROC / "factor_returns.parquet")
    _W["resid"] = pd.read_parquet(PROC / "specific_returns.parquet")
    _W["mkt"] = Market.from_long(load_long())
    _W["scores"] = pd.read_parquet(PROC / "model_scores.parquet")


def _predict(model: str, data: pd.DataFrame, chosen: list, signs: dict) -> pd.Series:
    from mf.models import predict_gbdt, predict_icir, predict_ridge
    if model == "Ridge":
        return predict_ridge(data, chosen)
    if model == "ICIR加权":
        return predict_icir(data, chosen, signs)
    return predict_gbdt(data)


def _summ(bt, info, bench, oos_start, oos_end, is_start=None, is_end=None, score=None, data=None) -> dict:
    out = {}
    for per, (a, b) in {"IS": (is_start, is_end), "OOS": (oos_start, oos_end)}.items():
        if a is None:
            continue
        r = bt["ret"].loc[a:b]
        if len(r) < 20:
            continue
        p = perf(r, bench, bt["turnover"].loc[a:b], bt["cost"].loc[a:b])
        out[per] = {k: float(p[k]) for k in ["超额年化", "信息比率", "跟踪误差", "年化单边换手", "年化成本拖累"]}
        if score is not None:
            d = data.set_index(["date", "code"]).join(score.rename("pred"), how="inner").reset_index()
            out[per]["RankIC"] = float(ic_stats(ic_series(d[label_window(d, a, b)], "pred"))["IC均值"])
    out["relaxed_months"] = int((info.relax > 1).sum()) if info is not None else 0
    out["ret"] = bt["ret"]
    return out


def task(spec: dict) -> dict:
    """One optimisation variant (runs in a worker process)."""
    f = CACHE / f"{spec['name']}.pkl"
    fresh = f.exists() and f.stat().st_mtime > (PROC / "backtests.pkl").stat().st_mtime   # newer than the main run
    if fresh and not spec.get("force"):
        return pickle.load(open(f, "rb"))
    if not _W:
        _init()
    from mf.portfolio import optimised_backtest
    data, fret, resid, mkt = _W["data"], _W["fret"], _W["resid"], _W["mkt"]
    S, base = CFG["split"], spec["base"]
    bench = load_benchmarks().rec
    pcfg = {**CFG["portfolio"], "risk_aversion": base["lam"], **spec.get("pcfg", {})}
    first_pred = pd.Timestamp(base["first_pred"])

    if spec["kind"] == "constraint":           # also used for lambda / model sensitivity
        score = _W["scores"][spec.get("model", base["best"])]
        score = score[score.index.get_level_values(0) >= first_pred]
        bt, info = optimised_backtest(score, data, fret, resid, mkt, pcfg, spec["name"], progress=False)
        res = _summ(bt, info, bench, S["oos_start"], S["oos_end"], first_pred, S["is_end"])

    elif spec["kind"] == "data":
        d = data.copy()
        cols = [f"z_{c}" for c in FUND]
        chosen, signs = base["chosen"], base["signs"]
        if spec["mode"] == "lag":            # fundamentals available one month later than assumed
            d = d.sort_values(["code", "date"])
            d[cols] = d.groupby("code")[cols].shift(1).fillna(0.0)
        elif spec["mode"] == "drop":         # price-volume factors only
            chosen = [c for c in chosen if c not in FUND]
            signs = {k: v for k, v in signs.items() if k in chosen}
            d[cols] = 0.0                    # GBDT also loses them
        elif spec["mode"] == "blank":        # 10% of fundamental values lost (set to the cross-sectional mean)
            rng = np.random.default_rng(0)
            m = rng.random((len(d), len(cols))) < 0.10
            d[cols] = d[cols].mask(m, 0.0)
        score = _predict(base["best"], d, chosen, signs)
        score = score[score.index.get_level_values(0) >= first_pred]
        bt, info = optimised_backtest(score, data, fret, resid, mkt, pcfg, spec["name"], progress=False)
        res = _summ(bt, info, bench, S["oos_start"], S["oos_end"], first_pred, S["is_end"], score, data)
        res["n_factors"] = len(chosen)

    elif spec["kind"] == "split":            # redo every in-sample decision with a different split
        is_end, oos_start = spec["is_end"], spec["oos_start"]
        CFG["split"] = {**S, "is_end": is_end, "oos_start": oos_start}
        tab, _ = run_factor_tests(data, (S["is_start"], is_end))
        chosen, signs, _ = select_factors(tab, data)
        scores = {m: _predict(m, data, chosen, signs) if m != "GBDT" else _W["scores"]["GBDT"].dropna()
                  for m in ["ICIR加权", "Ridge", "GBDT"]}
        icir = {}
        for m, sc in scores.items():
            dd = data.set_index(["date", "code"]).join(sc.rename("pred"), how="inner").reset_index()
            icir[m] = ic_stats(ic_series(dd[label_window(dd, first_pred, is_end)], "pred"))["ICIR(年化)"]
        best = max(icir, key=icir.get)
        score = scores[best][scores[best].index.get_level_values(0) >= first_pred]
        calib = {}
        for lam in (2.0, 10.0, 50.0):
            bt, _ = optimised_backtest(score, data, fret, resid, mkt, {**pcfg, "risk_aversion": lam},
                                       f"{spec['name']} λ={lam}", end=is_end, progress=False)
            calib[lam] = perf(bt["ret"], bench)["信息比率"]
        lam = max(calib, key=calib.get)
        bt, info = optimised_backtest(score, data, fret, resid, mkt, {**pcfg, "risk_aversion": lam},
                                      spec["name"], progress=False)
        res = _summ(bt, info, bench, oos_start, S["oos_end"], first_pred, is_end, score, data)
        res.update({"chosen": chosen, "best": best, "lam": lam})
    res.update({"name": spec["name"], "group": spec["group"], "label": spec["label"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return res


def block_bootstrap(x: np.ndarray, block: int = 21, n: int = 5000, seed: int = 0) -> np.ndarray:
    """Circular block bootstrap of daily active returns -> (annualised mean, IR) per replicate."""
    rng = np.random.default_rng(seed)
    T = len(x)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T] % T
    s = x[idx]
    mu, sd = s.mean(axis=1) * 252, s.std(axis=1, ddof=1) * np.sqrt(252)
    return np.c_[mu, mu / sd]


def capacity(res_main: dict, long: pd.DataFrame, oos: tuple, aums=(1e8, 5e8, 2e9, 1e10, 5e10),
             y: float = 1.0) -> pd.DataFrame:
    """Square-root impact: cost_i = y * sigma_i * sqrt(Q_i / ADV_i), sigma = 20d daily vol, ADV = 20d mean
    traded value, both up to the signal date. Q_i = |trade weight| * AUM * NAV on the execution day."""
    ret, amt = to_wide(long, "ret"), to_wide(long, "amount")
    sig = ret.rolling(20, min_periods=10).std()
    adv = amt.where(amt > 0).rolling(20, min_periods=10).mean()
    dates = ret.index
    rows, a, b = [], pd.Timestamp(oos[0]), pd.Timestamp(oos[1])
    trades = {d: t for d, t in res_main["trades"].items() if a <= d <= b}
    yrs = len(res_main["ret"].loc[a:b]) / 252
    for aum in aums:
        part, drag, n_big, n_all = [], 0.0, 0, 0
        for d, t in trades.items():
            s_ = dates[dates.searchsorted(d) - 1]                 # signal date (information set)
            q = t.abs() * aum
            v = adv.loc[s_].reindex(t.index)
            pr = (q / v).replace([np.inf], np.nan)
            part += list(pr.dropna().values)
            n_big += int((pr > 0.10).sum()); n_all += int(pr.notna().sum())
            imp = y * sig.loc[s_].reindex(t.index).fillna(sig.loc[s_].median()) * np.sqrt(pr.fillna(pr.max()))
            drag += float((t.abs() * imp).sum())
        part = np.array(part)
        rows.append({"规模(亿元)": aum / 1e8, "参与率中位数(按笔)": np.median(part), "参与率95分位": np.quantile(part, 0.95),
                     "参与率>10%的交易占比": n_big / max(n_all, 1), "冲击成本(年化)": drag / yrs})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    force = "--force" in sys.argv
    S = CFG["split"]
    bt_main = pickle.load(open(PROC / "backtests.pkl", "rb"))
    choice = json.load(open(PROC / "model_choice.json"))
    sel = json.load(open(PROC / "selected_factors.json"))
    best, lam = bt_main["best"], bt_main["lam"]
    main = f"组合优化({best})"
    base = {"best": best, "lam": lam, "first_pred": choice["first_pred"], "chosen": sel["selected"],
            "signs": {k: int(v) for k, v in sel["signs"].items()}}
    B = load_benchmarks()
    bench = B.rec
    oos = (S["oos_start"], S["oos_end"])
    first_pred = pd.Timestamp(choice["first_pred"])

    other_lams = [l for l in (2.0, 10.0, 50.0) if l != lam]
    other_models = [m for m in ("ICIR加权", "Ridge", "GBDT") if m != best]
    specs = [
        *[{"name": f"lam_{l:g}", "group": "参数", "label": f"λ = {l:g} (样本内选定 λ = {lam:g})", "kind": "constraint",
           "pcfg": {"risk_aversion": l}} for l in other_lams],
        *[{"name": f"model_{m}", "group": "参数", "label": f"模型换成 {m} (样本内选定 {best})", "kind": "constraint",
           "model": m} for m in other_models],
        {"name": "con_beta_vol", "group": "约束", "label": "Beta 与波动率暴露 ≤ 0.2σ", "kind": "constraint",
         "pcfg": {"style_limits": {"beta": 0.2, "volatility": 0.2}}},
        {"name": "con_all_styles", "group": "约束", "label": "全部 6 个风格暴露 ≤ 0.2σ", "kind": "constraint",
         "pcfg": {"style_limits": {k: 0.2 for k in ["beta", "momentum", "volatility", "value", "liquidity"]}}},
        {"name": "con_industry_2", "group": "约束", "label": "行业偏离 ±2% (原 ±5%)", "kind": "constraint",
         "pcfg": {"industry_dev": 0.02}},
        {"name": "split_2020", "group": "切分", "label": "样本内截至 2020-12, 样本外 2021-01 起", "kind": "split",
         "is_end": "2020-12-31", "oos_start": "2021-01-01"},
        {"name": "split_2022", "group": "切分", "label": "样本内截至 2022-12, 样本外 2023-01 起", "kind": "split",
         "is_end": "2022-12-31", "oos_start": "2023-01-01"},
        {"name": "data_lag", "group": "数据", "label": "财报因子再晚一个月可得", "kind": "data", "mode": "lag"},
        {"name": "data_drop", "group": "数据", "label": "去掉全部财报因子 (只用量价)", "kind": "data", "mode": "drop"},
        {"name": "data_blank", "group": "数据", "label": "随机丢失 10% 的财报因子值", "kind": "data", "mode": "blank"},
    ]
    for s in specs:
        s.update({"base": base, "force": force})
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(task, specs))
        for r in results:
            print(f"{r['label']:32s} OOS ex={r['OOS']['超额年化']:.2%} IR={r['OOS']['信息比率']:.2f}")

    # ---------------- A. cost scenarios (same trades)
    from mf.backtest import Market, run_backtest
    long = load_long()
    mkt = Market.from_long(long)
    targets = bt_main["opt_targets"]
    top_name = f"Top{CFG['portfolio']['topn']}等权({best})"
    from mf.portfolio import topn_targets
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    top_t = topn_targets(scores[best][scores.index.get_level_values(0) >= first_pred], CFG["portfolio"]["topn"])
    cost_rows = []
    for m in (0.0, 0.5, 1.0, 2.0, 3.0):
        for name, tg in {main: targets, top_name: top_t}.items():
            bt = run_backtest(tg, mkt, cost_mult=m)
            p = perf(bt["ret"].loc[oos[0]:oos[1]], bench, bt["turnover"].loc[oos[0]:oos[1]], bt["cost"].loc[oos[0]:oos[1]])
            cost_rows.append({"成本倍数": m, "组合": name, "超额年化": p["超额年化"], "信息比率": p["信息比率"],
                              "年化成本拖累": p["年化成本拖累"]})
    cost_df = pd.DataFrame(cost_rows)
    print(cost_df)

    # ---------------- E. bootstrap
    r_main = bt_main["results"][main]["ret"]
    boot = {}
    for per, (a, b) in {"样本内": (first_pred, S["is_end"]), "样本外": oos}.items():
        act = (r_main - bench.reindex(r_main.index).fillna(0)).loc[a:b].values
        bs = block_bootstrap(act)
        boot[per] = {"主动收益年化(算术)": act.mean() * 252, "IR": act.mean() / act.std(ddof=1) * np.sqrt(252),
                     "主动收益 95% 区间": (np.quantile(bs[:, 0], 0.025), np.quantile(bs[:, 0], 0.975)),
                     "IR 95% 区间": (np.quantile(bs[:, 1], 0.025), np.quantile(bs[:, 1], 0.975)),
                     "P(主动收益≤0)": float((bs[:, 0] <= 0).mean()), "年数": len(act) / 252,
                     "hist": np.histogram(bs[:, 1], bins=40)}
    print({k: {kk: vv for kk, vv in v.items() if kk != "hist"} for k, v in boot.items()})

    # ---------------- F. capacity
    cap = capacity(bt_main["results"][main], long, oos)
    ex_oos = perf(r_main.loc[oos[0]:oos[1]], bench)["超额年化"]
    cap["扣冲击后超额(近似)"] = ex_oos - cap["冲击成本(年化)"]
    print(cap)

    # ---------------- D. benchmark & coverage
    uni = pd.read_csv(PROC.parent / "raw" / "universe_monthly.csv")
    have = set(long.code.unique())
    cov = uni.assign(has=uni.code.isin(have)).groupby("month").has.mean()
    cov.index = pd.PeriodIndex(cov.index, freq="M")
    rec_y = (1 + B.rec).groupby(B.index.year).prod() - 1
    tri_y = (1 + B.tri.fillna(0)).groupby(B.index.year).prod() - 1
    bench_y = pd.DataFrame({"自定义基准": rec_y, "官方中证800全收益": tri_y, "差异": rec_y - tri_y})
    miss = 1 - cov
    miss_y = miss.groupby(miss.index.year).mean()
    sens = pd.DataFrame({"成分股缺数据占比(按数量)": miss_y,
                         **{f"若缺失股年化跑输 {x:.0%}: 基准高估": miss_y * x for x in (0.2, 0.5)}})

    # ---------------- write
    base_row = {"label": "基准方案", "group": "基准"}
    pm = perf(r_main.loc[oos[0]:oos[1]], bench, bt_main["results"][main]["turnover"].loc[oos[0]:oos[1]],
              bt_main["results"][main]["cost"].loc[oos[0]:oos[1]])
    rows = [{"组别": "基准方案", "方案": f"{main}, λ={lam}", "样本外起点": oos[0], "超额年化": pm["超额年化"],
             "信息比率": pm["信息比率"], "跟踪误差": pm["跟踪误差"], "年化单边换手": pm["年化单边换手"], "备注": ""}]
    for r in results:
        note = []
        if "chosen" in r:
            note.append(f"{len(r['chosen'])} 个因子, 模型 {r['best']}, λ={r['lam']:g}")
        if "n_factors" in r:
            note.append(f"{r['n_factors']} 个因子")
        if "RankIC" in r["OOS"]:
            note.append(f"样本外 RankIC {r['OOS']['RankIC']:.3f}")
        if r["relaxed_months"]:
            note.append(f"{r['relaxed_months']} 个月放宽约束")
        start = next(s.get("oos_start", oos[0]) for s in specs if s["name"] == r["name"])
        rows.append({"组别": r["group"], "方案": r["label"], "样本外起点": start, "超额年化": r["OOS"]["超额年化"],
                     "信息比率": r["OOS"]["信息比率"], "跟踪误差": r["OOS"]["跟踪误差"],
                     "年化单边换手": r["OOS"]["年化单边换手"], "备注": "; ".join(note)})
    var_df = pd.DataFrame(rows)
    pct = lambda v: f"{v:.2%}"
    vshow = var_df.copy()
    for c in ["超额年化", "跟踪误差", "年化单边换手"]:
        vshow[c] = vshow[c].map(pct)
    vshow["信息比率"] = vshow["信息比率"].map(lambda v: f"{v:.2f}")
    cshow = cost_df.pivot(index="成本倍数", columns="组合", values="超额年化").map(pct)
    bshow = pd.DataFrame({k: {"主动收益年化(算术)": pct(v["主动收益年化(算术)"]), "IR": f"{v['IR']:.2f}",
                              "主动收益 95% 区间": f"[{pct(v['主动收益 95% 区间'][0])}, {pct(v['主动收益 95% 区间'][1])}]",
                              "IR 95% 区间": f"[{v['IR 95% 区间'][0]:.2f}, {v['IR 95% 区间'][1]:.2f}]",
                              "P(主动收益≤0)": pct(v["P(主动收益≤0)"]), "年数": f"{v['年数']:.1f}"}
                          for k, v in boot.items()}).T
    capshow = cap.copy()
    for c in capshow.columns[1:]:
        capshow[c] = capshow[c].map(pct)
    capshow["规模(亿元)"] = capshow["规模(亿元)"].map(lambda v: f"{v:g}")
    md = [
        "# 稳健性、数据敏感性与容量检验 (自动生成)", "",
        f"主方案: {main}, λ={lam}; 主基准为自定义流通市值基准 (见 results.md)。除'切分'组外, 样本外均为 {oos[0]} ~ {oos[1]}。", "",
        "## 1. 方案对比 (样本外)", "",
        "- **参数**: 换成样本内没有选中的 λ 或模型。样本内的选择往往由很小的差距决定, 这一组看样本外结论是否依赖这些选择。",
        "- **约束**: 在原约束上增加风格暴露限制或收紧行业偏离, λ 与模型不变。",
        "- **切分**: 换一个样本内终点, 重新做因子筛选、模型选择和 λ 校准, 再看新的样本外。",
        "- **数据**: 财报因子更晚可得、完全去掉、或随机丢失 10%, 重新训练模型后回测, 检验结论是否依赖财报数据质量。", "",
        vshow.to_markdown(index=False), "",
        "## 2. 交易成本情景 (样本外超额年化, 同一组交易)", "",
        "成本倍数 1 = 佣金万2.5 + 滑点千1 + 印花税; 0 = 无成本。", "", cshow.to_markdown(), "",
        "## 3. 时间序列 Bootstrap (日度主动收益, 循环块长 21 天, 5000 次)", "",
        "块 Bootstrap 保留了收益的自相关和波动聚集。区间包含 0 说明在常用置信水平下不能拒绝 '没有超额'。", "",
        bshow.to_markdown(), "",
        "## 4. 容量: 成交额参与率与市场冲击", "",
        "按样本外每次调仓的实际交易, 假设资金规模为 AUM: 参与率 = 交易金额 / 该股过去 20 日日均成交额; "
        "冲击成本用平方根模型 σ·√(参与率), σ 为过去 20 日日波动率, 系数取 1。这部分是在已有千1滑点之外的额外成本。", "",
        capshow.to_markdown(index=False), "",
        "## 5. 基准与数据覆盖", "",
        "### 自定义基准 vs 官方中证800全收益 (分年)", "", bench_y.map(pct).to_markdown(), "",
        "两者差异来自加权方式 (流通市值 vs 自由流通市值 + 分级靠档 + 权重上限) 以及缺失的退市股票。"
        "所以本项目的超额应理解为 '相对自定义流通市值基准', 不能直接当作 '中证800 指数增强' 的超额。", "",
        "### 缺失退市股票的敏感性", "",
        "股票池中部分成分股 (主要是后来退市的股票) 拿不到行情, 基准和组合都无法包含它们。"
        "若这些股票在基准中的实际表现更差, 我们的基准会被高估, 超额被低估; 反之亦然。下表按数量占比给出影响上限 "
        "(退市股多为小市值, 按市值加权的实际影响更小)。", "",
        sens.map(pct).to_markdown(), "",
    ]
    (REPORTS / "robustness.md").write_text("\n".join(md), encoding="utf-8")

    payload = {
        "variants": var_df.to_dict(orient="records"),
        "costs": cost_df.to_dict(orient="records"),
        "bootstrap": {k: {"mean": v["主动收益年化(算术)"], "ir": v["IR"], "mean_ci": list(v["主动收益 95% 区间"]),
                          "ir_ci": list(v["IR 95% 区间"]), "p_le0": v["P(主动收益≤0)"], "years": v["年数"],
                          "hist": [v["hist"][0].tolist(), v["hist"][1].tolist()]} for k, v in boot.items()},
        "capacity": cap.to_dict(orient="records"),
        "bench_years": [{"year": int(y), **{k: float(v) for k, v in r.items()}} for y, r in bench_y.iterrows()],
        "coverage": {"min": float(cov.min()), "mean": float(cov.mean())},
    }
    json.dump(payload, open(PROC / "robustness.json", "w"), ensure_ascii=False, default=float)
    print("written", REPORTS / "robustness.md")
