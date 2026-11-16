"""Market-data quality audit. Produces reports/data_audit.md and figures.

Every check returns a number that is written into the report, so a reader can
see exactly how dirty the raw data is and what the cleaning rules did.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CFG, FIG, RAW, REPORTS
from .panel import load_index, load_meta, reconstruct_benchmark
from .plotting import setup_style


def audit(long: pd.DataFrame) -> dict:
    setup_style()
    uni, ind, basic = load_meta()
    cal = pd.to_datetime(pd.read_csv(RAW / "trade_calendar.csv").date)
    res: dict[str, object] = {}
    n = len(long)
    res["rows"] = n
    res["stocks_with_data"] = long.code.nunique()
    res["universe_stocks"] = uni.code.nunique()
    res["missing_stocks"] = sorted(set(uni.code) - set(long.code))

    # 1. duplicates & calendar alignment
    res["duplicate_rows"] = int(long.duplicated(["code", "date"]).sum())
    cal_in = cal[(cal >= long.date.min()) & (cal <= long.date.max())]
    in_range = long.date >= cal.min()      # calendar file starts at universe_start
    res["dates_not_in_calendar"] = int((in_range & ~long.date.isin(cal)).sum())
    span = long.groupby("code").date.agg(["min", "max", "count"])
    expect = span.apply(lambda r: ((cal_in >= r["min"]) & (cal_in <= r["max"])).sum(), axis=1)
    gaps = (expect - span["count"]).clip(lower=0)
    res["stocks_with_calendar_gaps"] = int((gaps > 0).sum())
    res["missing_stock_days"] = int(gaps.sum())

    t = long[long.traded]
    # 2. OHLC integrity
    bad_ohlc = (t.low > t[["open", "close"]].min(axis=1) + 1e-6) | \
               (t.high < t[["open", "close"]].max(axis=1) - 1e-6) | (t.low <= 0)
    res["bad_ohlc_rows"] = int(bad_ohlc.sum())

    # 3. cross-source validation against baostock (exchange preclose & true ST flag)
    res.update(cross_validate(long))

    # 4. corporate actions: preclose differs from previous close
    prev_close = long.groupby("code").close.shift()
    ca = (long.preclose - prev_close).abs() / prev_close > 0.001
    res["corporate_action_days"] = int(ca.sum())

    # 5. returns beyond price limit (should only happen on IPO / relisting days)
    lim = np.where(t.isST == 1, 0.05, 0.10)
    lim = np.where(t.code.str.startswith(("sh.688",)) |
                   (t.code.str.startswith(("sz.300", "sz.301")) & (t.date >= "2020-08-24")), 0.20, lim)
    beyond = t.ret.abs() > lim + 0.003
    first_days = t.groupby("code").cumcount() < 5
    res["ret_beyond_limit"] = int(beyond.sum())
    res["ret_beyond_limit_excl_first5d"] = int((beyond & ~first_days).sum())

    # 6. suspension, ST, limit hits
    res["suspended_ratio"] = float((~long.traded).mean())
    res["st_ratio"] = float((long.isST == 1).mean())
    res["limit_up_ratio"] = float(t.limit_up.mean())
    res["limit_down_ratio"] = float(t.limit_down.mean())
    res["one_price_limit_days"] = int((t.one_price_up | t.one_price_down).sum())

    # 7. fundamental-field completeness
    t_uni = t[t.date >= CFG["data"]["universe_start"]]
    for c in ["ep", "bp", "sp", "roe", "np_yoy", "turn"]:
        res[f"missing_{c}"] = float(t_uni[c].isna().mean())
    res["negative_ep"] = float((t_uni.ep < 0).mean())
    res["median_report_lag_days"] = float(t_uni.report_lag_days.median())
    res["delisted_in_universe"] = int(basic.reindex(uni.code.unique()).outDate.notna().sum())
    res["missing_industry"] = float((ind.reindex(long.code.unique()) == "未分类").mean())
    res["missing_ipo_date"] = int(basic.reindex(long.code.unique()).ipoDate.isna().sum())

    # 8. reconstruction check: float-mcap-weighted CSI800 member return vs official TR index
    uni["month"] = pd.PeriodIndex(uni.month, freq="M")
    lm = long.assign(month=long.date.dt.to_period("M"))
    lm = lm.merge(uni[["month", "code"]].drop_duplicates(), on=["month", "code"])
    rec = reconstruct_benchmark(long)
    bench = load_index(CFG["benchmark_tr"]).ret.reindex(rec.index)
    yr = lambda x: (1 + x).groupby(x.index.year).prod() - 1
    res["recon_yearly_gap"] = (yr(rec) - yr(bench)).round(4).to_dict()
    res["recon_corr_with_csi800"] = float(rec.corr(bench))
    res["recon_te_annual"] = float((rec - bench).std() * np.sqrt(252))

    # figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cov = lm.groupby("month").code.nunique()
    want = uni.groupby("month").code.nunique()
    pd.DataFrame({"universe": want, "with data": cov}).plot(ax=axes[0, 0], title="Point-in-time CSI800 coverage")
    long.groupby(long.date.dt.to_period("M")).traded.apply(lambda s: 1 - s.mean()).plot(
        ax=axes[0, 1], title="Share of suspended stock-days")
    t.groupby(t.date.dt.to_period("M")).limit_up.mean().plot(ax=axes[1, 0], label="limit up")
    t.groupby(t.date.dt.to_period("M")).limit_down.mean().plot(ax=axes[1, 0], label="limit down",
                                                              title="Limit-hit frequency")
    axes[1, 0].legend()
    pd.DataFrame({"reconstructed": (1 + rec).cumprod(), "CSI800 index": (1 + bench).cumprod()}).plot(
        ax=axes[1, 1], title=f"Index reconstruction (corr={res['recon_corr_with_csi800']:.3f})")
    plt.tight_layout()
    plt.savefig(FIG / "data_audit.png", dpi=130)
    plt.close()
    write_report(res)
    return res


def cross_validate(long: pd.DataFrame) -> dict:
    """Compare Sina-derived panel with independent baostock bars for the overlapping stocks."""
    files = sorted((RAW / "daily").glob("*.parquet"))
    if not files:
        print("WARNING: data/raw/daily is empty -> cross-source validation skipped "
              "(run scripts/01_fetch_data.py to fetch the baostock sample)", flush=True)
        return {}
    bs = pd.concat(pd.read_parquet(f) for f in files)
    bs["date"] = pd.to_datetime(bs.date)
    for c in ["close", "preclose", "pctChg", "isST", "volume"]:
        bs[c] = pd.to_numeric(bs[c].replace("", np.nan), errors="coerce")
    m = long.merge(bs[["code", "date", "close", "preclose", "pctChg", "isST", "volume"]],
                   on=["code", "date"], suffixes=("", "_bs"))
    tr = m[(m.tradestatus == 1) & (m.volume_bs > 0) & m.preclose.notna()]   # IPO day: Sina has no preclose
    ret_diff = (tr.ret * 100 - tr.pctChg).abs()
    st_true, st_pred = m.isST_bs == 1, m.isST == 1
    # split-half stability: the inference parameters were tuned on this whole sample, so this is NOT an
    # independent holdout; a true holdout needs ST flags for stocks never used in tuning (not available)
    import zlib
    half = m.code.map(lambda c: zlib.crc32(c.encode()) % 2)
    halves = {}
    for h in (0, 1):
        t_, p_ = st_true[half == h], st_pred[half == h]
        halves[h] = (float((t_ & p_).sum() / max(p_.sum(), 1)), float((t_ & p_).sum() / max(t_.sum(), 1)))
    return {
        "st_half_a": halves[0], "st_half_b": halves[1],
        "xval_stocks": int(m.code.nunique()), "xval_rows": len(m),
        "xval_close_match": float(((m.close - m.close_bs).abs() < 0.006).mean()),
        "xval_ret_match_0.1pp": float((ret_diff < 0.1).mean()),
        "xval_ret_corr": float(tr.ret.corr(tr.pctChg / 100)),
        "st_true_ratio": float(st_true.mean()),
        "st_precision": float((st_true & st_pred).sum() / max(st_pred.sum(), 1)),
        "st_recall": float((st_true & st_pred).sum() / max(st_true.sum(), 1)),
    }


def write_report(r: dict):
    pct = lambda x: f"{x:.2%}"
    lines = [
        "# 数据质量审计报告 (Data Quality Audit)", "",
        f"- 数据源: 新浪日线 (不复权 + 后复权因子) / 东方财富季度业绩报表 (含公告日) / baostock (股票池、行业、上市日期、指数); 样本 {CFG['data']['start']} ~ {CFG['data']['end']}",
        f"- 股票池: 每月时点 CSI800 成分股 (沪深300 + 中证500), 共 {r['universe_stocks']} 只出现过; 有数据 {r['stocks_with_data']} 只",
        f"- 总记录数: {r['rows']:,}", "",
        "| 检查项 | 结果 | 处理方式 |", "|---|---|---|",
        f"| 缺失股票 | {len(r['missing_stocks'])} 只 {r['missing_stocks'][:5]} | 剔除 |",
        f"| 重复 (code,date) | {r['duplicate_rows']} | 保留最后一条 |",
        f"| 非交易日记录 | {r['dates_not_in_calendar']} | 剔除 |",
        f"| 交易日缺口 (上市后缺失日) | {r['stocks_with_calendar_gaps']} 只 / {r['missing_stock_days']:,} 个股票日 | 视作停牌, 收益记 0 |",
        f"| OHLC 不一致 (low>open/close 等) | {r['bad_ohlc_rows']} | 标记, 不用于开高低价因子 |",
        f"| 除权除息日 (复权因子变化) | {r['corporate_action_days']:,} | preclose = 前收 × f(t-1)/f(t) |",
        f"| 超出涨跌幅限制的收益 | {r['ret_beyond_limit']} (剔除上市前5日后 {r['ret_beyond_limit_excl_first5d']}) | 新股过滤 (上市<250日剔除) |",
        f"| 停牌股票日占比 | {pct(r['suspended_ratio'])} | 调仓日停牌不可买卖 |",
        f"| ST 股票日占比 | {pct(r['st_ratio'])} | 剔除 |",
        f"| 涨停 / 跌停 占比 | {pct(r['limit_up_ratio'])} / {pct(r['limit_down_ratio'])} | 涨停不可买、跌停不可卖 |",
        f"| 一字板天数 | {r['one_price_limit_days']:,} | 不可交易 |",
        f"| EP / BP / SP / ROE / 净利同比 缺失 | {pct(r['missing_ep'])} / {pct(r['missing_bp'])} / {pct(r['missing_sp'])} / {pct(r['missing_roe'])} / {pct(r['missing_np_yoy'])} | 行业中位数填充 |",
        f"| 换手率缺失 | {pct(r['missing_turn'])} | — |",
        f"| 负 EP (亏损) 占比 | {pct(r['negative_ep'])} | 用 EP 而非 PE, 保持单调 |",
        f"| 财报滞后 (交易日 - 报告期, 中位数) | {r['median_report_lag_days']:.0f} 天 | 按公告日对齐, 公告次日起可用 |",
        f"| 股票池中后来退市的股票 | {r['delisted_in_universe']} 只 | 新浪无退市股数据 → 残余幸存者偏差 (见局限) |",
        f"| 行业缺失 | {pct(r['missing_industry'])} | 归入 '未分类' |",
        "",
        "## 跨数据源交叉验证 (新浪 vs baostock)",
        f"在两个数据源都有的 {r.get('xval_stocks', 0)} 只股票、{r.get('xval_rows', 0):,} 个股票日上: "
        f"收盘价一致率 **{pct(r.get('xval_close_match', np.nan))}**, 日收益误差<0.1pp 的比例 **{pct(r.get('xval_ret_match_0.1pp', np.nan))}**, "
        f"收益相关系数 **{r.get('xval_ret_corr', np.nan):.4f}**。",
        f"ST 推断 (近60日收盘/最高/最低触及±5%涨跌停价≥3次, 且无>5.5%波动; 参数在该样本上网格搜索) 对照 baostock 真实 ST 标记: "
        f"精确率 **{pct(r.get('st_precision', np.nan))}**, 召回率 **{pct(r.get('st_recall', np.nan))}** (真实 ST 占比 {pct(r.get('st_true_ratio', np.nan))})。"
        f"按股票代码随机分成两半: A 半精确率 / 召回率 {pct(r['st_half_a'][0]) if 'st_half_a' in r else '-'} / {pct(r['st_half_a'][1]) if 'st_half_a' in r else '-'}, "
        f"B 半 {pct(r['st_half_b'][0]) if 'st_half_b' in r else '-'} / {pct(r['st_half_b'][1]) if 'st_half_b' in r else '-'}。"
        "注意: 推断参数是在整个样本上网格搜索得到的, 这只是稳定性检查, **不是独立持出样本**; 独立验证需要另一批从未参与调参的股票的真实 ST 记录, 目前没有。", "",
        "## 指数复原检验",
        f"用成分股流通市值加权 (含分红) 收益复原中证800, 对比官方中证800全收益指数 (H00906): 日收益相关系数 **{r['recon_corr_with_csi800']:.4f}**, "
        f"年化跟踪误差 {pct(r['recon_te_annual'])}。 相关系数接近 1 说明收益、市值、成分股数据整体可信; "
        "但两者每年有 3-5% 的系统性差异, 主要因为官方指数用**自由流通市值** (剔除控股股东等战略持股) 加权并分级靠档, "
        "而这里只有流通市值, 大型国企 (银行、石油等) 权重被高估。逐年差异 (复原 − 官方): "
        + ", ".join(f"{k}: {v:+.1%}" for k, v in r["recon_yearly_gap"].items()) + "。",
        "**处理**: 组合优化器的基准权重就是流通市值权重, 所以策略 alpha 以复原基准衡量 (主口径), 同时报告相对官方全收益指数的超额 (副口径); "
        "两者之差是基准构造差异, 不是选股能力。", "",
        "## 已知局限",
        "- 行业分类为 baostock 提供的当前证监会行业分类 (二级, 约 76 个; 非时点数据), 存在轻微前视偏差;",
        "- 新浪不提供已退市股票数据: 股票池中已退市股票缺失, 存在残余幸存者偏差 (会轻微高估收益);",
        "- ST 为推断值 (见交叉验证), 创业板/科创板 20% 涨跌幅下无法识别 ST;",
        "- 东方财富报表只给'最新公告日期' (常为次年年报重述比较数据的日期), 首次披露日按 min(最新公告日, 法定截止日) 估计 "
        "(一季报 4/30, 半年报 8/31, 三季报 10/31, 年报次年 4/30); 对逾期披露的极少数公司存在轻微前视;",
        "- 报表数值可能是重述后的数值而非首次披露值, 存在轻微前视。", "",
        "![audit](figures/data_audit.png)",
    ]
    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
