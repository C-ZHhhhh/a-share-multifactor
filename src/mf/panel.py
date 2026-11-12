"""Build the clean daily panel (one row per stock-day) from raw downloads.

Sources: Sina daily bars + back-adjustment factors, Eastmoney quarterly reports,
baostock metadata (universe, industry, IPO date) and index bars.

Derived fields
--------------
tradestatus   0 on days the stock has no bar inside its listed life (suspension)
preclose      close(t-1) * f(t-1) / f(t): ex-rights reference price from the hfq factor
ret           close / preclose - 1  (total return incl. dividends & splits)
adj_close     back-adjusted price = close * f
vwap          amount / volume
float_mcap    close * float shares (Sina outstanding_share)
isST          inferred: hit the ±5% limit in the last 60 days and never moved >5.5%
limit_up/down, one_price_up/down   board- and ST-specific price-limit flags
ep, bp, sp, roe, gpm, np_yoy, rev_yoy   point-in-time fundamentals (see fundamentals())
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROC, RAW


# ------------------------------------------------------------------ metadata
def load_index(code: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / "index" / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df.close / df.preclose - 1
    return df.set_index("date").sort_index()


def load_meta() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (universe_monthly, industry by code, stock_basic)."""
    uni = pd.read_csv(RAW / "universe_monthly.csv")
    ind = pd.read_csv(RAW / "stock_industry.csv")
    ind = ind.set_index("code")["industry"].replace("", np.nan).fillna("未分类")
    basic = pd.read_csv(RAW / "stock_basic.csv")
    basic["ipoDate"] = pd.to_datetime(basic["ipoDate"], errors="coerce")
    basic["outDate"] = pd.to_datetime(basic["outDate"], errors="coerce")
    return uni, ind, basic.set_index("code")


def calendar() -> pd.DatetimeIndex:
    from .config import CFG
    return load_index(CFG["benchmark"]).index


# ------------------------------------------------------------------ prices
def limit_pct(code: pd.Series, date: pd.Series, is_st: pd.Series) -> np.ndarray:
    star = code.str.startswith("sh.688")
    gem = code.str.startswith(("sz.300", "sz.301"))
    gem_reform = gem & (date >= "2020-08-24")
    pct = np.where(is_st == 1, 0.05, 0.10)
    return np.where(star | gem_reform, 0.20, pct)


def _one_stock(bars: pd.DataFrame, fac: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    span = cal[(cal >= bars.index.min()) & (cal <= bars.index.max())]
    df = bars.reindex(span)
    df["code"] = bars["code"].iloc[0]
    df["tradestatus"] = (df.volume.fillna(0) > 0).astype(int)
    df["close"] = df.close.ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].where(df.tradestatus == 1, df.close)
    df[["volume", "amount"]] = df[["volume", "amount"]].where(df.tradestatus == 1, 0.0)
    df["outstanding_share"] = df.outstanding_share.ffill()
    f = fac.assign(date=pd.to_datetime(fac.date), f=pd.to_numeric(fac.hfq_factor, errors="coerce"))
    f = f.sort_values("date")[["date", "f"]]
    df = pd.merge_asof(df.rename_axis("date").reset_index(), f, on="date", direction="backward")
    df["f"] = df.f.fillna(1.0)
    df["preclose"] = df.close.shift() * df.f.shift() / df.f
    df["turn"] = df.turnover * 100
    return df


def load_prices() -> pd.DataFrame:
    cal = calendar()
    out = []
    for p in sorted((RAW / "sina").glob("*.parquet")):
        fp = RAW / "sina_factor" / p.name
        if not fp.exists():
            continue
        bars, fac = pd.read_parquet(p), pd.read_parquet(fp)
        if len(bars):
            out.append(_one_stock(bars, fac, cal))
    return pd.concat(out, ignore_index=True)


def infer_st(df: pd.DataFrame, window: int = 60, min_hits: int = 3) -> pd.Series:
    """ST stocks trade with a ±5% limit (main board). Flag a stock-day as ST when, over the
    past `window` days, its close/high/low touched a 5% limit price at least `min_hits` times
    and it never moved more than 5.5%. Parameters grid-searched against baostock's true isST
    flag on the 180-stock cross-validation set (precision 0.78 / recall 0.70)."""
    main_board = ~(df.code.str.startswith("sh.688") |
                   (df.code.str.startswith(("sz.300", "sz.301")) & (df.date >= "2020-08-24")))
    up5, dn5 = np.round(df.preclose * 1.05, 2), np.round(df.preclose * 0.95, 2)
    traded = df.tradestatus == 1
    touch = lambda px, lim: (px - lim).abs() < 0.006
    hit5 = traded & main_board & (touch(df.close, up5) | touch(df.close, dn5) |
                                  touch(df.high, up5) | touch(df.low, dn5))
    big = traded & (df.ret.abs() > 0.055)
    g = df.assign(hit5=hit5.astype(int), big=big.astype(int)).groupby("code")
    n_hit = g.hit5.transform(lambda s: s.rolling(window, min_periods=1).sum())
    n_big = g.big.transform(lambda s: s.rolling(window, min_periods=1).sum())
    return ((n_hit >= min_hits) & (n_big == 0) & main_board).astype(int)


# ------------------------------------------------------------------ fundamentals
REPORT_COLS = {"股票代码": "sym", "每股收益": "eps", "营业总收入-营业总收入": "rev",
               "营业总收入-同比增长": "rev_yoy", "净利润-净利润": "np", "净利润-同比增长": "np_yoy",
               "每股净资产": "bvps", "销售毛利率": "gpm", "最新公告日期": "ann_date"}


def load_reports() -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "reports").glob("*.parquet")):
        r = pd.read_parquet(p)
        r = r.rename(columns=REPORT_COLS)[list(REPORT_COLS.values()) + ["report_period"]]
        frames.append(r)
    r = pd.concat(frames, ignore_index=True)
    r["sym"] = r.sym.astype(str).str.zfill(6)
    r = r[r.sym.str[0].isin(["0", "3", "6"])]
    r["code"] = np.where(r.sym.str[0] == "6", "sh.", "sz.") + r.sym
    r["period"] = pd.to_datetime(r.report_period)
    r["ann_date"] = pd.to_datetime(r.ann_date, errors="coerce")
    for c in ["eps", "rev", "rev_yoy", "np", "np_yoy", "bvps", "gpm"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    # '最新公告日期' is the LATEST announcement touching this period (often next year's report
    # restating comparatives), not the first one. First-release date is therefore estimated as
    # min(latest date, statutory deadline): Q1 -> Apr 30, H1 -> Aug 31, Q3 -> Oct 31, FY -> Apr 30 (+1y).
    m = r.period.dt.month
    deadline = pd.to_datetime(np.select(
        [m == 3, m == 6, m == 9],
        [r.period.dt.year.astype(str) + "-04-30", r.period.dt.year.astype(str) + "-08-31",
         r.period.dt.year.astype(str) + "-10-31"],
        (r.period.dt.year + 1).astype(str) + "-04-30"))
    ok = r.ann_date.notna() & (r.ann_date > r.period)
    r["ann_date"] = r.ann_date.where(ok & (r.ann_date < deadline), deadline)
    return r.drop_duplicates(["code", "period"], keep="last")


def fundamentals(reports: pd.DataFrame) -> pd.DataFrame:
    """Per (code, period): TTM net profit / revenue per share and BVPS on the report's share base."""
    r = reports.sort_values(["code", "period"]).copy()
    r["year"], r["q"] = r.period.dt.year, r.period.dt.quarter
    key = r.set_index(["code", "year", "q"])
    def lookup(col, dy, q):
        idx = pd.MultiIndex.from_arrays([r.code, r.year + dy, q])
        return key[col].reindex(idx).values
    for c in ["np", "rev"]:
        annual_prev = lookup(c, -1, np.full(len(r), 4))
        same_q_prev = lookup(c, -1, r.q.values)
        r[f"{c}_ttm"] = np.where(r.q == 4, r[c], r[c] + annual_prev - same_q_prev)
    r["shares"] = (r.np / r.eps).where(r.eps.abs() > 0.005)          # report-period share base
    r["shares"] = r.groupby("code").shares.transform(lambda s: s.ffill())
    r["eps_ttm"] = r.np_ttm / r.shares
    r["sps_ttm"] = r.rev_ttm / r.shares
    r["roe"] = (r.np_ttm / (r.bvps * r.shares)).where(r.bvps > 0)
    return r[["code", "period", "ann_date", "eps_ttm", "sps_ttm", "bvps", "roe", "gpm", "np_yoy", "rev_yoy"]]


def attach_fundamentals(px: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join: on day t use the most recent *period* among reports announced
    strictly before t. Per-share items are moved to today's share base with the hfq factor:
    x_today = x_report * f(ann) / f(t), so value ratios = x_report * f(ann) / adj_close(t)."""
    fac_at = px[["code", "date", "f"]].sort_values("date")
    fund = fund.sort_values("ann_date")
    fund = pd.merge_asof(fund, fac_at.rename(columns={"date": "ann_date", "f": "f_ann"}),
                         on="ann_date", by="code", direction="backward")
    fund["f_ann"] = fund.f_ann.fillna(1.0)
    # drop restatements of older periods that arrive after a newer period was published
    fund = fund.sort_values(["code", "ann_date", "period"])
    fund = fund[fund.period >= fund.groupby("code").period.cummax()]
    px = px.sort_values("date")
    out = pd.merge_asof(px, fund.rename(columns={"ann_date": "avail"}).sort_values("avail"),
                        left_on="date", right_on="avail", by="code", direction="backward",
                        allow_exact_matches=False)
    stale = (out.date - out.period).dt.days > 500
    adj = out.close * out.f
    out["ep"] = (out.eps_ttm * out.f_ann / adj).where(~stale)
    out["bp"] = (out.bvps * out.f_ann / adj).where(~stale & (out.bvps > 0))
    out["sp"] = (out.sps_ttm * out.f_ann / adj).where(~stale & (out.sps_ttm > 0))
    for c in ["roe", "gpm", "np_yoy", "rev_yoy"]:
        out[c] = out[c].where(~stale)
    out["report_lag_days"] = (out.date - out.period).dt.days
    return out.drop(columns=["eps_ttm", "sps_ttm", "bvps", "f_ann", "avail"])


# ------------------------------------------------------------------ assemble
def build_long() -> pd.DataFrame:
    df = load_prices().sort_values(["code", "date"]).reset_index(drop=True)
    traded = (df.tradestatus == 1) & (df.volume > 0)
    df["ret"] = np.where(traded, df.close / df.preclose - 1, 0.0)
    df["ret"] = df.ret.where(df.preclose.notna(), 0.0)
    df["adj_close"] = df.close * df.f
    df["vwap"] = np.where(traded, df.amount / df.volume, np.nan)
    df["float_mcap"] = df.close * df.outstanding_share
    df["isST"] = infer_st(df)
    lim = limit_pct(df.code, df.date, df.isST)
    up_px, dn_px = np.round(df.preclose * (1 + lim), 2), np.round(df.preclose * (1 - lim), 2)
    df["limit_up"] = traded & (df.close >= up_px - 0.005)
    df["limit_down"] = traded & (df.close <= dn_px + 0.005)
    df["one_price_up"] = df.limit_up & (df.low >= up_px - 0.005)
    df["one_price_down"] = df.limit_down & (df.high <= dn_px + 0.005)
    df["traded"] = traded
    df = attach_fundamentals(df, fundamentals(load_reports()))
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def to_wide(long: pd.DataFrame, col: str) -> pd.DataFrame:
    return long.pivot(index="date", columns="code", values=col).sort_index()


def build_and_save():
    long = build_long()
    long.to_parquet(PROC / "daily_long.parquet", index=False)
    return long


def load_long() -> pd.DataFrame:
    return pd.read_parquet(PROC / "daily_long.parquet")


# ------------------------------------------------------------------ benchmarks
def reconstruct_benchmark(long: pd.DataFrame) -> pd.Series:
    """Daily total return of the point-in-time CSI800 members weighted by previous-day float
    market cap. This is exactly the benchmark the optimiser tracks (bench_weights uses the same
    float-mcap weights), so the strategy's alpha is measured against it. The official index uses
    free-float caps (strategic holdings excluded), which drifts 3-5%/yr from this one."""
    uni = pd.read_csv(RAW / "universe_monthly.csv")
    uni["month"] = pd.PeriodIndex(uni.month, freq="M")
    lm = long[["code", "date", "ret", "float_mcap"]].assign(month=long.date.dt.to_period("M"))
    lm = lm.merge(uni[["month", "code"]].drop_duplicates(), on=["month", "code"])
    lm = lm.sort_values(["code", "date"])
    lm["w"] = lm.groupby("code").float_mcap.shift()
    lm = lm.dropna(subset=["w"])
    lm["wr"] = lm.w * lm.ret
    g = lm.groupby("date")
    return (g.wr.sum() / g.w.sum()).rename("bench_rec")


def load_benchmarks() -> pd.DataFrame:
    """rec = reconstructed float-cap CSI800 (strategy benchmark, primary);
    tri = official CSI800 total-return index H00906 (reported alongside)."""
    from .config import CFG
    rec = pd.read_parquet(PROC / "bench_rec.parquet")["bench_rec"]
    tri = load_index(CFG["benchmark_tr"]).ret
    return pd.DataFrame({"rec": rec, "tri": tri.reindex(rec.index)})
