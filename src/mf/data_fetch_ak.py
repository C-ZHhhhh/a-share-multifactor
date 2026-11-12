"""Market + fundamental data via AkShare (Sina daily bars, Eastmoney quarterly reports).

Why two sources
---------------
* Sina `stock_zh_a_daily` : unadjusted OHLCV, amount, float shares, turnover, plus the
  back-adjustment factor (`hfq-factor`) -> total returns and split-consistent per-share data.
* Eastmoney `stock_yjbb_em`: one call returns the whole market's quarterly report
  (EPS, BVPS, ROE, growth, gross margin, operating CF per share) **with the announcement
  date**, which lets us build point-in-time valuation and quality factors.

Both providers rate-limit aggressively, so requests are throttled and retried with
exponential back-off. All writes are atomic (tmp file + rename) and every step is
resumable: re-running only downloads what is missing.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
from tqdm import tqdm

from .config import CFG, RAW

SINA_DIR = RAW / "sina"
FACTOR_DIR = RAW / "sina_factor"
REPORT_DIR = RAW / "reports"
for d in (SINA_DIR, FACTOR_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.8          # seconds between any two Sina requests (global, across threads)


def _throttle():
    with _lock:
        wait = _last_call[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _atomic_write(df: pd.DataFrame, path):
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def _sina_symbol(code: str) -> str:       # 'sh.600000' -> 'sh600000'
    return code.replace(".", "")


def fetch_one_sina(code: str, retries: int = 4) -> str:
    bar_path, fac_path = SINA_DIR / f"{code}.parquet", FACTOR_DIR / f"{code}.parquet"
    if bar_path.exists() and fac_path.exists():
        return "cached"
    sym = _sina_symbol(code)
    start, end = CFG["data"]["start"].replace("-", ""), CFG["data"]["end"].replace("-", "")
    err = ""
    for k in range(retries):
        try:
            if not bar_path.exists():
                _throttle()
                df = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="")
                df["code"] = code
                _atomic_write(df, bar_path)
            if not fac_path.exists():
                _throttle()
                f = ak.stock_zh_a_daily(symbol=sym, adjust="hfq-factor")
                f["code"] = code
                _atomic_write(f, fac_path)
            return "ok"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            if "No value to decode" in err or "KeyError" in err:   # no data (e.g. delisted)
                return f"nodata: {err}"
            time.sleep(15 * 2 ** k)                                 # 15s, 30s, 60s, 120s
    return f"fail: {err}"


def fetch_daily_sina(codes: list[str], n_workers: int = 3) -> dict:
    todo = [c for c in codes if not ((SINA_DIR / f"{c}.parquet").exists()
                                     and (FACTOR_DIR / f"{c}.parquet").exists())]
    res = {}
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(fetch_one_sina, c): c for c in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="sina daily"):
            res[futs[f]] = f.result()
    return res


def report_dates(start="2014-03-31", end=None) -> list[str]:
    end = end or CFG["data"]["end"]
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="QE")]


def fetch_reports(retries: int = 4) -> dict:
    """Whole-market quarterly reports (Eastmoney), one file per report period."""
    res = {}
    for d in tqdm(report_dates(), desc="quarterly reports"):
        path = REPORT_DIR / f"{d}.parquet"
        if path.exists():
            res[d] = "cached"
            continue
        for k in range(retries):
            try:
                df = ak.stock_yjbb_em(date=d)
                df["report_period"] = d
                _atomic_write(df.astype({"股票代码": str}), path)
                res[d] = "ok"
                break
            except Exception as e:
                res[d] = f"fail: {str(e)[:120]}"
                time.sleep(30 * 2 ** k)
        time.sleep(5)
    return res


TR_INDICES = {"H00906": "中证800全收益", "H00300": "沪深300全收益", "H00905": "中证500全收益"}


def fetch_tr_indices(start: str = "20150101", end: str | None = None) -> None:
    """Total-return (dividends reinvested) indices from CSIndex. Portfolio returns include
    dividends, so excess returns must be measured against a total-return benchmark; the price
    index understates the benchmark by the dividend yield (~2%/yr for CSI800)."""
    end = end or CFG["data"]["end"].replace("-", "")
    out_dir = RAW / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in TR_INDICES:
        path = out_dir / f"{code}.parquet"
        if path.exists():
            continue
        _throttle()
        d = ak.stock_zh_index_hist_csindex(symbol=code, start_date=start, end_date=end)
        d = d.rename(columns={"日期": "date", "收盘": "close", "涨跌幅": "pctChg", "成交量": "volume",
                              "成交金额": "amount"})
        d["date"] = pd.to_datetime(d.date)
        cal = pd.to_datetime(pd.read_parquet(out_dir / f"{CFG['benchmark']}.parquet").date)
        d = d[d.date.isin(cal)].sort_values("date")              # CSIndex also lists holidays
        d["preclose"] = d.close.shift().fillna(d.close / (1 + d.pctChg / 100))
        for c in ["open", "high", "low"]:
            d[c] = d.close
        d["code"] = code
        _atomic_write(d[["date", "code", "open", "high", "low", "close", "preclose", "volume",
                         "amount", "pctChg"]].astype({"date": str}), path)
