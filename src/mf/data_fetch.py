"""Market data acquisition from baostock (free, no token).

Raw (unadjusted) daily bars are stored per stock. Total returns are later
rebuilt from close / preclose, where preclose is the exchange ex-rights
reference price, so no separate adjustment-factor download is needed.
"""
from __future__ import annotations

import time
from multiprocessing import Pool

import baostock as bs
import pandas as pd
from tqdm import tqdm

from .config import CFG, RAW

DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,turn,"
                "tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def _rows(rs) -> pd.DataFrame:
    """baostock's get_data() uses DataFrame.append (removed in pandas 2)."""
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    out = []
    while rs.next():
        out.append(rs.get_row_data())
    return pd.DataFrame(out, columns=rs.fields)


def _login(retries: int = 6):
    """baostock occasionally refuses logins under load: retry with backoff."""
    for k in range(retries):
        try:
            lg = bs.login()
            if lg.error_code == "0":
                return
            msg = lg.error_msg
        except Exception as e:
            msg = str(e)
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"baostock login failed: {msg}")


def fetch_universe() -> pd.DataFrame:
    """Point-in-time CSI800 membership: HS300 + CSI500 snapshot each month."""
    _login()
    cal = _rows(bs.query_trade_dates(CFG["data"]["universe_start"], CFG["data"]["end"]))
    cal = cal[cal.is_trading_day == "1"].calendar_date
    cal.to_frame("date").to_csv(RAW / "trade_calendar.csv", index=False)
    month_first = pd.to_datetime(cal).groupby(pd.to_datetime(cal).dt.to_period("M")).min()
    recs = []
    for d in tqdm(month_first.dt.strftime("%Y-%m-%d"), desc="universe"):
        for name, fn in (("hs300", bs.query_hs300_stocks), ("zz500", bs.query_zz500_stocks)):
            df = _rows(fn(date=d))
            df["month"] = d[:7]
            df["index"] = name
            recs.append(df[["month", "code", "index"]])
    uni = pd.concat(recs, ignore_index=True)
    uni.to_csv(RAW / "universe_monthly.csv", index=False)

    basic = _rows(bs.query_stock_basic())
    basic.to_csv(RAW / "stock_basic.csv", index=False)
    ind = _rows(bs.query_stock_industry())
    ind.to_csv(RAW / "stock_industry.csv", index=False)
    bs.logout()
    return uni


def _fetch_one(code: str, fields: str, out_dir, retries: int = 3) -> str:
    path = out_dir / f"{code}.parquet"
    if path.exists():
        return "cached"
    for k in range(retries):
        try:
            df = _rows(bs.query_history_k_data_plus(
                code, fields, start_date=CFG["data"]["start"], end_date=CFG["data"]["end"],
                frequency="d", adjustflag="3"))
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.rename(path)                    # atomic: no half-written files if killed
            return "ok"
        except Exception as e:  # network hiccups: re-login and retry
            time.sleep(2 * (k + 1))
            try:
                bs.logout(); _login()
            except Exception:
                pass
            err = str(e)
    return f"fail: {err}"


def _worker(args):
    codes, fields, out_dir = args
    try:
        _login()
    except Exception as e:  # never let one worker kill the pool; the outer loop retries
        return {c: f"fail: {e}" for c in codes}
    res = {c: _fetch_one(c, fields, out_dir) for c in codes}
    try:
        bs.logout()
    except Exception:
        pass
    return res


def fetch_daily(codes: list[str], n_workers: int | None = None) -> dict:
    n_workers = n_workers or CFG["data"]["n_workers"]
    out_dir = RAW / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if not (out_dir / f"{c}.parquet").exists()]
    chunks = [todo[i::n_workers * 8] for i in range(n_workers * 8)]  # small chunks -> progress bar
    res = {}
    with Pool(n_workers) as pool:
        for r in tqdm(pool.imap_unordered(_worker, [(c, DAILY_FIELDS, out_dir) for c in chunks if c]),
                      total=len([c for c in chunks if c]), desc="daily bars"):
            res.update(r)
    return res


def fetch_indices():
    out_dir = RAW / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    _login()
    for code in CFG["indices"]:
        _fetch_one(code, INDEX_FIELDS, out_dir)
    bs.logout()
