"""Market + fundamental data via AkShare (Sina daily bars, Eastmoney quarterly reports).

Why two sources
---------------
* Sina `stock_zh_a_daily` : unadjusted OHLCV, amount, float shares, turnover, plus the
  back-adjustment factor (`hfq-factor`) -> total returns and split-consistent per-share data.
* Eastmoney `stock_yjbb_em`: one call returns the whole market's quarterly report
  (EPS, BVPS, ROE, growth, gross margin, operating CF per share) **with the announcement
  date**, which lets us build point-in-time valuation and quality factors.

Both providers rate-limit aggressively, so requests are throttled (one global limiter across
threads) and retried with exponential back-off. All network calls go through
:class:`AkshareSource`, so tests can inject a fake. The refresh is incremental and fail-fast
(see ``data_fetch``): stale files are re-fetched, revisions of the cached history are checked,
and missing / incomplete data raises instead of being silently kept.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import CFG, RAW
from .data_fetch import (BaostockSource, CoverageError, FetchError, PerCodeSpec, RefreshContext,
                         _dates, atomic_parquet, check_revision, compare_overlap, gap_days,
                         refresh_calendar, refresh_indices, refresh_per_code, refresh_universe,
                         refresh_xval_daily)

SINA_DIR = RAW / "sina"
FACTOR_DIR = RAW / "sina_factor"
REPORT_DIR = RAW / "reports"
for d in (SINA_DIR, FACTOR_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

XVAL_CODES = Path(__file__).resolve().parent / "xval_codes.txt"

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
    atomic_parquet(df, Path(path))


def _sina_symbol(code: str) -> str:       # 'sh.600000' -> 'sh600000'
    return code.replace(".", "")


TR_INDICES = {"H00906": "中证800全收益", "H00300": "沪深300全收益", "H00905": "中证500全收益"}


# ------------------------------------------------------------------ network source
class AkshareSource:
    """All AkShare network calls (Sina, CSIndex, Eastmoney), throttled + exponential back-off.

    Empty / "no data" answers (e.g. delisted stocks) are returned as empty frames, not retried."""
    name = "akshare"

    def __init__(self, retries: int = 4, base_sleep: float = 15.0, report_pause: float = 5.0):
        self.retries, self.base_sleep, self.report_pause = retries, base_sleep, report_pause

    def _call(self, what: str, fn, *a, base: Optional[float] = None, **k) -> pd.DataFrame:
        err = ""
        for attempt in range(self.retries):
            try:
                _throttle()
                return fn(*a, **k)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                if "No value to decode" in err or "KeyError" in err:   # no data (e.g. delisted)
                    return pd.DataFrame()
                time.sleep((base or self.base_sleep) * 2 ** attempt)   # 15s, 30s, 60s, 120s
        raise FetchError(f"{what} failed after {self.retries} tries: {err}")

    def sina_bars(self, code: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak
        return self._call(f"sina bars {code}", ak.stock_zh_a_daily, symbol=_sina_symbol(code),
                          start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="")

    def sina_factor(self, code: str) -> pd.DataFrame:
        import akshare as ak
        return self._call(f"sina hfq-factor {code}", ak.stock_zh_a_daily, symbol=_sina_symbol(code),
                          adjust="hfq-factor")

    def tr_index(self, code: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak
        return self._call(f"CSIndex {code} (is www.csindex.com.cn reachable?)",
                          ak.stock_zh_index_hist_csindex, symbol=code,
                          start_date=start.replace("-", ""), end_date=end.replace("-", ""), base=5.0)

    def report(self, period: str) -> pd.DataFrame:
        import akshare as ak
        try:
            return self._call(f"eastmoney report {period}", ak.stock_yjbb_em, date=period, base=30.0)
        finally:
            time.sleep(self.report_pause)


# ------------------------------------------------------------------ Sina bars + hfq factor
def refresh_sina(ctx: RefreshContext, src, codes: List[str], active: Optional[set] = None,
                 n_workers: int = 3) -> dict:
    """Sina returns the whole history, so a stale code is re-fetched entirely (bars + factor)."""
    def fetch(code):
        bars = src.sina_bars(code, ctx.start, ctx.end)
        fac = src.sina_factor(code) if bars is not None and len(bars) else pd.DataFrame()
        if bars is not None and len(bars):
            bars = bars.copy(); bars["code"] = code
        if fac is not None and len(fac):
            fac = fac.copy(); fac["code"] = code
        return {"bars": bars, "factor": fac}

    spec = PerCodeSpec(group="sina", subdirs={"bars": "sina", "factor": "sina_factor"}, fetch=fetch,
                       compare={"bars": ("date", ("open", "high", "low", "close")),
                                "factor": ("date", ("hfq_factor",))},
                       strict_gaps=False, n_workers=n_workers)
    return refresh_per_code(ctx, spec, codes, active=active)


# ------------------------------------------------------------------ total-return indices
def _csindex_to_bars(d: pd.DataFrame, code: str, cal: List[str]) -> pd.DataFrame:
    d = d.rename(columns={"日期": "date", "收盘": "close", "涨跌幅": "pctChg", "成交量": "volume",
                          "成交金额": "amount"})
    d["date"] = pd.to_datetime(d.date)
    d = d[d.date.isin(pd.to_datetime(pd.Series(cal)))].sort_values("date")  # CSIndex also lists holidays
    d["preclose"] = d.close.shift().fillna(d.close / (1 + d.pctChg / 100))
    for c in ["open", "high", "low"]:
        d[c] = d.close
    d["code"] = code
    return d[["date", "code", "open", "high", "low", "close", "preclose", "volume",
              "amount", "pctChg"]].astype({"date": str}).reset_index(drop=True)


def refresh_tr_indices(ctx: RefreshContext, src, codes=None) -> None:
    """Total-return (dividends reinvested) indices from CSIndex. Portfolio returns include
    dividends, so excess returns must be measured against a total-return benchmark; the price
    index understates the benchmark by the dividend yield (~2%/yr for CSI800)."""
    out_dir = ctx.raw_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for code in codes or TR_INDICES:
        path = out_dir / f"{code}.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        if old is not None and len(old) and _dates(old.date).max() >= ctx.expected_last:
            continue
        raw = src.tr_index(code, ctx.start, ctx.end)
        if raw is None or not len(raw):
            raise FetchError(f"total-return index {code}: empty download")
        new = _csindex_to_bars(raw, code, ctx.ref_calendar)
        last = new.date.max() if len(new) else "none"
        if not len(new) or last < ctx.expected_last:
            raise CoverageError(f"total-return index {code} ends {last}, expected {ctx.expected_last}")
        gaps = gap_days(new.date, ctx.window_days())
        if gaps:
            raise CoverageError(f"total-return index {code}: {len(gaps)} benchmark trading days "
                                f"missing inside the window ({gaps[:5]})")
        err = check_revision(ctx, "tr_index", path.name,
                             compare_overlap(old, new, "date", ["close"]))
        if err:
            errors.append(err)
            continue
        atomic_parquet(new, path)
        ctx.mark_fetched("tr_index", f"index/{path.name}")
    if errors:
        raise FetchError("\n".join(errors))


# ------------------------------------------------------------------ quarterly reports
def report_dates(start="2014-03-31", end=None) -> list[str]:
    end = end or CFG["data"]["end"]
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="QE")]


def report_deadline(period: str) -> str:
    """Statutory disclosure deadline: Q1 Apr 30, H1 Aug 31, Q3 Oct 31, annual Apr 30 next year."""
    y, md = int(period[:4]), period[4:]
    return {"0331": f"{y}-04-30", "0630": f"{y}-08-31", "0930": f"{y}-10-31",
            "1231": f"{y + 1}-04-30"}[md]


def required_report_period(end: str) -> Optional[str]:
    """Latest report period whose statutory deadline is on or before `end`."""
    ok = [p for p in report_dates("2014-03-31", end) if report_deadline(p) <= end[:10]]
    return ok[-1] if ok else None


def _fetched_at(path: Path, prev_manifest: Optional[dict]) -> str:
    """Earliest of file mtime and manifest fetched_at (conservative: re-fetch when in doubt)."""
    t = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
    try:
        m = prev_manifest["groups"]["reports"]["files"][f"reports/{path.name}"]["fetched_at"][:10]
        t = min(t, m)
    except (KeyError, TypeError):
        pass
    return t


def refresh_reports(ctx: RefreshContext, src, prev_manifest: Optional[dict] = None,
                    start: str = "2014-03-31") -> dict:
    """Whole-market quarterly reports (Eastmoney), one file per report period.

    A period is re-fetched when missing, or when its cached snapshot was taken before the
    statutory deadline (so it may lack late filers). The latest period that must be available by
    the end date has to be present and non-empty."""
    out_dir = ctx.raw_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    required = required_report_period(ctx.end_window)
    res = {}
    for p in report_dates(start, ctx.end):
        path = out_dir / f"{p}.parquet"
        if path.exists() and _fetched_at(path, prev_manifest) > report_deadline(p):
            res[p] = "cached"
            continue
        df = src.report(p)
        if df is None or not len(df):
            if report_deadline(p) <= ctx.end_window:
                raise FetchError(f"reports {p}: empty download although the statutory deadline "
                                 f"{report_deadline(p)} has passed")
            ctx.warn(f"reports {p}: empty (deadline {report_deadline(p)} not reached yet)")
            res[p] = "empty"
            continue
        df = df.copy()
        df["report_period"] = p
        df = df.astype({"股票代码": str})
        if path.exists():                     # restatements / late filers are expected: log only
            old = pd.read_parquet(path)
            num = [c for c in df.columns if c in old.columns and pd.api.types.is_numeric_dtype(df[c])
                   and c != "序号"]
            rev = compare_overlap(old, df, "股票代码", num)
            if rev.get("revised_rows"):
                ctx.log["revisions"].setdefault("reports", {})[path.name] = {
                    **rev, "accepted": True, "note": "report restatements / late filers (not gated)"}
        atomic_parquet(df, path)
        ctx.mark_fetched("reports", f"reports/{path.name}")
        res[p] = "ok"
    if required:
        rp = out_dir / f"{required}.parquet"
        if not rp.exists() or not len(pd.read_parquet(rp, columns=["股票代码"])):
            raise CoverageError(f"reports: latest required period {required} (deadline "
                                f"{report_deadline(required)} <= {ctx.end_window}) is missing")
    return res


# ------------------------------------------------------------------ orchestration
def refresh_all(cfg: Optional[dict] = None, raw_dir: Optional[Path] = None, end: Optional[str] = None,
                bs_source=None, ak_source=None, what: str = "all", accept_revisions: bool = False,
                xval_codes: Optional[List[str]] = None, as_of: Optional[str] = None,
                prev_manifest: Optional[dict] = None, n_workers: int = 3, **ctx_kw) -> dict:
    """Incremental refresh of every raw artifact up to `end` (default: config data.end).

    Order: trading calendar -> price indices (benchmark = reference calendar) -> universe /
    stock_basic / industry -> total-return indices -> baostock cross-validation sample ->
    Sina bars + hfq factors || Eastmoney reports. Raises FetchError / RevisionError /
    CoverageError on the first problem; returns the refresh log (for the manifest)."""
    cfg = cfg or CFG
    raw_dir = Path(raw_dir or RAW)
    kw = dict(accept_revisions=accept_revisions, **ctx_kw)
    if as_of:
        kw["as_of"] = as_of
    ctx = RefreshContext.from_config(cfg, raw_dir, end=end, **kw)
    bs_source = bs_source if bs_source is not None else BaostockSource()
    ak_source = ak_source if ak_source is not None else AkshareSource()
    from . import manifest as mf
    mf.invalidate_manifest(raw_dir, f"refresh to {ctx.end} started {mf.utcnow()}")

    # The reference end of the *previous complete* cache decides which stocks are expected to trade
    # and where the gap-check window starts. Persist it until the refresh succeeds, so a re-run
    # after a failure (benchmark already updated) keeps the same strictness.
    state = raw_dir / ".refresh_state.json"
    bench = raw_dir / "index" / f"{ctx.benchmark}.parquet"
    cur = _dates(pd.read_parquet(bench, columns=["date"]).date).max() if bench.exists() else None
    prev_ref = json.loads(state.read_text()).get("old_ref_last") if state.exists() else None
    ctx.old_ref_last = min([x for x in (prev_ref, cur) if x], default=None)
    state.write_text(json.dumps({"old_ref_last": ctx.old_ref_last, "end": ctx.end,
                                 "started": mf.utcnow()}))

    enter = getattr(bs_source, "__enter__", None)
    if enter:
        enter()
    try:
        refresh_calendar(ctx, bs_source)
        refresh_indices(ctx, bs_source)
        uni = refresh_universe(ctx, bs_source)
        refresh_tr_indices(ctx, ak_source)
        if what == "all":
            # independent baostock bars for a fixed sample: used only by the data audit to
            # cross-validate Sina returns and the inferred ST flag (step 2)
            codes = xval_codes if xval_codes is not None else XVAL_CODES.read_text().split()
            refresh_xval_daily(ctx, bs_source, codes)
    finally:
        ex = getattr(bs_source, "__exit__", None)
        if ex:
            ex(None, None, None)

    codes = sorted(uni.code.unique())
    active = set(uni[uni.month == uni.month.max()].code)
    # Codes the source has never served (recorded as 'nodata' by an earlier download and with no cached
    # file) are a known data limitation, not a fresh failure: they are still re-tried, but an empty answer
    # for them is logged instead of aborting the refresh.
    fails = raw_dir / "fetch_failures_sina.csv"
    known_unavailable = set()
    if fails.exists():
        f = pd.read_csv(fails, header=None, names=["code", "msg"], dtype=str)
        known_unavailable = {c for c, m in zip(f.code, f.msg.fillna(""))
                             if "nodata" in m and not (raw_dir / "sina" / f"{c}.parquet").exists()}
    if known_unavailable & active:
        print(f"sina: {len(known_unavailable & active)} active codes never served by the source "
              f"(known limitation, not fatal): {sorted(known_unavailable & active)}", flush=True)
    ctx.log["sina_known_unavailable"] = sorted(known_unavailable)
    active = active - known_unavailable
    print(f"universe: {len(codes)} unique stocks over {uni.month.nunique()} months", flush=True)
    with ThreadPoolExecutor(1) as pool:                 # reports run in parallel with the bars
        rep = pool.submit(refresh_reports, ctx, ak_source, prev_manifest) \
            if what in ("reports", "all") else None
        try:
            if what in ("bars", "all"):
                refresh_sina(ctx, ak_source, codes, active=active, n_workers=n_workers)
        finally:
            if rep is not None:
                rep.result()                                # re-raises a reports failure
    ctx.log.update(end=ctx.end, as_of=ctx.as_of, expected_last_trading_day=ctx.expected_last,
                   previous_reference_end=ctx.old_ref_last, what=what,
                   accept_revisions=bool(accept_revisions), benchmark=ctx.benchmark,
                   indices=list(dict.fromkeys([ctx.benchmark] + ctx.indices)),
                   tr_indices=list(TR_INDICES))
    state.unlink(missing_ok=True)
    fails = ctx.log["failures"].get("sina")
    if fails:
        pd.Series(fails).to_csv(raw_dir / "fetch_failures_sina.csv")
    return ctx.log
