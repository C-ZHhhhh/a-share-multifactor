"""Market data acquisition from baostock (free, no token) + the incremental refresh engine.

Raw (unadjusted) daily bars are stored per stock. Total returns are later
rebuilt from close / preclose, where preclose is the exchange ex-rights
reference price, so no separate adjustment-factor download is needed.

Incremental, fail-fast refresh
------------------------------
Every cached artifact is compared with what the configured ``data.end`` requires:

* price-like files (indices, per-stock bars) are re-fetched when their last date is before the
  expected last trading day of the window (the last day of the exchange calendar <= end), unless
  the stock is known to be delisted;
* the re-fetched series is compared with the cached one on the overlapping history; revisions are
  logged and, above a threshold (>1% of overlapping rows or any revision older than one year),
  raise :class:`RevisionError` unless ``accept_revisions`` is set;
* empty / failed fetches for codes expected to trade, gaps vs the reference calendar (the
  benchmark index), and indices that do not reach the end raise immediately;
* per-stock files are first written to ``raw/.staging`` and moved into place only once the
  whole group passed its checks, so a failed refresh never leaves a half-updated cache
  (staged files are reused by the next run, after being re-checked).

The network is reached only through *source* objects (:class:`BaostockSource` here,
``AkshareSource`` in ``data_fetch_ak``), so tests can inject fakes.
"""
from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,turn,"
                "tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


# ------------------------------------------------------------------ errors
class FetchError(RuntimeError):
    """A download failed or returned unusable data for something that is expected to exist."""


class RevisionError(FetchError):
    """A re-fetched series rewrites too much of the cached history."""


class CoverageError(FetchError):
    """An artifact does not cover the configured window (stale end, date gaps, missing period)."""


# ------------------------------------------------------------------ baostock source
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
    import baostock as bs
    msg = ""
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


class BaostockSource:
    """All baostock network calls. One session, retried with re-login + linear backoff.

    baostock keeps a module-global socket, so use one instance from a single thread."""
    name = "baostock"

    def __init__(self, retries: int = 3):
        self.retries = retries
        self._open = False

    def __enter__(self):
        _login()
        self._open = True
        return self

    def __exit__(self, *exc):
        if self._open:
            import baostock as bs
            try:
                bs.logout()
            except Exception:
                pass
            self._open = False

    def _call(self, fn: Callable, *a, **k) -> pd.DataFrame:
        import baostock as bs
        err = ""
        for attempt in range(self.retries):
            try:
                return _rows(fn(*a, **k))
            except Exception as e:  # network hiccups: re-login and retry
                err = str(e)
                time.sleep(2 * (attempt + 1))
                try:
                    bs.logout(); _login()
                except Exception:
                    pass
        raise FetchError(f"baostock {getattr(fn, '__name__', fn)} failed after {self.retries} tries: {err}")

    def trade_dates(self, start: str, end: str) -> List[str]:
        import baostock as bs
        cal = self._call(bs.query_trade_dates, start, end)
        return sorted(cal[cal.is_trading_day == "1"].calendar_date.tolist())

    def _k(self, code, fields, start, end):
        import baostock as bs
        return self._call(bs.query_history_k_data_plus, code, fields, start_date=start, end_date=end,
                          frequency="d", adjustflag="3")

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._k(code, INDEX_FIELDS, start, end)

    def stock_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._k(code, DAILY_FIELDS, start, end)

    def constituents(self, day: str) -> pd.DataFrame:
        """HS300 + CSI500 members on `day` -> columns code, index."""
        import baostock as bs
        out = []
        for name, fn in (("hs300", bs.query_hs300_stocks), ("zz500", bs.query_zz500_stocks)):
            df = self._call(fn, date=day)
            out.append(pd.DataFrame({"code": df.get("code", pd.Series(dtype=str)), "index": name}))
        return pd.concat(out, ignore_index=True)

    def stock_basic(self) -> pd.DataFrame:
        import baostock as bs
        return self._call(bs.query_stock_basic)

    def stock_industry(self) -> pd.DataFrame:
        import baostock as bs
        return self._call(bs.query_stock_industry)


# ------------------------------------------------------------------ refresh context
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _d(x) -> str:
    """Normalise a date-like value to 'YYYY-MM-DD'."""
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def atomic_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)                     # atomic: no half-written files if killed


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


@dataclass
class RefreshContext:
    raw_dir: Path
    start: str
    end: str
    universe_start: str
    benchmark: str
    indices: List[str]
    as_of: str = field(default_factory=lambda: date.today().isoformat())
    accept_revisions: bool = False
    revision_share: float = 0.01        # > 1% of overlapping rows revised -> raise
    revision_age_days: int = 365        # any revised row older than this (vs expected end) -> raise
    gap_share: float = 0.05             # a day missing for > 5% of active stocks = outage -> raise
    fresh_gap_days: int = 60            # gap-check window when there is no previous cache
    # filled while refreshing
    calendar: List[str] = field(default_factory=list)       # exchange trading days <= end
    expected_last: Optional[str] = None                      # last trading day the data must reach
    old_ref_last: Optional[str] = None                       # benchmark's last date before refresh
    ref_calendar: List[str] = field(default_factory=list)   # benchmark dates (reference calendar)
    delisted: Dict[str, str] = field(default_factory=dict)  # code -> outDate
    log: dict = field(default_factory=lambda: {
        "fetched": {}, "revisions": {}, "inactive": {}, "failures": {}, "gaps": {}, "warnings": []})

    @classmethod
    def from_config(cls, cfg: dict, raw_dir: Path, **kw) -> "RefreshContext":
        d = cfg["data"]
        return cls(raw_dir=Path(raw_dir), start=d["start"], end=kw.pop("end", None) or d["end"],
                   universe_start=d["universe_start"], benchmark=cfg["benchmark"],
                   indices=list(cfg["indices"]), **kw)

    # ---- helpers
    @property
    def end_window(self) -> str:
        """Last calendar date data could exist for (end capped by today)."""
        return min(self.end, self.as_of)

    def window_days(self, cal: Optional[List[str]] = None) -> List[str]:
        """Reference trading days inside the requested (new) window: (old benchmark end,
        expected_last], or the last `fresh_gap_days` days when there was no cache."""
        cal = cal or self.ref_calendar or self.calendar
        if self.old_ref_last:
            lo = self.old_ref_last
        else:
            past = [d for d in cal if d <= (self.expected_last or self.end)]
            lo = past[-self.fresh_gap_days - 1] if len(past) > self.fresh_gap_days else ""
        return [d for d in cal if lo < d <= (self.expected_last or self.end)]

    def mark_fetched(self, group: str, rel: str):
        self.log["fetched"].setdefault(group, {})[rel] = utcnow()

    def warn(self, msg: str):
        self.log["warnings"].append(msg)
        print("WARNING:", msg, flush=True)

    def is_delisted(self, code: str, by: Optional[str] = None) -> bool:
        out = self.delisted.get(code)
        return bool(out) and (by is None or out <= by)


def load_delisted(raw_dir: Path) -> Dict[str, str]:
    p = Path(raw_dir) / "stock_basic.csv"
    if not p.exists():
        return {}
    b = pd.read_csv(p, dtype=str)
    b = b[b.outDate.notna() & (b.outDate.str.len() > 0)]
    return dict(zip(b.code, b.outDate.str[:10]))


# ------------------------------------------------------------------ revision detection
def compare_overlap(old: pd.DataFrame, new: pd.DataFrame, key: str, cols: List[str],
                    old_until: Optional[str] = None, rtol: float = 1e-6, atol: float = 1e-8) -> dict:
    """Compare the cached history `old` with a re-fetched `new` series.

    revised rows = rows whose values in `cols` changed + rows dropped from the history
    + rows that appeared inside the old history (key <= old_until). Keys are compared as strings."""
    if old is None or not len(old):
        return {"overlap_rows": 0, "revised_rows": 0}
    o = old.assign(_k=old[key].astype(str)).drop_duplicates("_k", keep="last").set_index("_k")
    n = new.assign(_k=new[key].astype(str)).drop_duplicates("_k", keep="last").set_index("_k")
    common = o.index.intersection(n.index)
    changed = pd.Series(False, index=common)
    for c in cols:
        if c in o.columns and c in n.columns:
            a = pd.to_numeric(o.loc[common, c], errors="coerce").to_numpy(float)
            b = pd.to_numeric(n.loc[common, c], errors="coerce").to_numpy(float)
            changed |= ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    dropped = o.index.difference(n.index)
    lo, hi = o.index.min(), old_until or o.index.max()
    added = [k for k in n.index.difference(o.index) if lo <= k <= hi]
    keys = sorted(list(changed[changed].index) + list(dropped) + added)
    out = {"overlap_rows": int(len(o)), "revised_rows": len(keys),
           "changed": int(changed.sum()), "dropped": int(len(dropped)), "added": len(added)}
    if keys:
        out.update(first=keys[0], last=keys[-1])
    return out


def check_revision(ctx: RefreshContext, group: str, name: str, rev: dict) -> Optional[str]:
    """Record a revision; return an error message when it exceeds the threshold."""
    if not rev.get("revised_rows"):
        return None
    ctx_rev = dict(rev)
    share = rev["revised_rows"] / max(rev["overlap_rows"], 1)
    old_cut = (pd.Timestamp(ctx.expected_last or ctx.end) - timedelta(days=ctx.revision_age_days)).strftime("%Y-%m-%d")
    first = str(rev.get("first", ""))[:10]
    too_many = share > ctx.revision_share
    too_old = bool(first) and first < old_cut
    ctx_rev["share"] = round(share, 6)
    ctx_rev["exceeds_threshold"] = bool(too_many or too_old)
    ctx_rev["accepted"] = bool(ctx.accept_revisions) if ctx_rev["exceeds_threshold"] else True
    ctx.log["revisions"].setdefault(group, {})[name] = ctx_rev
    if (too_many or too_old) and not ctx.accept_revisions:
        why = []
        if too_many:
            why.append(f"{share:.2%} of {rev['overlap_rows']} overlapping rows > {ctx.revision_share:.0%}")
        if too_old:
            why.append(f"revision as old as {first} (< {old_cut})")
        return (f"{group}/{name}: re-fetched history differs from cache in {rev['revised_rows']} rows "
                f"[{rev.get('first')} .. {rev.get('last')}] ({'; '.join(why)}). "
                f"Re-run with --accept-revisions to overwrite the cache.")
    return None


def gap_days(dates: pd.Series, window: List[str]) -> List[str]:
    """Reference days inside `window` and inside the series' own span that the series lacks."""
    if not len(dates) or not window:
        return []
    have = set(dates.astype(str).str[:10])
    lo, hi = min(have), max(have)
    return [d for d in window if lo <= d <= hi and d not in have]


# ------------------------------------------------------------------ calendar / indices / universe
def refresh_calendar(ctx: RefreshContext, src) -> List[str]:
    """Exchange trading calendar [universe_start, end] -> trade_calendar.csv. Always re-fetched
    (one call); previously cached days must be unchanged."""
    path = ctx.raw_dir / "trade_calendar.csv"
    days = sorted(_d(x) for x in src.trade_dates(ctx.universe_start, ctx.end))
    days = [d for d in days if d <= ctx.end]
    if not days:
        raise FetchError(f"empty trading calendar for {ctx.universe_start}..{ctx.end}")
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        hi = min(old.date.max(), days[-1])
        o = set(old.date[old.date <= hi]); n = set(d for d in days if d <= hi)
        diff = sorted(o ^ n)
        if diff:
            err = check_revision(ctx, "calendar", "trade_calendar.csv",
                                 {"overlap_rows": len(o), "revised_rows": len(diff),
                                  "first": diff[0], "last": diff[-1]})
            if err:
                raise RevisionError(err)
    atomic_csv(pd.DataFrame({"date": days}), path)
    ctx.mark_fetched("calendar", "trade_calendar.csv")
    ctx.calendar = days
    past = [d for d in days if d <= ctx.end_window]
    if not past:
        raise CoverageError(f"no trading day on or before {ctx.end_window}")
    ctx.expected_last = past[-1]
    if ctx.end > ctx.as_of:
        ctx.warn(f"data.end {ctx.end} is after today ({ctx.as_of}); data can reach {ctx.expected_last} at most")
    return days


def _read_old(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if path.exists() else None


def refresh_indices(ctx: RefreshContext, src) -> None:
    """baostock price indices (benchmark first: it is the reference calendar for everything else)."""
    out_dir = ctx.raw_dir / "index"
    bench = out_dir / f"{ctx.benchmark}.parquet"
    if ctx.old_ref_last is None and bench.exists():
        ctx.old_ref_last = _dates(pd.read_parquet(bench, columns=["date"]).date).max()
    errors = []
    codes = [ctx.benchmark] + [c for c in ctx.indices if c != ctx.benchmark]
    for code in codes:
        path = out_dir / f"{code}.parquet"
        old = _read_old(path)
        if old is not None and len(old) and _dates(old.date).max() >= ctx.expected_last:
            if code == ctx.benchmark:
                ctx.ref_calendar = sorted(_dates(old.date))
            continue
        new = src.index_daily(code, ctx.start, ctx.end)
        if new is None or not len(new):
            raise FetchError(f"index {code}: empty download for {ctx.start}..{ctx.end}")
        new = new.sort_values("date").reset_index(drop=True)
        last = _dates(new.date).max()
        if last < ctx.expected_last:
            raise CoverageError(f"index {code} ends {last}, expected {ctx.expected_last} "
                                f"(last trading day <= {ctx.end_window}); data not published yet?")
        # benchmark is checked against the exchange calendar, the others against the benchmark
        win = ctx.window_days(ctx.calendar) if code == ctx.benchmark else ctx.window_days()
        gaps = gap_days(_dates(new.date), win)
        if gaps:
            raise CoverageError(f"index {code}: {len(gaps)} trading days missing inside the window "
                                f"({gaps[:5]})")
        if code == ctx.benchmark:
            ctx.ref_calendar = sorted(_dates(new.date))
        err = check_revision(ctx, "index", path.name,
                             compare_overlap(old, new, "date", ["close", "preclose"]))
        if err:
            errors.append(err)
            continue
        atomic_parquet(new, path)
        ctx.mark_fetched("index", f"index/{path.name}")
    if errors:
        raise RevisionError("\n".join(errors))


def month_first_days(days: List[str]) -> Dict[str, str]:
    s = pd.Series(sorted(days))
    return s.groupby(s.str[:7]).min().to_dict()


def refresh_universe(ctx: RefreshContext, src) -> pd.DataFrame:
    """Point-in-time CSI800 membership (HS300 + CSI500 snapshot on each month's first trading day).

    Months already in universe_monthly.csv are never rewritten; only missing months are appended.
    The last cached month is re-fetched as a consistency check: a difference is recorded (and only
    replaces the cached month with accept_revisions)."""
    path = ctx.raw_dir / "universe_monthly.csv"
    old = pd.read_csv(path, dtype=str) if path.exists() else None
    firsts = {m: d for m, d in month_first_days(ctx.calendar).items()
              if m >= ctx.universe_start[:7] and d <= ctx.expected_last}
    have = set(old.month) if old is not None else set()
    todo = [m for m in sorted(firsts) if m not in have]
    check = [max(have)] if have and max(have) in firsts else []
    new_parts = []
    for m in todo + check:
        df = src.constituents(firsts[m])
        if df is None or not len(df):
            raise FetchError(f"universe: empty constituent list for {m} ({firsts[m]})")
        df = pd.DataFrame({"month": m, "code": df["code"].astype(str).values, "index": df["index"].values})
        cnt = df["index"].value_counts().to_dict()
        if cnt.get("hs300", 0) != 300 or cnt.get("zz500", 0) != 500:
            ctx.warn(f"universe {m}: constituent counts {cnt} (expected hs300=300, zz500=500)")
        if m in check:
            o = old[old.month == m]
            a, b = set(zip(o.code, o["index"])), set(zip(df.code, df["index"]))
            if a != b:
                ctx.log["revisions"].setdefault("universe", {})[m] = {
                    "overlap_rows": len(a), "revised_rows": len(a ^ b), "added": len(b - a),
                    "removed": len(a - b), "first": m, "last": m, "accepted": bool(ctx.accept_revisions)}
                ctx.warn(f"universe {m}: re-fetched snapshot differs from cache ({len(b - a)} added, "
                         f"{len(a - b)} removed); " + ("replaced (--accept-revisions)" if ctx.accept_revisions
                                                         else "kept the cached snapshot"))
                if ctx.accept_revisions:
                    old = old[old.month != m]
                    new_parts.append(df)
            continue
        new_parts.append(df)
    if new_parts or old is None:
        uni = pd.concat(([old] if old is not None else []) + new_parts, ignore_index=True)
        uni = uni.sort_values("month", kind="stable").reset_index(drop=True)
        atomic_csv(uni, path)
        ctx.mark_fetched("universe", "universe_monthly.csv")
    else:
        uni = old
    for name, fn in (("stock_basic.csv", src.stock_basic), ("stock_industry.csv", src.stock_industry)):
        p = ctx.raw_dir / name
        if todo or not p.exists():                  # current snapshots: refreshed with new months
            df = fn()
            if df is None or not len(df):
                raise FetchError(f"{name}: empty download")
            atomic_csv(df, p)
            ctx.mark_fetched("meta", name)
    last_month = uni.month.max()
    if last_month < ctx.expected_last[:7]:
        raise CoverageError(f"universe ends {last_month}, expected {ctx.expected_last[:7]}")
    ctx.delisted = load_delisted(ctx.raw_dir)
    return uni


# ------------------------------------------------------------------ per-stock engine
@dataclass
class PerCodeSpec:
    """How to refresh one per-stock group (one or more files per code)."""
    group: str                                   # manifest / log name of the primary file
    subdirs: Dict[str, str]                      # part -> sub-directory under raw_dir ("bars" first)
    fetch: Callable[[str], Dict[str, pd.DataFrame]]
    compare: Dict[str, tuple]                    # part -> (key, value cols)
    strict_gaps: bool                            # any gap raises (source has suspended-day rows)
    n_workers: int = 1

    @property
    def primary(self) -> str:
        return next(iter(self.subdirs))


def refresh_per_code(ctx: RefreshContext, spec: PerCodeSpec, codes: List[str],
                     active: Optional[set] = None) -> dict:
    """Re-fetch stale per-stock files (whole history), check, stage, then commit as a group.

    A code is *stale* when a file is missing or its last date < expected_last, unless the stock
    was already delisted when the old cache was built. It is *expected to trade* when it is not
    delisted by expected_last and its cache reached the old reference end (or, without a cache,
    it is in `active` = the latest universe month)."""
    raw = ctx.raw_dir
    stage = raw / ".staging"
    dirs = {p: raw / s for p, s in spec.subdirs.items()}
    sdirs = {p: stage / s for p, s in spec.subdirs.items()}
    for d in list(dirs.values()) + list(sdirs.values()):
        d.mkdir(parents=True, exist_ok=True)
    window = ctx.window_days()
    active = active or set()
    prim = spec.primary
    res = {}

    def plan(code):
        paths = {p: dirs[p] / f"{code}.parquet" for p in dirs}
        old_last = None
        if paths[prim].exists():
            d = pd.read_parquet(paths[prim], columns=["date"]).date
            old_last = _dates(d).max() if len(d) else None
        complete = all(p.exists() for p in paths.values())
        out = ctx.delisted.get(code)
        if complete and old_last and old_last >= ctx.expected_last:
            return None
        if out and complete and out <= (ctx.old_ref_last or old_last or ""):
            return None                              # delisted before the cache was built
        if out and not complete and out < ctx.start:
            return None
        expected = (not ctx.is_delisted(code, ctx.expected_last)) and (
            (old_last is not None and ctx.old_ref_last is not None and old_last >= ctx.old_ref_last)
            or (old_last is None and code in active))
        return {"old_last": old_last, "expected": expected}

    def work(code, info):
        staged = {p: sdirs[p] / f"{code}.parquet" for p in sdirs}
        new = None
        if all(p.exists() for p in staged.values()):          # resume a previous, aborted refresh
            new = {p: pd.read_parquet(f) for p, f in staged.items()}
            if not len(new[prim]) or _dates(new[prim].date).max() < ctx.expected_last:
                new = None
        if new is None:
            try:
                new = spec.fetch(code)
            except Exception as e:
                return {"status": "fail", "msg": f"{type(e).__name__}: {str(e)[:200]}"}
            if any(v is None or not len(v) for v in new.values()):
                return {"status": "nodata", "msg": "empty download"}
        r = {"status": "ok", "revisions": {}, "gaps": []}
        for p, df in new.items():
            old = _read_old(dirs[p] / f"{code}.parquet")
            key, cols = spec.compare[p]
            o = old.assign(**{key: _dates(old[key])}) if old is not None and len(old) else None
            n = df.assign(**{key: _dates(df[key])})
            r["revisions"][p] = compare_overlap(o, n, key, list(cols), old_until=info["old_last"])
        r["first"], r["last"] = _dates(new[prim].date).min(), _dates(new[prim].date).max()
        r["gaps"] = gap_days(_dates(new[prim].date), window)
        for p, df in new.items():
            atomic_parquet(df, staged[p])
        return r

    def drop_staged(code):
        for d in sdirs.values():
            (d / f"{code}.parquet").unlink(missing_ok=True)

    plans = {c: plan(c) for c in codes}
    todo = {c: i for c, i in plans.items() if i is not None}
    print(f"{spec.group}: {len(todo)}/{len(codes)} codes stale or missing -> re-fetch", flush=True)
    fatal = []
    ex = ThreadPoolExecutor(max(1, spec.n_workers))
    try:
        futs = {ex.submit(work, c, i): c for c, i in todo.items()}
        for f in as_completed(futs):
            code = futs[f]
            info, r = todo[code], f.result()
            res[code] = r
            if r["status"] != "ok":
                ctx.log["failures"].setdefault(spec.group, {})[code] = f"{r['status']}: {r['msg']}"
                if info["expected"]:
                    fatal.append(f"{spec.group}/{code}: {r['status']} ({r['msg']}) although the stock "
                                 f"traded on {ctx.old_ref_last or 'the last month'} and is not delisted")
            else:
                for p, rev in r["revisions"].items():
                    err = check_revision(ctx, spec.subdirs[p], f"{code}.parquet", rev)
                    if err:
                        fatal.append(err)
                        drop_staged(code)
                if info["expected"] and r["gaps"] and spec.strict_gaps:
                    fatal.append(f"{spec.group}/{code}: {len(r['gaps'])} reference trading days missing "
                                 f"inside the window ({r['gaps'][:5]})")
                    drop_staged(code)
                if r["last"] < ctx.expected_last:
                    why = ("delisted " + ctx.delisted[code]) if code in ctx.delisted else \
                        "ends early (suspended?)"
                    ctx.log["inactive"].setdefault(spec.group, {})[code] = {"last": r["last"], "why": why}
            if fatal:                                   # fail fast: stop scheduling more downloads
                for g in futs:
                    g.cancel()
                break
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    if not fatal and not spec.strict_gaps:
        # Sina omits suspended days, so a single stock missing a day is normal; the same day missing
        # for a large share of the actively traded stocks is a data outage.
        act = [c for c, r in res.items() if r["status"] == "ok" and todo[c]["expected"]]
        for d in window:
            span = [c for c in act if res[c]["first"] <= d <= res[c]["last"]]
            miss = [c for c in act if d in res[c]["gaps"]]
            if miss and len(miss) / max(len(span), 1) > ctx.gap_share:
                fatal.append(f"{spec.group}: {d} missing for {len(miss)}/{len(span)} actively traded "
                             f"stocks (> {ctx.gap_share:.0%}) e.g. {miss[:5]}")
                for c in miss:
                    drop_staged(c)
        for c in act:
            if res[c]["gaps"]:
                ctx.log["gaps"].setdefault(spec.group, {})[c] = res[c]["gaps"][:20]
    if fatal:
        raise (RevisionError if all("accept-revisions" in m for m in fatal) else FetchError)(
            f"{spec.group}: refresh aborted, cache left unchanged:\n  " + "\n  ".join(fatal[:20]))
    for code, r in res.items():                         # commit: staged -> cache
        if r["status"] != "ok":
            continue
        for p in dirs:
            os.replace(sdirs[p] / f"{code}.parquet", dirs[p] / f"{code}.parquet")
            ctx.mark_fetched(spec.subdirs[p], f"{spec.subdirs[p]}/{code}.parquet")
    for d in sdirs.values():
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    if stage.exists() and not any(stage.iterdir()):
        shutil.rmtree(stage, ignore_errors=True)
    return res


def refresh_xval_daily(ctx: RefreshContext, src, codes: List[str]) -> dict:
    """baostock bars (incl. true ST flag / tradestatus) for the fixed cross-validation sample.
    baostock lists suspended days, so any gap vs the reference calendar is an error."""
    spec = PerCodeSpec(
        group="daily", subdirs={"bars": "daily"},
        fetch=lambda c: {"bars": src.stock_daily(c, ctx.start, ctx.end)},
        compare={"bars": ("date", ("close", "preclose", "open", "high", "low"))},
        strict_gaps=True, n_workers=1)                 # baostock: one session, one thread
    return refresh_per_code(ctx, spec, codes)
