"""Content manifest of data/raw (data/raw/manifest.json).

Per artifact group and per file: row count, min/max date (report period for reports, month for
the universe), sha256 of the file bytes, fetched_at (UTC), source; per group and overall a
content hash. Later pipeline steps should key their caches on ``content_hash(group)`` instead of
file mtimes, and ``check_coverage(manifest, end)`` tells whether the raw data covers the window.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

MANIFEST_NAME = "manifest.json"
VERSION = 1

# group -> (glob under raw_dir, date column or None, source)
GROUPS = {
    "calendar": ("trade_calendar.csv", "date", "baostock query_trade_dates"),
    "universe": ("universe_monthly.csv", "month", "baostock query_hs300_stocks/query_zz500_stocks"),
    "meta": ("stock_*.csv", None, "baostock query_stock_basic/query_stock_industry"),
    "index": ("index/[!H]*.parquet", "date", "baostock query_history_k_data_plus"),
    "tr_index": ("index/H*.parquet", "date", "csindex (akshare stock_zh_index_hist_csindex)"),
    "sina": ("sina/*.parquet", "date", "sina (akshare stock_zh_a_daily)"),
    "sina_factor": ("sina_factor/*.parquet", "date", "sina (akshare stock_zh_a_daily hfq-factor)"),
    "daily": ("daily/*.parquet", "date", "baostock query_history_k_data_plus (xval sample)"),
    "reports": ("reports/*.parquet", "report_period", "eastmoney (akshare stock_yjbb_em)"),
}
REQUIRED = ("calendar", "universe", "meta", "index", "tr_index", "sina", "sina_factor", "reports")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_raw() -> Path:
    from .config import RAW
    return RAW


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_lines(lines: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _norm(v, col) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if col == "month":
        return str(v)[:7]
    if col == "report_period":
        return str(v)
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def file_stats(path: Path, col: Optional[str]) -> dict:
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str, usecols=[col] if col else None)
    else:
        if col == "report_period":                     # one file per period: name is the period
            import pyarrow.parquet as pq
            return {"rows": pq.ParquetFile(path).metadata.num_rows, "min": path.stem, "max": path.stem}
        df = pd.read_parquet(path, columns=[col] if col else None)
    out = {"rows": int(len(df)), "min": None, "max": None}
    if col and len(df):
        s = df[col].dropna()
        if len(s):
            if col == "month":
                out["min"], out["max"] = s.min()[:7], s.max()[:7]
            else:
                d = pd.to_datetime(s)
                out["min"], out["max"] = _norm(d.min(), col), _norm(d.max(), col)
    return out


def build_manifest(raw_dir: Optional[Path] = None, refresh: Optional[dict] = None,
                   end: Optional[str] = None, previous: Optional[dict] = None) -> dict:
    """Scan raw_dir and describe every artifact. `refresh` is the log returned by
    data_fetch_ak.refresh_all (fetch times, revisions, inactive codes, ...). fetched_at comes from
    the refresh log, else from the previous manifest when the file is unchanged, else the mtime."""
    raw = Path(raw_dir or _default_raw())
    if previous is None:
        try:
            previous = load_manifest(raw / MANIFEST_NAME)
        except (FileNotFoundError, ValueError):
            previous = {}
    refresh = refresh or {}
    fetched = refresh.get("fetched", {})
    groups = {}
    for g, (pattern, col, source) in GROUPS.items():
        files = {}
        prev_files = previous.get("groups", {}).get(g, {}).get("files", {})
        for p in sorted(raw.glob(pattern)):
            rel = p.relative_to(raw).as_posix()
            st = file_stats(p, col)
            sha = sha256_file(p)
            when = None
            for fg in fetched.values():
                when = when or fg.get(rel)
            if when is None and prev_files.get(rel, {}).get("sha256") == sha:
                when = prev_files[rel].get("fetched_at")
            if when is None:
                when = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            files[rel] = {**st, "sha256": sha, "fetched_at": when, "source": source}
        maxes = [f["max"] for f in files.values() if f["max"]]
        mins = [f["min"] for f in files.values() if f["min"]]
        groups[g] = {
            "source": source, "n_files": len(files), "rows": sum(f["rows"] for f in files.values()),
            "min": min(mins) if mins else None, "max": max(maxes) if maxes else None,
            "earliest_max": min(maxes) if maxes else None,
            "content_hash": _hash_lines([f"{k}:{v['sha256']}" for k, v in files.items()]),
            "files": files}
    # inactive codes (suspended / delisted, checked by the refresh) carry over while unchanged
    inactive = {g: dict(v) for g, v in (refresh.get("inactive") or {}).items()}
    for g, codes in (previous.get("inactive") or {}).items():
        for code, info in codes.items():
            rel = f"{g}/{code}.parquet"
            if code in inactive.get(g, {}):
                continue
            prev_sha = previous.get("groups", {}).get(g, {}).get("files", {}).get(rel, {}).get("sha256")
            if rel in groups.get(g, {}).get("files", {}) and groups[g]["files"][rel]["sha256"] == prev_sha:
                inactive.setdefault(g, {})[code] = info
    m = {
        "version": VERSION, "built_at": utcnow(), "raw_dir": str(raw),
        "end": end or refresh.get("end"),
        "expected_last_trading_day": refresh.get("expected_last_trading_day"),
        "benchmark": refresh.get("benchmark"), "indices": refresh.get("indices"),
        "tr_indices": refresh.get("tr_indices"),
        "complete": False,
        "content_hash": _hash_lines([f"{g}:{v['content_hash']}" for g, v in groups.items()]),
        "groups": groups, "inactive": inactive,
        "refresh": {k: v for k, v in refresh.items() if k not in ("fetched", "inactive")},
    }
    return m


def write_manifest(manifest: dict, raw_dir: Optional[Path] = None) -> Path:
    raw = Path(raw_dir or manifest.get("raw_dir") or _default_raw())
    path = raw / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, path)
    return path


def load_manifest(path: Optional[Path] = None) -> dict:
    path = Path(path or _default_raw() / MANIFEST_NAME)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/01_fetch_data.py first")
    return json.loads(path.read_text())


def invalidate_manifest(raw_dir: Optional[Path] = None, reason: str = "") -> None:
    """Mark an existing manifest incomplete (called before a refresh starts writing files)."""
    path = Path(raw_dir or _default_raw()) / MANIFEST_NAME
    if not path.exists():
        return
    try:
        m = json.loads(path.read_text())
    except ValueError:
        path.unlink()
        return
    m["complete"] = False
    m["invalidated"] = reason
    write_manifest(m, path.parent)


def content_hash(group: str = "all", manifest: Optional[dict] = None) -> str:
    """Content hash of one artifact group ('sina', 'reports', ...) or of everything ('all')."""
    m = manifest if manifest is not None else load_manifest()
    if group == "all":
        return m["content_hash"]
    if group not in m["groups"]:
        raise KeyError(f"unknown manifest group {group!r}; have {sorted(m['groups'])}")
    return m["groups"][group]["content_hash"]


def _report_deadline(period: str) -> str:
    y, md = int(period[:4]), period[4:]
    return {"0331": f"{y}-04-30", "0630": f"{y}-08-31", "0930": f"{y}-10-31",
            "1231": f"{y + 1}-04-30"}[md]


def _required_period(end: str) -> Optional[str]:
    ps = [d.strftime("%Y%m%d") for d in pd.date_range("2014-03-31", end, freq="QE")]
    ok = [p for p in ps if _report_deadline(p) <= end[:10]]
    return ok[-1] if ok else None


def check_coverage(manifest: dict, end: str, calendar: Optional[List[str]] = None,
                   groups: Optional[List[str]] = None) -> List[str]:
    """Problems that keep the raw data from covering the window ending at `end` ([] = OK).

    The expected last trading day is the last day <= end of `calendar` (default: the
    manifest's trade_calendar.csv). Without a calendar only month-level checks are possible."""
    end = str(end)[:10]
    probs: List[str] = []
    G = manifest.get("groups", {})
    want = set(groups or GROUPS)
    raw = Path(manifest.get("raw_dir") or _default_raw())
    if manifest.get("refresh", {}).get("error"):
        probs.append(f"last refresh failed: {manifest['refresh']['error'][:300]}")
    if calendar is None and (raw / "trade_calendar.csv").exists():
        calendar = pd.read_csv(raw / "trade_calendar.csv", dtype=str).date.tolist()
    exp = None
    if calendar:
        past = sorted(str(d)[:10] for d in calendar if str(d)[:10] <= end)
        exp = past[-1] if past else None
        if not past:
            probs.append(f"calendar: no trading day on or before {end}")
        elif max(str(d)[:10] for d in calendar) < end and exp[:7] < end[:7]:
            probs.append(f"calendar: trading calendar ends {exp}, does not reach {end[:7]}")
    end_month = exp[:7] if exp else end[:7]

    for g in REQUIRED:
        if g in want and not G.get(g, {}).get("n_files"):
            probs.append(f"{g}: no files")

    def dated(g, label):
        for rel, f in G.get(g, {}).get("files", {}).items():
            mx = f.get("max")
            if not f.get("rows") or not mx:
                probs.append(f"{label} {rel}: empty")
            elif exp and mx < exp:
                probs.append(f"{label} {rel}: max date {mx} < last trading day of window {exp}")
            elif mx[:7] < end_month:
                probs.append(f"{label} {rel}: ends {mx}, does not reach {end_month}")

    if "index" in want:
        dated("index", "index")
        bench = manifest.get("benchmark")
        if bench and f"index/{bench}.parquet" not in G.get("index", {}).get("files", {}):
            probs.append(f"index: benchmark {bench} missing")
    if "tr_index" in want:
        dated("tr_index", "total-return index")
    if "universe" in want and G.get("universe", {}).get("max"):
        if G["universe"]["max"] < end_month:
            probs.append(f"universe: last month {G['universe']['max']} < {end_month}")
    if "calendar" in want and G.get("calendar", {}).get("max") and exp and G["calendar"]["max"] < exp:
        probs.append(f"calendar: ends {G['calendar']['max']} < {exp}")

    if "reports" in want:
        req = _required_period(end)
        f = G.get("reports", {}).get("files", {}).get(f"reports/{req}.parquet") if req else None
        if req and (not f or not f.get("rows")):
            probs.append(f"reports: latest required period {req} (deadline {_report_deadline(req)}) missing")

    delisted = {}
    if (raw / "stock_basic.csv").exists():
        b = pd.read_csv(raw / "stock_basic.csv", dtype=str)
        b = b[b.outDate.notna() & (b.outDate.str.len() > 0)]
        delisted = dict(zip(b.code, b.outDate.str[:10]))
    inactive = manifest.get("inactive", {})
    for g in ("sina", "daily"):
        if g not in want or g not in G:
            continue
        stale = []
        for rel, f in G[g]["files"].items():
            code = Path(rel).stem
            if code in delisted and (not exp or delisted[code] <= exp):
                continue
            if code in inactive.get(g, {}):
                continue
            if not f.get("max") or (exp and f["max"] < exp) or f["max"][:7] < end_month:
                stale.append(f"{code}({f.get('max')})")
        if stale:
            probs.append(f"{g}: {len(stale)} files of non-delisted, not-known-inactive stocks end "
                         f"before {exp or end_month}: {', '.join(stale[:10])}")
    if "sina" in want and "sina_factor" in G:
        miss = [Path(r).name for r in G.get("sina", {}).get("files", {})
                if f"sina_factor/{Path(r).name}" not in G["sina_factor"]["files"]]
        if miss:
            probs.append(f"sina_factor: {len(miss)} stocks without hfq factor file: {miss[:10]}")
    return probs
