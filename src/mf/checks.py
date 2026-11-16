"""Data-integrity checks that must hold before results are computed (fail fast) plus soft diagnostics.

Hard failures (raise): duplicate (code, date) rows, trading days missing from the benchmark calendar
inside the panel window, non-positive float market caps on traded days.
Soft diagnostics (reported): one-day returns beyond the widest legal limit (possible adjustment errors),
suspended stock-days, members without a previous-day cap (benchmark coverage).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import PROC


def validate_long(long: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    dup = int(long.duplicated(["code", "date"]).sum())
    if dup:
        raise ValueError(f"{dup} duplicate (code, date) rows in the daily panel")
    days = pd.DatetimeIndex(sorted(long.date.unique()))
    cal = calendar[(calendar >= days[0]) & (calendar <= days[-1])]
    missing = cal.difference(days)
    if len(missing):
        raise ValueError(f"{len(missing)} benchmark trading days have no stock data, e.g. {missing[0].date()}")
    traded = long[long.traded.astype(bool)]
    bad_cap = int((traded.float_mcap <= 0).sum())
    if bad_cap:
        raise ValueError(f"{bad_cap} traded stock-days with non-positive float market cap")
    jumps = traded[traded.ret.abs() > 0.21]                     # beyond the 20% STAR/ChiNext limit
    out = {"rows": int(len(long)), "stocks": int(long.code.nunique()), "first_day": str(days[0].date()),
           "last_day": str(days[-1].date()), "duplicate_rows": dup, "missing_calendar_days": 0,
           "nonpositive_caps": bad_cap, "suspended_share": float(1 - long.traded.astype(bool).mean()),
           "returns_beyond_21pct": int(len(jumps)),
           "returns_beyond_21pct_examples": jumps.nlargest(5, "ret")[["code", "date", "ret"]].astype(str).values.tolist()}
    cov = PROC / "bench_coverage.parquet"
    if cov.exists():
        c = pd.read_parquet(cov)
        out.update({"bench_coverage_mean": float(c.coverage.mean()), "bench_coverage_min": float(c.coverage.min())})
    json.dump(out, open(PROC / "data_checks.json", "w"), ensure_ascii=False, indent=2)
    return out
