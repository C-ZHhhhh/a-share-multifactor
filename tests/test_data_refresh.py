"""Incremental, fail-fast data refresh + manifest (issue F01). No network: fake sources only."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from mf import manifest as mf
from mf.data_fetch import DAILY_FIELDS, INDEX_FIELDS, CoverageError, FetchError, RevisionError
from mf.data_fetch_ak import refresh_all

INDICES = ["sh.000300", "sh.000905", "sh.000906"]
CFG = {"data": {"start": "2024-01-02", "end": "2026-10-31", "universe_start": "2024-01-02"},
       "benchmark": "sh.000906", "indices": INDICES}
HOLIDAYS = {"2024-01-01", "2025-01-01", "2026-01-01"} | {f"2026-10-0{d}" for d in range(1, 8)}
CAL = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", "2026-12-31")
       if d.strftime("%Y-%m-%d") not in HOLIDAYS]
DELISTED = {"sz.000002": "2026-06-30"}
LISTED = {"sz.000003": "2026-07-01"}
XVAL = ["sh.600000", "sz.000001"]


def _px(code, d):
    return round(10 + int(code[-1]) + (int(d.replace("-", "")) % 97) / 10, 2)


def _days(code, start, end, until):
    lo = max(start, LISTED.get(code, "0000"))
    hi = min(end, until, DELISTED.get(code, "9999"))
    return [d for d in CAL if lo <= d <= hi]


class FakeBS:
    """Stands in for BaostockSource: calendar, indices, constituents, basic, xval daily bars."""

    def __init__(self, until, drop=None, members=None):
        self.until, self.drop, self.members = until, drop or {}, members or {}
        self.calls = []

    def trade_dates(self, start, end):
        return [d for d in CAL if start <= d <= end]          # baostock also lists future days

    def _bars(self, code, start, end, fields):
        days = [d for d in _days(code, start, end, self.until) if d not in self.drop.get(code, ())]
        rows = []
        for d in days:
            c = _px(code, d)
            rec = dict(date=d, code=code, open=c, high=c, low=c, close=c, preclose=c, volume=1000,
                       amount=c * 1000, pctChg=0.0, turn=1.0, tradestatus=1, peTTM=10, pbMRQ=1,
                       psTTM=1, isST=0)
            rows.append({k: str(rec[k]) for k in fields.split(",")})
        return pd.DataFrame(rows, columns=fields.split(","))

    def index_daily(self, code, start, end):
        self.calls.append(("index", code))
        return self._bars(code, start, end, INDEX_FIELDS)

    def stock_daily(self, code, start, end):
        self.calls.append(("daily", code))
        return self._bars(code, start, end, DAILY_FIELDS)

    def constituents(self, day):
        self.calls.append(("universe", day))
        m = day[:7]
        if m in self.members:
            codes = self.members[m]
        else:
            codes = ["sh.600000", "sh.600001", "sz.000001"]
            codes += ["sz.000002"] if m <= "2026-06" else ["sz.000003"]
        return pd.DataFrame({"code": codes, "index": ["hs300", "hs300"] + ["zz500"] * (len(codes) - 2)})

    def stock_basic(self):
        codes = ["sh.600000", "sh.600001", "sz.000001", "sz.000002", "sz.000003"]
        return pd.DataFrame({"code": codes, "code_name": codes, "ipoDate": "2000-01-01",
                             "outDate": [DELISTED.get(c, "") for c in codes], "type": "1",
                             "status": ["0" if c in DELISTED else "1" for c in codes]})

    def stock_industry(self):
        return pd.DataFrame({"updateDate": "2026-11-01", "code": ["sh.600000"], "code_name": ["x"],
                             "industry": ["J66"], "industryClassification": ["csrc"]})


class FakeAK:
    """Stands in for AkshareSource: Sina bars + hfq factor, CSIndex TR indices, Eastmoney reports."""

    def __init__(self, until, fail=None, revise=None, drop=None):
        self.until, self.fail, self.revise, self.drop = until, fail or {}, revise or {}, drop or {}
        self.calls = []

    def sina_bars(self, code, start, end):
        self.calls.append(("sina", code))
        mode = self.fail.get(code)
        if mode == "raise":
            raise RuntimeError("HTTP 456 rate limited")
        if mode == "empty":
            return pd.DataFrame()
        days = [d for d in _days(code, start, end, self.until) if d not in self.drop.get(code, ())]
        c = [_px(code, d) for d in days]
        df = pd.DataFrame({"date": days, "open": c, "high": c, "low": c, "close": c,
                           "volume": 1000.0, "amount": [x * 1000 for x in c],
                           "outstanding_share": 1e9, "turnover": 0.001})
        if code in self.revise:
            df = self.revise[code](df)
        return df

    def sina_factor(self, code):
        self.calls.append(("factor", code))
        return pd.DataFrame({"date": pd.to_datetime(["2025-07-16", "1900-01-01"]),
                             "hfq_factor": ["1.2000000000000000", "1.0000000000000000"]})

    def tr_index(self, code, start, end):
        self.calls.append(("tr", code))
        days = [d for d in CAL if start <= d <= min(end, self.until)]
        days += ["2026-10-01"] if start <= "2026-10-01" <= min(end, self.until) else []  # holiday row
        c = [_px("H0090" + code[-1], d) * 100 for d in days]
        return pd.DataFrame({"日期": days, "收盘": c, "涨跌幅": 0.1, "成交量": 1.0, "成交金额": 1.0})

    def report(self, period):
        self.calls.append(("report", period))
        return pd.DataFrame({"序号": [1, 2], "股票代码": ["600000", "000001"], "股票简称": ["a", "b"],
                             "每股收益": [1.0, 2.0], "最新公告日期": ["2026-01-01", "2026-01-02"]})


def _run(raw, end, bs, ak, as_of, **kw):
    log = refresh_all(cfg=CFG, raw_dir=raw, end=end, bs_source=bs, ak_source=ak, as_of=as_of,
                      xval_codes=XVAL, n_workers=2, **kw)
    m = mf.build_manifest(raw, refresh=log, end=end)
    m["complete"] = not mf.check_coverage(m, end)
    mf.write_manifest(m, raw)
    return log, m


@pytest.fixture
def cache(tmp_path):
    """A complete fake cache ending 2026-10-30 (built by the refresh itself, end = 2026-10-31)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    log, m = _run(raw, "2026-10-31", FakeBS("2026-10-30"), FakeAK("2026-10-30"), "2026-10-31")
    assert m["complete"], mf.check_coverage(m, "2026-10-31")
    assert m["groups"]["sina"]["max"] == "2026-10-30"
    # the Q3 snapshot was taken before its statutory deadline (Oct 31) -> must be re-fetched later
    t = time.mktime((2026, 10, 20, 12, 0, 0, 0, 0, -1))
    os.utime(raw / "reports" / "20260930.parquet", (t, t))
    return raw


def _sha(raw, rel):
    return mf.sha256_file(raw / rel)


# ---------------------------------------------------------------- (a) incremental refresh
def test_refresh_refetches_stale_artifacts_and_keeps_universe_history(cache):
    raw = cache
    uni_before = pd.read_csv(raw / "universe_monthly.csv", dtype=str)
    delisted_sha = _sha(raw, "sina/sz.000002.parquet")
    bs, ak = FakeBS("2026-11-30"), FakeAK("2026-11-30")
    log, m = _run(raw, "2026-11-30", bs, ak, "2026-12-15")

    last_nov = max(d for d in CAL if d <= "2026-11-30")
    assert log["expected_last_trading_day"] == last_nov == "2026-11-30"
    # every stale artifact was re-fetched
    assert {c for k, c in bs.calls if k == "index"} == set(INDICES)
    assert {c for k, c in ak.calls if k == "tr"} == {"H00906", "H00300", "H00905"}
    assert {c for k, c in ak.calls if k == "sina"} == {"sh.600000", "sh.600001", "sz.000001", "sz.000003"}
    assert {c for k, c in bs.calls if k == "daily"} == set(XVAL)
    assert ("universe", "2026-11-02") in bs.calls                 # new month appended
    assert ("report", "20260930") in ak.calls                     # pre-deadline snapshot re-fetched
    # delisted stock is not re-downloaded and stays untouched
    assert ("sina", "sz.000002") not in ak.calls
    assert _sha(raw, "sina/sz.000002.parquet") == delisted_sha
    # manifest max dates reach the last fake trading day of November
    for g in ("index", "tr_index", "daily"):
        assert m["groups"][g]["earliest_max"] == last_nov, g
    for code in ("sh.600000", "sh.600001", "sz.000001", "sz.000003"):
        assert m["groups"]["sina"]["files"][f"sina/{code}.parquet"]["max"] == last_nov
    assert m["groups"]["universe"]["max"] == "2026-11"
    assert mf.check_coverage(m, "2026-11-30") == []
    assert m["complete"]
    # historical universe months untouched, only November appended
    uni = pd.read_csv(raw / "universe_monthly.csv", dtype=str)
    pd.testing.assert_frame_equal(uni[uni.month < "2026-11"].reset_index(drop=True), uni_before)
    assert set(uni.month) - set(uni_before.month) == {"2026-11"}
    # manifest content hashes / helpers
    m2 = mf.load_manifest(raw / "manifest.json")
    assert mf.content_hash("sina", m2) == m["groups"]["sina"]["content_hash"]
    assert mf.content_hash("all", m2) == m["content_hash"]
    f = m2["groups"]["sina"]["files"]["sina/sh.600000.parquet"]
    assert f["sha256"] == _sha(raw, "sina/sh.600000.parquet") and f["fetched_at"].endswith("Z")
    assert f["rows"] > 0 and f["source"].startswith("sina")
    # nothing left in staging; a second run is a no-op for prices
    assert not (raw / ".staging").exists()
    ak2 = FakeAK("2026-11-30")
    _run(raw, "2026-11-30", FakeBS("2026-11-30"), ak2, "2026-12-15")
    assert not [c for c in ak2.calls if c[0] == "sina"]


def test_changed_historical_universe_month_is_kept_and_logged(cache):
    raw = cache
    before = pd.read_csv(raw / "universe_monthly.csv", dtype=str)
    bs = FakeBS("2026-11-30", members={"2026-10": ["sh.600000", "sh.600001", "sz.000009"]})
    log, _ = _run(raw, "2026-11-30", bs, FakeAK("2026-11-30"), "2026-12-15")
    rev = log["revisions"]["universe"]["2026-10"]
    assert rev["revised_rows"] > 0 and not rev["accepted"]
    uni = pd.read_csv(raw / "universe_monthly.csv", dtype=str)
    pd.testing.assert_frame_equal(uni[uni.month <= "2026-10"].reset_index(drop=True), before)


# ---------------------------------------------------------------- (b) failed fetch
@pytest.mark.parametrize("mode", ["empty", "raise"])
def test_failed_fetch_for_active_code_raises_and_manifest_not_complete(cache, mode):
    raw = cache
    old_sha = {c: _sha(raw, f"sina/{c}.parquet") for c in ("sh.600000", "sh.600001")}
    with pytest.raises(FetchError, match="sh.600001"):
        _run(raw, "2026-11-30", FakeBS("2026-11-30"),
             FakeAK("2026-11-30", fail={"sh.600001": mode}), "2026-12-15")
    m = mf.load_manifest(raw / "manifest.json")
    assert m["complete"] is False                      # invalidated before writing anything
    # the sina group was not committed: cached files unchanged
    for c, s in old_sha.items():
        assert _sha(raw, f"sina/{c}.parquet") == s
    # indices were already updated by the failed run; a re-run must stay just as strict
    with pytest.raises(FetchError, match="sh.600001"):
        _run(raw, "2026-11-30", FakeBS("2026-11-30"),
             FakeAK("2026-11-30", fail={"sh.600001": mode}), "2026-12-15")
    # once the source is fixed, the refresh completes (staged files from the failed run reused)
    _, m = _run(raw, "2026-11-30", FakeBS("2026-11-30"), FakeAK("2026-11-30"), "2026-12-15")
    assert m["complete"] and m["groups"]["sina"]["files"]["sina/sh.600001.parquet"]["max"] == "2026-11-30"
    assert not (raw / ".refresh_state.json").exists()


# ---------------------------------------------------------------- (c) revisions
def _big_revision(df):
    df = df.copy()
    old = df.date < "2025-06-01"
    df.loc[old, "close"] = df.loc[old, "close"] * 1.05
    return df


def test_large_revision_raises_without_accept_flag(cache):
    raw = cache
    sha = _sha(raw, "sina/sh.600000.parquet")
    with pytest.raises(RevisionError, match="accept-revisions"):
        _run(raw, "2026-11-30", FakeBS("2026-11-30"),
             FakeAK("2026-11-30", revise={"sh.600000": _big_revision}), "2026-12-15")
    assert _sha(raw, "sina/sh.600000.parquet") == sha
    log, m = _run(raw, "2026-11-30", FakeBS("2026-11-30"),
                  FakeAK("2026-11-30", revise={"sh.600000": _big_revision}), "2026-12-15",
                  accept_revisions=True)
    rev = m["refresh"]["revisions"]["sina"]["sh.600000.parquet"]
    assert rev["revised_rows"] > 100 and rev["first"] == "2024-01-02" and rev["last"] < "2025-06-01"
    assert rev["exceeds_threshold"] and rev["accepted"]
    assert m["complete"]


def test_small_recent_revision_is_logged_not_fatal(cache):
    def tweak(df):
        df = df.copy()
        df.loc[df.date == "2026-10-30", "close"] += 0.01
        return df
    log, m = _run(cache, "2026-11-30", FakeBS("2026-11-30"),
                  FakeAK("2026-11-30", revise={"sh.600001": tweak}), "2026-12-15")
    rev = log["revisions"]["sina"]["sh.600001.parquet"]
    assert rev["revised_rows"] == 1 and rev["first"] == rev["last"] == "2026-10-30"
    assert not rev["exceeds_threshold"]


# ---------------------------------------------------------------- (d) gaps
def test_date_gap_in_window_raises(cache):
    with pytest.raises(FetchError, match="2026-11-10"):
        _run(cache, "2026-11-30", FakeBS("2026-11-30"),
             FakeAK("2026-11-30", drop={"sh.600001": ["2026-11-10"]}), "2026-12-15")
    assert mf.load_manifest(cache / "manifest.json")["complete"] is False


def test_index_gap_raises(cache):
    with pytest.raises(CoverageError, match="sh.000300"):
        _run(cache, "2026-11-30", FakeBS("2026-11-30", drop={"sh.000300": ["2026-11-10"]}),
             FakeAK("2026-11-30"), "2026-12-15")


def test_xval_gap_raises(cache):
    with pytest.raises(FetchError, match="sz.000001"):
        _run(cache, "2026-11-30", FakeBS("2026-11-30", drop={"sz.000001": ["2026-11-12"]}),
             FakeAK("2026-11-30"), "2026-12-15")


def test_index_that_does_not_reach_end_raises(cache):
    with pytest.raises(CoverageError, match="ends 2026-11-20"):
        _run(cache, "2026-11-30", FakeBS("2026-11-20"), FakeAK("2026-11-30"), "2026-12-15")


# ---------------------------------------------------------------- (e) check_coverage
def test_check_coverage_flags_stale_index(cache):
    m = mf.load_manifest(cache / "manifest.json")
    assert mf.check_coverage(m, "2026-10-31") == []
    nov = [d for d in CAL if d <= "2026-11-30"]
    probs = mf.check_coverage(m, "2026-11-30", calendar=nov)
    assert any(p.startswith("index index/sh.000906.parquet") and "2026-11-30" in p for p in probs)
    assert any("universe" in p for p in probs) and any(p.startswith("sina:") for p in probs)
