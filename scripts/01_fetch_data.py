"""Step 1: download data.

* universe / industry / IPO dates / indices : baostock (cached in data/raw)
* daily bars + adjustment factors          : Sina via AkShare
* quarterly financial reports              : Eastmoney via AkShare (separate thread)

Usage: python scripts/01_fetch_data.py [bars|reports|all]
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mf.config import RAW

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    uni_path = RAW / "universe_monthly.csv"
    if not uni_path.exists():                       # baostock only for metadata
        from mf.data_fetch import fetch_indices, fetch_universe
        fetch_universe()
        fetch_indices()
    from mf.data_fetch_ak import fetch_daily_sina, fetch_reports, fetch_tr_indices

    fetch_tr_indices()

    uni = pd.read_csv(uni_path)
    codes = sorted(uni.code.unique())
    print(f"universe: {len(codes)} unique stocks over {uni.month.nunique()} months", flush=True)

    rep = {}
    t = None
    if what in ("reports", "all"):
        t = threading.Thread(target=lambda: rep.update(fetch_reports()))
        t.start()
    if what in ("bars", "all"):
        res = fetch_daily_sina(codes)
        bad = {k: v for k, v in res.items() if not v.startswith(("ok", "cached"))}
        print(f"bars done. new={sum(v == 'ok' for v in res.values())} not-ok={len(bad)}", flush=True)
        if bad:
            pd.Series(bad).to_csv(RAW / "fetch_failures_sina.csv")
    if t:
        t.join()
        print("reports:", pd.Series(rep).str[:4].value_counts().to_dict(), flush=True)
