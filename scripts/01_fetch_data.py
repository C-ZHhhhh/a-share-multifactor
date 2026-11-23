"""Step 1: download / incrementally refresh data, then write data/raw/manifest.json.

* trading calendar / universe / industry / IPO dates / indices : baostock
* total-return indices (H00906 ...)                            : CSIndex via AkShare
* daily bars + adjustment factors                              : Sina via AkShare
* quarterly financial reports                                  : Eastmoney via AkShare (parallel thread)
* cross-validation sample (192 stocks)                         : baostock daily bars incl. true ST flag (audit only)

The refresh is incremental: anything that does not reach `data.end` is re-fetched, the overlap with
the cache is checked for revisions, and missing/gapped/stale data raises (exit code 1) instead of
being silently kept. Historical universe months are never rewritten.

Usage: python scripts/01_fetch_data.py [bars|reports|all] [--accept-revisions] [--end YYYY-MM-DD]
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf import manifest as mf
from mf.config import CFG, RAW

GROUPS_FOR = {
    "all": None,
    "bars": ["calendar", "universe", "meta", "index", "tr_index", "sina", "sina_factor"],
    "reports": ["calendar", "universe", "meta", "index", "tr_index", "reports"],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["all", "bars", "reports"])
    ap.add_argument("--accept-revisions", action="store_true",
                    help="overwrite cached history even when the source revised it beyond the threshold")
    ap.add_argument("--end", default=None, help="override data.end from config.yaml")
    args = ap.parse_args()
    end = args.end or CFG["data"]["end"]

    from mf.data_fetch_ak import refresh_all

    try:
        prev = mf.load_manifest(RAW / mf.MANIFEST_NAME)
    except (FileNotFoundError, ValueError):
        prev = None
    try:
        log = refresh_all(end=end, what=args.what, accept_revisions=args.accept_revisions,
                          prev_manifest=prev)
    except Exception as e:
        traceback.print_exc()
        m = mf.build_manifest(RAW, refresh={"end": end, "error": f"{type(e).__name__}: {e}"}, end=end)
        m["complete"] = False
        mf.write_manifest(m, RAW)
        print(f"\nREFRESH FAILED ({type(e).__name__}); manifest marked incomplete.\n{e}", file=sys.stderr)
        sys.exit(1)

    m = mf.build_manifest(RAW, refresh=log, end=end)
    problems = mf.check_coverage(m, end, groups=GROUPS_FOR[args.what])
    m["problems"] = problems
    m["complete"] = not problems
    path = mf.write_manifest(m, RAW)
    n_rev = sum(len(v) for v in log.get("revisions", {}).values())
    print(f"manifest -> {path}  content_hash={m['content_hash'][:12]}  revisions logged: {n_rev}  "
          f"expected last trading day: {log.get('expected_last_trading_day')}")
    if problems:
        print("COVERAGE PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    print("coverage OK")
