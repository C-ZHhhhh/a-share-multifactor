"""Freeze a NEW rules version (reports/forward/rules/v{n}.json). Never edits an existing version.

usage: python scripts/freeze_rules.py --reason "why a new version is needed"

Run it on a clean, committed working tree after the pipeline has produced the results you are freezing;
then commit reports/forward/ immediately (the commit time is the independent timestamp)."""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.config import CFG, PROC
from mf.forward import freeze_new_version
from mf.metrics import perf
from mf.panel import load_benchmarks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="for tests only")
    a = ap.parse_args()
    bt = pickle.load(open(PROC / "backtests.pkl", "rb"))
    model = json.load(open(PROC / "model_choice.json"))["best_model"]
    main = f"组合优化({bt['best']})"
    ret = bt["results"][main]["ret"]
    S = CFG["split"]
    bench = load_benchmarks().rec
    hist_end = str(ret.loc[:S["oos_end"]].index[-1].date())
    p = perf(ret.loc[S["oos_start"]:S["oos_end"]], bench)
    hist = {"oos_window": [S["oos_start"], S["oos_end"]], "oos_excess_ann": p["超额年化"], "oos_ir": p["信息比率"],
            "oos_te": p["跟踪误差"]}
    try:
        from mf.manifest import content_hash, load_manifest
        data_hash = content_hash(None) if load_manifest() else None
    except Exception:
        data_hash = None
    rules = freeze_new_version(model, bt["lam"], hist_end, a.reason, hist, data_hash, a.allow_dirty)
    print(json.dumps(rules, ensure_ascii=False, indent=2))
