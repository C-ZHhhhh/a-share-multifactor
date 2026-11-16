"""Step 5: risk model, portfolio construction (top-N and optimised) and daily backtests."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mf.backtest import Market, run_backtest
from mf.config import CFG, PROC
from mf.dataset import load_dataset
from mf.metrics import perf
from mf.panel import load_benchmarks, load_long
from mf.portfolio import calibrate_lambda, optimised_backtest, topn_targets
from mf.risk import estimate_factor_returns


if __name__ == "__main__":
    long = load_long()
    from mf.checks import validate_long
    chk = validate_long(long, load_benchmarks().index)
    print("data checks:", {k: v for k, v in chk.items() if not k.endswith("examples")})
    data = load_dataset()
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    choice = json.load(open(PROC / "model_choice.json"))
    best, first_pred = choice["best_model"], pd.Timestamp(choice["first_pred"])
    scores = scores[scores.index.get_level_values(0) >= first_pred]

    fpath = PROC / "factor_returns.parquet"
    # the cache is keyed on the CONTENT of its inputs (dataset + daily panel + risk-model code), not on mtimes
    import hashlib
    from mf.config import ROOT
    h = hashlib.sha256()
    for f in (PROC / "dataset.parquet", PROC / "daily_long.parquet", ROOT / "src/mf/risk.py"):
        h.update(f.read_bytes())
    key = h.hexdigest()[:16]
    meta = PROC / "factor_returns.key"
    if fpath.exists() and meta.exists() and meta.read_text().strip() == key:
        fret = pd.read_parquet(fpath)
        resid = pd.read_parquet(PROC / "specific_returns.parquet")
    else:
        print("estimating risk-model factor returns ...")
        fret, resid = estimate_factor_returns(data, long)
        meta.write_text(key)

    mkt = Market.from_long(long)
    B = load_benchmarks()
    bench = B.rec          # strategy benchmark = the float-cap weights the optimiser tracks
    S = CFG["split"]

    # ---- risk-aversion calibration on IS only: lambda whose mean ex-ante TE is closest to the target
    is_score = scores[best][scores.index.get_level_values(0) <= S["is_end"]]
    lam_best, calib = calibrate_lambda(is_score, data, fret, resid, mkt, bench, CFG["portfolio"], S["is_end"])
    print(calib.round(3)); print(f"λ = {lam_best:g} (target ex-ante TE {CFG['portfolio']['target_te']:.1%})")
    from mf.forward import load_frozen
    import os
    frozen = None if os.environ.get("MF_IGNORE_FROZEN") else load_frozen()   # set only when preparing a NEW rules version
    if frozen and float(frozen["lambda"]) != lam_best:   # rules are frozen
        print(f"WARNING: calibration now gives λ = {lam_best:g}, frozen λ = {frozen['lambda']:g}; using the frozen one")
        lam_best = float(frozen["lambda"])
    pcfg = {**CFG["portfolio"], "risk_aversion": lam_best}

    # ---- final portfolios over the whole walk-forward window
    main = f"组合优化({best})"
    opt_bt, opt_info = optimised_backtest(scores[best], data, fret, resid, mkt, pcfg, "optimised")
    opt_targets = opt_bt["targets"]
    results = {main: opt_bt}
    for m in scores.columns:
        results[f"Top{CFG['portfolio']['topn']}等权({m})"] = run_backtest(topn_targets(scores[m], CFG["portfolio"]["topn"]), mkt)
    results[f"{main} 无成本"] = run_backtest(opt_targets, mkt, cost_on=False)   # same targets, zero costs
    pickle.dump({"results": results, "opt_info": opt_info, "calib": calib, "lam": lam_best,
                 "best": best, "opt_targets": opt_targets}, open(PROC / "backtests.pkl", "wb"))

    for name, r in results.items():
        for per, (a, b) in {"IS": (first_pred, S["is_end"]), "OOS": (S["oos_start"], S["oos_end"])}.items():
            sub = r["ret"].loc[a:b]
            p = perf(sub, bench, r["turnover"].loc[a:b], r["cost"].loc[a:b])
            p2 = perf(sub, B.tri)
            print(f"{name:28s} {per:3s} ann={p['年化收益']:.2%} ex={p['超额年化']:.2%} "
                  f"IR={p['信息比率']:.2f} ex_vs_TRI={p2['超额年化']:.2%} MDD={p['最大回撤']:.2%} "
                  f"TO={p['年化单边换手']:.1f}")
