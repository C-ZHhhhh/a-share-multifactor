"""Step 5: risk model, portfolio construction (top-N and optimised) and daily backtests."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from mf.attribution import bench_weights
from mf.backtest import Market, run_backtest
from mf.config import CFG, PROC
from mf.dataset import load_dataset
from mf.metrics import perf
from mf.optimizer import grinold_alpha, optimize
from mf.panel import load_benchmarks, load_long
from mf.risk import estimate_factor_returns, risk_model_at


def topn_targets(scores: pd.Series, n: int) -> dict:
    out = {}
    for d, s in scores.dropna().groupby(level=0):
        top = s.droplevel(0).nlargest(n)
        out[d] = pd.Series(1.0 / len(top), index=top.index)
    return out


def optimised_targets(score: pd.Series, data, fret, resid, pcfg=None, label="opt") -> tuple[dict, pd.DataFrame]:
    targets, info, w0 = {}, [], None
    dates = sorted(score.dropna().index.get_level_values(0).unique())
    for d in tqdm(dates, desc=label):
        s = score.loc[d].dropna()
        X, F, D = risk_model_at(d, data, fret, resid)
        g = data[data.date == d].set_index("code")
        alpha = grinold_alpha(s, D.reindex(s.index).fillna(D.median()))
        w, inf = optimize(alpha, bench_weights(data, d), X, F, D, w0, g.tradable, pcfg)
        targets[d], w0 = w, w
        info.append({"date": d, **inf})
    return targets, pd.DataFrame(info).set_index("date")


if __name__ == "__main__":
    long = load_long()
    data = load_dataset()
    scores = pd.read_parquet(PROC / "model_scores.parquet")
    choice = json.load(open(PROC / "model_choice.json"))
    best, first_pred = choice["best_model"], pd.Timestamp(choice["first_pred"])
    scores = scores[scores.index.get_level_values(0) >= first_pred]

    fpath = PROC / "factor_returns.parquet"
    # cache is valid only if newer than the dataset it was estimated from
    if fpath.exists() and fpath.stat().st_mtime > (PROC / "dataset.parquet").stat().st_mtime:
        fret = pd.read_parquet(fpath)
        resid = pd.read_parquet(PROC / "specific_returns.parquet")
    else:
        print("estimating risk-model factor returns ...")
        fret, resid = estimate_factor_returns(data, long)

    mkt = Market.from_long(long)
    B = load_benchmarks()
    bench = B.rec          # strategy benchmark = the float-cap weights the optimiser tracks
    S = CFG["split"]

    # ---- risk-aversion calibration on IS only (target ex-ante TE 4-6%)
    is_score = scores[best][scores.index.get_level_values(0) <= S["is_end"]]
    calib = {}
    for lam in (2.0, 10.0, 50.0):
        pc = {**CFG["portfolio"], "risk_aversion": lam}
        tg, inf = optimised_targets(is_score, data, fret, resid, pc, f"calib λ={lam}")
        bt = run_backtest(tg, mkt, end=S["is_end"])
        p = perf(bt["ret"], bench, bt["turnover"], bt["cost"])
        calib[lam] = {"ex_ante_te": inf.ex_ante_te.mean(), "IR": p["信息比率"], "超额年化": p["超额年化"],
                      "跟踪误差": p["跟踪误差"], "持仓数": inf.n_holdings.mean()}
    calib = pd.DataFrame(calib).T
    print(calib.round(3))
    lam_best = float(calib["IR"].idxmax())
    pcfg = {**CFG["portfolio"], "risk_aversion": lam_best}

    # ---- final portfolios over the whole walk-forward window
    portfolios = {}
    opt_targets, opt_info = optimised_targets(scores[best], data, fret, resid, pcfg, "optimised")
    portfolios[f"组合优化({best})"] = opt_targets
    for m in scores.columns:
        portfolios[f"Top{CFG['portfolio']['topn']}等权({m})"] = topn_targets(scores[m], CFG["portfolio"]["topn"])

    results = {name: run_backtest(tg, mkt) for name, tg in portfolios.items()}
    results[f"组合优化({best}) 无成本"] = run_backtest(opt_targets, mkt, cost_on=False)
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
