"""Step 4: walk-forward stock-selection models and their predictive power."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mf.config import CFG, PROC
from mf.dataset import label_window, load_dataset
from mf.factor_test import ic_series, ic_stats, quantile_returns
from mf.models import predict_gbdt, predict_icir, predict_ridge

if __name__ == "__main__":
    data = load_dataset()
    selection = pd.read_parquet(PROC / "factor_selection.parquet")   # walk-forward factor screen

    preds = {
        "ICIR加权": predict_icir(data, selection),
        "Ridge": predict_ridge(data, selection),
        "GBDT": predict_gbdt(data),
    }
    scores = pd.DataFrame(preds)
    scores.to_parquet(PROC / "model_scores.parquet")

    S = CFG["split"]
    first_pred = scores["GBDT"].dropna().index.get_level_values(0).min()
    rows = []
    for name in scores:
        d = data.set_index(["date", "code"]).join(scores[name].rename("pred"), how="inner").reset_index()
        d = d[d.date >= first_pred]            # common evaluation window for all models
        for per, (a, b) in {"IS": (first_pred, S["is_end"]), "OOS": (S["oos_start"], S["oos_end"])}.items():
            sub = d[label_window(d, a, b)]     # only labels realised inside the period
            st = ic_stats(ic_series(sub, "pred"))
            q = quantile_returns(sub, "pred", 10)
            ls = q[10] - q[1]
            rows.append({"模型": name, "区间": per, "RankIC": st["IC均值"], "ICIR(年化)": st["ICIR(年化)"],
                         "IC>0占比": st["IC>0占比"], "Top10%月均超额": (q[10] - q.mean(axis=1)).mean(),
                         "多空年化": (1 + ls).prod() ** (12 / len(ls)) - 1})
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "model_eval.csv", index=False)

    # Q11: theoretical (all scored names) vs executable (could actually be bought at the next-day VWAP).
    # Next-day tradability is used ONLY for this evaluation, never as an input to the signal (it is future info).
    from mf.panel import load_long
    lg = load_long()[["code", "date", "traded", "one_price_up"]].rename(columns={"date": "exec_date"})
    ex_rows = []
    for name in scores:
        d = data.set_index(["date", "code"]).join(scores[name].rename("pred"), how="inner").reset_index()
        d = d[d.date >= first_pred].merge(lg, on=["code", "exec_date"], how="left")
        d["exec_ok"] = d.traded.fillna(False).astype(bool) & ~d.one_price_up.fillna(False).astype(bool)
        for per, (a, b) in {"IS": (first_pred, S["is_end"]), "OOS": (S["oos_start"], S["oos_end"])}.items():
            sub = d[label_window(d, a, b) & d.tradable]
            ex_rows.append({"模型": name, "区间": per, "理论 RankIC": ic_stats(ic_series(sub, "pred"))["IC均值"],
                            "可成交样本 RankIC": ic_stats(ic_series(sub[sub.exec_ok], "pred"))["IC均值"],
                            "次日不可买入占比": 1 - sub.exec_ok.mean(),
                            "前10%中次日不可买入占比": 1 - sub[sub.groupby("date").pred.rank(pct=True) > 0.9].exec_ok.mean()})
    pd.DataFrame(ex_rows).to_csv(PROC / "model_eval_exec.csv", index=False)
    print(pd.DataFrame(ex_rows).round(4).to_string())
    print(tab.round(4).to_string())
    is_tab = tab[tab["区间"] == "IS"].set_index("模型")
    best = is_tab["ICIR(年化)"].idxmax()     # model choice made on IS only
    from mf.forward import load_frozen
    import os
    frozen = None if os.environ.get("MF_IGNORE_FROZEN") else load_frozen()   # set only when preparing a NEW rules version
    if frozen and frozen["model"] != best:  # rules are frozen: keep the frozen model, but say so loudly
        print(f"WARNING: the IS rule now picks {best}, but the frozen model is {frozen['model']}; using the frozen one")
        best = frozen["model"]
    json.dump({"best_model": best, "first_pred": str(first_pred.date())}, open(PROC / "model_choice.json", "w"))
    print("model chosen on IS:", best)
