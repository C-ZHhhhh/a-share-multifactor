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
    sel = json.load(open(PROC / "selected_factors.json"))
    chosen, signs = sel["selected"], {k: int(v) for k, v in sel["signs"].items()}

    preds = {
        "ICIR加权": predict_icir(data, chosen, signs),
        "Ridge": predict_ridge(data, [f for f in chosen]),
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
    print(tab.round(4).to_string())
    is_tab = tab[tab["区间"] == "IS"].set_index("模型")
    best = is_tab["ICIR(年化)"].idxmax()     # model choice made on IS only
    json.dump({"best_model": best, "first_pred": str(first_pred.date())}, open(PROC / "model_choice.json", "w"))
    print("model chosen on IS:", best)
