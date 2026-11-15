"""One self-contained, interactive HTML report per run: reports/report.html (+ a timestamped copy in reports/runs/).

Everything the page shows is computed here from the backtest objects and embedded as JSON; the page draws its
own SVG charts with plain JavaScript (src/mf/report_template.html), so it opens offline with no extra packages.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from .config import CFG, PROC, REPORTS, ROOT
from .factors import FACTOR_INFO
from .holdings import _names, latest_snapshot

TEMPLATE = ROOT / "src" / "mf" / "report_template.html"
SHORT_CN = {"size": "市值", "ep": "盈利收益率 EP", "bp": "账面市值比 BP", "sp": "营收市值比 SP", "roe": "ROE",
            "gpm": "毛利率", "np_yoy": "净利润同比", "rev_yoy": "营收同比", "mom_12_1": "12-1月动量",
            "rev_1m": "1月反转", "vol_20": "20日波动率", "ivol_60": "60日特质波动", "beta_250": "Beta",
            "max_ret_20": "20日最大涨幅", "turn_20": "20日换手率", "abn_turn": "异常换手", "illiq_20": "非流动性",
            "pv_corr_20": "量价相关性", "skew_60": "60日偏度"}
STYLE_CN = {"industry": "行业", "size": "市值", "beta": "Beta", "momentum": "动量", "volatility": "波动率",
            "value": "价值", "liquidity": "流动性", "specific+trading": "个股选择+交易"}


def _git_commit() -> str:
    """Short commit hash, flagged when the working tree has uncommitted code changes (reports/ excluded)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout
        h = run("rev-parse", "--short", "HEAD").strip() or "-"
        dirty = run("status", "--porcelain", "--", ".", ":(exclude)reports").strip()
        return h + (" (含未提交改动)" if dirty else "")
    except Exception:
        return "-"


def _selection_reasons(tab: pd.DataFrame, selected: list, corr: pd.DataFrame,
                       t_min=2.0, icir_min=0.5, corr_max=0.6) -> dict:
    """Why each factor was or wasn't selected (mirrors factor_test.select_factors)."""
    out = {}
    for f in tab.index:
        t, icir = tab.loc[f, "t值"], tab.loc[f, "ICIR(年化)"]
        if f in selected:
            out[f] = "入选"
        elif abs(t) < t_min:
            out[f] = f"|t| = {abs(t):.2f} < {t_min:g}"
        elif abs(icir) < icir_min:
            out[f] = f"|ICIR| = {abs(icir):.2f} < {icir_min:g}"
        else:
            g = max(selected, key=lambda g: abs(corr.loc[f, g]))
            out[f] = f"与已入选的 {SHORT_CN.get(g, g)} 相关性 {corr.loc[f, g]:.2f}，超过 {corr_max:g}"
    return out


def _num(x, nd=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _table(df: pd.DataFrame, index_name: str | None = None) -> dict:
    if index_name:
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: index_name})
    return {"columns": [str(c) for c in df.columns],
            "rows": [[v if isinstance(v, str) else _num(v, 4) if isinstance(v, (float, np.floating)) else
                      (int(v) if isinstance(v, (np.integer,)) else str(v)) for v in row] for row in df.values]}


def monthly_ic(scores: pd.Series, data: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC of the model score vs next-month return, per signal date."""
    y = data.set_index(["date", "code"]).fwd_ret
    df = pd.DataFrame({"s": scores, "y": y.reindex(scores.index)}).dropna()
    return df.groupby(level=0).apply(lambda g: g.s.rank().corr(g.y.rank()))


def build_html_report(*, res: dict, main: str, top_name: str, bench: pd.Series, tri: pd.Series,
                      opt_info: pd.DataFrame, perf_tabs: dict, yearly: pd.DataFrame, model_eval: pd.DataFrame,
                      calib: pd.DataFrame, lam, best: str, first_pred: pd.Timestamp, scores: pd.Series,
                      data: pd.DataFrame, br_df: pd.DataFrame, fa_df: pd.DataFrame, oos_br: pd.DataFrame,
                      holdings_main: pd.DataFrame, factor_tests_is: pd.DataFrame, selected: dict) -> dict:
    S = CFG["split"]
    r = res[main]["ret"]
    idx = r.index
    series = {"strat": r, "nocost": res[main + " 无成本"]["ret"].reindex(idx), "top": res[top_name]["ret"].reindex(idx),
              "bench": bench.reindex(idx).fillna(0), "tri": tri.reindex(idx).fillna(0)}

    ic = monthly_ic(scores, data)
    sel = set(selected["selected"])
    ft = factor_tests_is.set_index("因子")
    corr = pd.read_csv(PROC / "factor_corr_is.csv", index_col=0)
    why = _selection_reasons(ft, selected["selected"], corr)
    factors = [{"code": f, "short": SHORT_CN.get(f, f), "why": why[f], "name": FACTOR_INFO[f][1], "cat": FACTOR_INFO[f][0], "t": _num(ft.loc[f, "t值"], 3),
                "icir": _num(ft.loc[f, "ICIR(年化)"], 3), "selected": f in sel,
                "sign": selected["signs"].get(f)} for f in ft.index if f in FACTOR_INFO]

    snap = latest_snapshot(holdings_main, data)
    cur = snap["cur"]
    top_b = snap["bench"].sort_values(ascending=False).head(10)
    wmap = cur.set_index("code").weight
    allnames = _names()
    ind = snap["industry_active"]
    ind = ind[ind.abs() >= 0.005]
    info = opt_info.loc[pd.Timestamp(snap["date"])]

    payload = {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": _git_commit(),
                 "data_start": str(idx[0].date()), "data_end": str(idx[-1].date()),
                 "first_pred": str(first_pred.date()), "is_end": S["is_end"], "oos_start": S["oos_start"],
                 "oos_end": S["oos_end"], "model": best, "lam": _num(lam, 3), "main": main, "top": top_name,
                 "n_factors": len(sel), "costs": CFG["costs"], "portfolio": CFG["portfolio"]},
        "daily": {"dates": [d.strftime("%Y-%m-%d") for d in idx],
                  **{k: [_num(v) for v in s.values] for k, s in series.items()}},
        "monthly": {"dates": [d.strftime("%Y-%m-%d") for d in opt_info.index],
                    "te": [_num(v, 4) for v in opt_info.ex_ante_te], "n": [int(v) for v in opt_info.n_holdings],
                    "turnover": [_num(v, 4) for v in opt_info.turnover], "relax": [_num(v, 2) for v in opt_info.relax]},
        "ic": {"dates": [d.strftime("%Y-%m-%d") for d in ic.index], "values": [_num(v, 4) for v in ic.values]},
        "yearly": [{"year": int(y), "strat": _num(row["策略"]), "bench": _num(row["基准"]), "excess": _num(row["超额"])}
                   for y, row in yearly.iterrows()],
        "factors": factors,
        "perf": {k: _table(v, "组合") for k, v in perf_tabs.items()},
        "model_eval": _table(model_eval.round(4), None),
        "calib": _table(calib, "λ"),
        "brinson": {p: {k: _num(br_df.loc[p, k]) for k in ["allocation", "selection", "interaction", "合计"]}
                    for p in br_df.index},
        "factor_attr": {p: [{"name": STYLE_CN.get(k, k), "value": _num(fa_df.loc[p, k])}
                            for k in fa_df.columns if k != "active"] for p in fa_df.index},
        "oos_industry": [{"name": i, "value": _num(v)} for i, v in
                         pd.concat([oos_br.total.head(6), oos_br.total.tail(6)]).items()],
        "latest": {
            "date": str(snap["date"]), "n_ranked": snap["n_ranked"], "turnover": _num(info.turnover, 4),
            "te": _num(info.ex_ante_te, 4), "relax": _num(info.relax, 2),
            "holdings": [{"code": row.code, "name": row.name, "industry": row.industry, "w": _num(row.weight),
                          "b": _num(row.bench_weight), "a": _num(row.active_weight), "chg": _num(row.change),
                          "new": bool(row.is_new), "rank": int(row.model_rank)} for row in cur.itertuples()],
            "sold": [{"code": c, "name": allnames.get(c, c)} for c in snap["sold"]],
            "bench_top": [{"code": c, "name": allnames.get(c, c), "b": _num(v), "w": _num(wmap.get(c, 0.0))}
                          for c, v in top_b.items()],
            "industry": [{"name": i, "value": _num(v)} for i, v in ind.items()],
        },
    }
    return payload


def write_html_report(payload: dict) -> list:
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", blob)
    out = [REPORTS / "report.html"]
    runs = REPORTS / "runs"
    runs.mkdir(exist_ok=True)
    out.append(runs / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    for p in out:
        p.write_text(html, encoding="utf-8")
    return out
