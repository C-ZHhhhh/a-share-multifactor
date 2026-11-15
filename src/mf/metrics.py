from __future__ import annotations

import numpy as np
import pandas as pd


def perf(ret: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None,
         cost: pd.Series | None = None, rf: float | None = None) -> dict:
    """Annualised return/excess are geometric (what an investor actually earns). Ratios use arithmetic
    daily means, the textbook definitions:
      Sharpe = mean(r - rf_daily) * 252 / (std(r) * sqrt(252))
      IR     = mean(r - b) * 252 / (std(r - b) * sqrt(252))
    """
    from .config import CFG
    rf = CFG.get("risk_free", 0.0) if rf is None else rf
    ret = ret.dropna()
    n = len(ret)
    nav = (1 + ret).cumprod()
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = ret.std() * np.sqrt(252)
    dd = nav / nav.cummax() - 1
    sharpe = (ret.mean() - rf / 252) * 252 / vol if vol > 0 else np.nan
    out = {"年化收益": ann, "年化波动": vol, "夏普比率": sharpe,
           "最大回撤": dd.min(), "卡玛比率": ann / -dd.min() if dd.min() < 0 else np.nan}
    if bench is not None:
        b = bench.reindex(ret.index).fillna(0)
        bnav = (1 + b).cumprod()
        b_ann = bnav.iloc[-1] ** (252 / n) - 1
        active = ret - b
        te = active.std() * np.sqrt(252)
        ex_nav = nav / bnav
        ex_ann = ex_nav.iloc[-1] ** (252 / n) - 1
        mret = (1 + ret).groupby(ret.index.to_period("M")).prod()
        mb = (1 + b).groupby(b.index.to_period("M")).prod()
        out.update({"基准年化": b_ann, "超额年化": ex_ann, "跟踪误差": te,
                    "信息比率": active.mean() * 252 / te if te > 0 else np.nan,
                    "超额最大回撤": (ex_nav / ex_nav.cummax() - 1).min(),
                    "月度胜率": (mret > mb).mean()})
    if turnover is not None and len(turnover):
        years = n / 252
        out["年化单边换手"] = turnover.sum() / years
        out["年化成本拖累"] = cost.sum() / years if cost is not None else np.nan
    return out


def yearly_table(ret: pd.Series, bench: pd.Series) -> pd.DataFrame:
    b = bench.reindex(ret.index).fillna(0)
    y = pd.DataFrame({"策略": (1 + ret).groupby(ret.index.year).prod() - 1,
                      "基准": (1 + b).groupby(b.index.year).prod() - 1})
    y["超额"] = (1 + y["策略"]) / (1 + y["基准"]) - 1
    return y


def fmt_table(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    pct_cols = [c for c in df.columns if c not in ("夏普比率", "卡玛比率", "信息比率")]
    out = df.copy().astype(object)
    for c in df.columns:
        out[c] = df[c].map(lambda v: "-" if pd.isna(v) else (f"{v:.2%}" if c in pct_cols else f"{v:.2f}"))
    return out
