"""Factor construction on daily wide panels, sampled at month-end rebalance dates.

All factors at date t use information available at the close of t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG
from .panel import load_index, to_wide

FACTOR_INFO = {
    # name: (category, description)
    "size":       ("Size",       "ln(流通市值)"),
    "ep":         ("Value",      "EPS_TTM / 价格 (财报按公告日对齐, 复权因子调整股本)"),
    "bp":         ("Value",      "每股净资产 / 价格"),
    "sp":         ("Value",      "每股营收_TTM / 价格"),
    "roe":        ("Quality",    "净利润_TTM / 净资产"),
    "gpm":        ("Quality",    "销售毛利率"),
    "np_yoy":     ("Growth",     "净利润同比增速 (最新财报)"),
    "rev_yoy":    ("Growth",     "营业收入同比增速 (最新财报)"),
    "mom_12_1":   ("Momentum",   "过去12个月收益, 剔除最近1个月"),
    "rev_1m":     ("Reversal",   "过去1个月收益"),
    "vol_20":     ("Volatility", "20日收益波动率"),
    "ivol_60":    ("Volatility", "60日特质波动率 (对中证800回归残差)"),
    "beta_250":   ("Volatility", "250日市场Beta"),
    "max_ret_20": ("Volatility", "20日最大单日收益 (彩票效应)"),
    "turn_20":    ("Liquidity",  "ln(20日平均换手率)"),
    "abn_turn":   ("Liquidity",  "20日均换手 / 250日均换手"),
    "illiq_20":   ("Liquidity",  "ln(Amihud 非流动性 |r|/成交额, 20日)"),
    "pv_corr_20": ("PriceVol",   "20日 收盘价-成交量 相关系数"),
    "skew_60":    ("PriceVol",   "60日收益偏度"),
}
FACTORS = list(FACTOR_INFO)


def rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)


def compute_daily_factors(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    W = {c: to_wide(long, c) for c in ["adj_close", "close", "ret", "turn", "amount", "float_mcap",
                                       "volume", "traded", "ep", "bp", "sp", "roe", "gpm",
                                       "np_yoy", "rev_yoy"]}
    traded = W["traded"].fillna(False).astype(bool)
    r = W["ret"].where(traded)                       # NaN on suspended days -> excluded from stats
    px = W["adj_close"].ffill()
    mkt = load_index(CFG["benchmark"]).ret.reindex(px.index)

    f = {}
    f["size"] = np.log(W["float_mcap"])
    # point-in-time fundamentals, already aligned to announcement dates in panel.py
    for c in ["ep", "bp", "sp", "roe", "gpm"]:
        f[c] = W[c]
    f["roe"] = f["roe"].clip(-1, 1)
    f["np_yoy"] = W["np_yoy"].clip(-500, 500)       # reported in %, heavy tails
    f["rev_yoy"] = W["rev_yoy"].clip(-200, 500)

    f["mom_12_1"] = px.shift(21) / px.shift(252) - 1
    f["rev_1m"] = px / px.shift(21) - 1
    f["vol_20"] = r.rolling(20, min_periods=15).std()

    rm = r.notna().mul(mkt, axis=0).where(r.notna())   # market return aligned to stock trading days
    def beta_and_resid(n, mp):
        mean_r = r.rolling(n, min_periods=mp).mean()
        mean_m = rm.rolling(n, min_periods=mp).mean()
        cov = (r * rm).rolling(n, min_periods=mp).mean() - mean_r * mean_m
        var_m = (rm ** 2).rolling(n, min_periods=mp).mean() - mean_m ** 2
        var_r = (r ** 2).rolling(n, min_periods=mp).mean() - mean_r ** 2
        beta = cov / var_m
        resid_var = (var_r - beta ** 2 * var_m).clip(lower=0)
        return beta, np.sqrt(resid_var)
    _, f["ivol_60"] = beta_and_resid(60, 40)
    f["beta_250"], _ = beta_and_resid(250, 160)
    f["max_ret_20"] = r.rolling(20, min_periods=15).max()

    turn = W["turn"].where(traded)
    t20 = turn.rolling(20, min_periods=10).mean()
    f["turn_20"] = np.log(t20.where(t20 > 0))
    f["abn_turn"] = t20 / turn.rolling(250, min_periods=120).mean()
    amihud = (r.abs() / W["amount"].where(traded)).rolling(20, min_periods=10).mean()
    f["illiq_20"] = np.log(amihud.where(amihud > 0) * 1e9)
    f["pv_corr_20"] = W["close"].where(traded).rolling(20, min_periods=15).corr(W["volume"].where(traded))
    f["skew_60"] = r.rolling(60, min_periods=40).skew()
    return f


# ------------------------------------------------------------------ preprocessing
def mad_winsorize(x: pd.Series, k: float = 5.0) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    return x.clip(med - k * mad, med + k * mad)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def neutralize(y: pd.Series, industry: pd.Series, size: pd.Series | None) -> pd.Series:
    """OLS residual of y on industry dummies (+ size). Returns z-scored residual."""
    X = pd.get_dummies(industry, dtype=float)
    if size is not None:
        X["size"] = zscore(size)
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return zscore(pd.Series(y.values - X.values @ beta, index=y.index))


def preprocess_cross_section(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """df: one rebalance date, rows=stocks, includes 'industry' and 'size'."""
    out = pd.DataFrame(index=df.index)
    for f in factors:
        x = df[f].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.groupby(df.industry).transform("median")).fillna(x.median())
        x = zscore(mad_winsorize(x))
        size = None if f == "size" else df["size_raw"]
        out[f] = neutralize(x, df.industry, size)
    return out
