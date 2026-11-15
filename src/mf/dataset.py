"""Monthly modelling dataset: (rebalance date, stock) -> factors, flags, forward returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, PROC
from .factors import FACTORS, compute_daily_factors, preprocess_cross_section, rebalance_dates
from .panel import load_meta, to_wide


def build_dataset(long: pd.DataFrame) -> pd.DataFrame:
    uni, industry, basic = load_meta()
    dates = pd.DatetimeIndex(sorted(long.date.unique()))
    rdates = rebalance_dates(dates)
    rdates = rdates[(rdates >= CFG["data"]["universe_start"])]

    daily_f = compute_daily_factors(long)
    W = {c: to_wide(long, c) for c in ["traded", "isST", "limit_up", "float_mcap", "vwap",
                                       "adj_close", "close"]}

    # ---- forward return label: execute at next-day VWAP, hold to the next rebalance's next-day VWAP
    adj_vwap = (W["vwap"] * W["adj_close"] / W["close"])
    adj_px_ff = W["adj_close"].ffill()
    pos = {d: i for i, d in enumerate(dates)}
    exec_dates = [dates[pos[d] + 1] if pos[d] + 1 < len(dates) else pd.NaT for d in rdates]
    entry = adj_vwap.reindex(exec_dates)                       # NaN if suspended -> can't enter
    exit_ = adj_vwap.reindex(exec_dates).fillna(adj_px_ff.reindex(exec_dates))
    fwd = pd.DataFrame(exit_.shift(-1).values / entry.values - 1, index=rdates, columns=entry.columns)
    # the label of rebalance d is only known after the NEXT rebalance's execution (its VWAP fill)
    label_dates = dict(zip(rdates, exec_dates[1:] + [pd.NaT]))

    uni["month"] = uni.month.astype(str)
    members = uni.groupby("month").code.apply(set)

    rows = []
    for d, ex in zip(rdates, exec_dates):
        m = d.strftime("%Y-%m")
        if m not in members.index:
            continue
        codes = sorted(members[m] & set(W["traded"].columns))
        df = pd.DataFrame({f: daily_f[f].loc[d].reindex(codes) for f in FACTORS})
        df["date"] = d
        df["exec_date"] = ex
        df["label_date"] = label_dates[d]
        df["industry"] = industry.reindex(codes).fillna("未分类").values
        df["mcap"] = W["float_mcap"].loc[d].reindex(codes)
        df["size_raw"] = df["size"]
        ipo = basic.ipoDate.reindex(codes)
        listed_days = (d - ipo).dt.days
        df["tradable"] = (
            W["traded"].loc[d].reindex(codes).fillna(False).astype(bool)
            & (W["isST"].loc[d].reindex(codes).fillna(1) == 0)
            & (listed_days.values >= CFG["filters"]["min_list_days"] * 365 / 250)
            & ~W["limit_up"].loc[d].reindex(codes).fillna(False).astype(bool)
            & df["mcap"].notna()
        )
        df["fwd_ret"] = fwd.loc[d].reindex(codes)
        rows.append(df.rename_axis("code").reset_index())
    data = pd.concat(rows, ignore_index=True)

    # cross-sectional preprocessing on the tradable universe only
    proc = []
    for d, g in data[data.tradable].groupby("date"):
        p = preprocess_cross_section(g.set_index("code"), FACTORS).add_prefix("z_")
        p["date"] = d
        proc.append(p.reset_index())
    data = data.merge(pd.concat(proc), on=["date", "code"], how="left")
    # rank-normalised label in [-0.5, 0.5] for model training
    data["fwd_rank"] = data[data.tradable].groupby("date").fwd_ret.rank(pct=True) - 0.5
    data.to_parquet(PROC / "dataset.parquet", index=False)
    return data


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(PROC / "dataset.parquet")


def label_window(data: pd.DataFrame, start, end) -> pd.Series:
    """Rows usable for research on [start, end]: signal date >= start AND the label is fully realised
    (label_date) by `end`. Every in-sample decision (factor screen, model choice, parameter choice) uses
    this mask, so no label that is realised after the in-sample end leaks into it."""
    return (data.date >= pd.Timestamp(start)) & (data.label_date <= pd.Timestamp(end))
