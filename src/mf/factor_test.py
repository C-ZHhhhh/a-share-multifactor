"""Single-factor effectiveness tests: IC / RankIC, ICIR, t-stat, quintile returns,
IC decay, factor turnover and correlation. Factor selection uses in-sample only."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CFG, FIG
from .factors import FACTOR_INFO
from .plotting import setup_style


def ic_series(data: pd.DataFrame, col: str, label: str = "fwd_ret", method: str = "spearman") -> pd.Series:
    d = data[data.tradable & data[label].notna()]
    return d.groupby("date").apply(lambda g: g[col].corr(g[label], method=method))


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    mu, sd = ic.mean(), ic.std()
    return {"IC均值": mu, "IC标准差": sd, "ICIR(年化)": mu / sd * np.sqrt(12) if sd > 0 else np.nan,
            "t值": mu / sd * np.sqrt(n) if sd > 0 else np.nan, "IC>0占比": (ic > 0).mean(), "月数": n}


def quantile_returns(data: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    d = data[data.tradable & data.fwd_ret.notna() & data[col].notna()].copy()
    d["grp"] = d.groupby("date")[col].transform(lambda x: pd.qcut(x.rank(method="first"), q, labels=False) + 1)
    return d.groupby(["date", "grp"]).fwd_ret.mean().unstack()


def factor_autocorr(data: pd.DataFrame, col: str) -> float:
    w = data[data.tradable].pivot(index="date", columns="code", values=col)
    ac = [w.iloc[i].corr(w.iloc[i - 1], method="spearman") for i in range(1, len(w))]
    return float(np.nanmean(ac))


def ic_decay(data: pd.DataFrame, col: str, lags: int = 6) -> list[float]:
    w = data.pivot(index="date", columns="code", values="fwd_ret")
    f = data[data.tradable].pivot(index="date", columns="code", values=col)
    out = []
    for k in range(lags):
        lab = w.shift(-k)
        out.append(np.nanmean([f.loc[d].corr(lab.loc[d], method="spearman") for d in f.index[: len(f) - k]]))
    return out


def run_factor_tests(data: pd.DataFrame, period: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    from .dataset import label_window
    sub = data[label_window(data, *period)]
    rows, extra = [], {"ic": {}, "quant": {}}
    for f in FACTOR_INFO:
        col = f"z_{f}"
        ric = ic_series(sub, col)
        pic = ic_series(sub, col, method="pearson")
        qr = quantile_returns(sub, col)
        s = ic_stats(ric)
        ls = (qr[5] - qr[1]) * np.sign(s["IC均值"])
        mono = pd.Series(qr.mean().values).corr(pd.Series(range(5)), method="spearman")
        s.update({"因子": f, "类别": FACTOR_INFO[f][0], "IC(Pearson)": pic.mean(),
                  "多空年化": (1 + ls).prod() ** (12 / len(ls)) - 1,
                  "多空夏普": ls.mean() / ls.std() * np.sqrt(12),
                  "单调性": mono * np.sign(s["IC均值"]), "因子自相关": factor_autocorr(sub, col)})
        rows.append(s)
        extra["ic"][f], extra["quant"][f] = ric, qr
    tab = pd.DataFrame(rows).set_index("因子")
    return tab, extra


def select_factors(tab: pd.DataFrame, data: pd.DataFrame, t_min: float = 2.0, icir_min: float = 0.5,
                   corr_max: float = 0.6) -> tuple[list[str], dict[str, int], pd.DataFrame]:
    """Greedy selection by |ICIR|, dropping factors too correlated with an already-selected one."""
    cand = tab[(tab["t值"].abs() >= t_min) & (tab["ICIR(年化)"].abs() >= icir_min)]
    cand = cand.reindex(cand["ICIR(年化)"].abs().sort_values(ascending=False).index)
    from .dataset import label_window
    corr = factor_corr(data[label_window(data, CFG["split"]["is_start"], CFG["split"]["is_end"])])
    chosen = []
    for f in cand.index:
        if all(abs(corr.loc[f, g]) < corr_max for g in chosen):
            chosen.append(f)
    signs = {f: int(np.sign(tab.loc[f, "IC均值"])) for f in chosen}
    return chosen, signs, corr


def factor_corr(data: pd.DataFrame) -> pd.DataFrame:
    cols = [f"z_{f}" for f in FACTOR_INFO]
    c = data[data.tradable].groupby("date")[cols].corr(method="spearman")
    c = c.groupby(level=1).mean().loc[cols, cols]
    c.index = c.columns = list(FACTOR_INFO)
    return c


# ---------------------------------------------------------------- plots
def plot_factor_tests(tab: pd.DataFrame, extra: dict, corr: pd.DataFrame, tag: str):
    setup_style()
    fs = list(FACTOR_INFO)
    n = len(fs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    for ax, f in zip(axes.flat, fs):
        nav = (1 + extra["quant"][f]).cumprod()
        for g in nav.columns:
            ax.plot(nav.index, nav[g], label=f"Q{g}", lw=1.2, color=plt.cm.RdYlGn((g - 1) / 4))
        ax.set_title(f"{f}  RankIC={tab.loc[f, 'IC均值']:.3f}  ICIR={tab.loc[f, 'ICIR(年化)']:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / f"quantile_nav_{tag}.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Average cross-sectional Spearman correlation of neutralised factors")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(FIG / f"factor_corr_{tag}.png", dpi=120)
    plt.close()

    ic = pd.DataFrame(extra["ic"])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    (ic.cumsum()).plot(ax=ax, lw=1.2, colormap="tab20")
    ax.set_title("Cumulative RankIC (sign as raw factor)")
    ax.legend(ncol=6, fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / f"cum_ic_{tag}.png", dpi=120)
    plt.close()


def plot_ic_decay(decay: dict[str, list[float]], tag: str):
    setup_style()
    df = pd.DataFrame(decay, index=[f"t+{k + 1}" for k in range(len(next(iter(decay.values()))))])
    ax = df.plot(kind="bar", figsize=(11, 4), width=0.85, colormap="tab10")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("IC decay: RankIC with returns k months ahead")
    plt.tight_layout()
    plt.savefig(FIG / f"ic_decay_{tag}.png", dpi=120)
    plt.close()
