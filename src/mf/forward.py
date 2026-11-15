"""Frozen rules and the forward (genuinely new) sample.

The 2022-01 .. 2026-10 out-of-sample period was looked at repeatedly while the method was being fixed
(v1-v5), so it is no longer a clean test. From the freeze on:

  reports/forward/frozen_rules.json   written once (scripts/freeze_rules.py) and never changed: model, lambda,
                                      factor-screen rule, constraints, costs, split, the code commit and the
                                      last data day seen during development (history_end)
  reports/forward/registry.csv        append-only log of every TARGET portfolio whose holding period lies
                                      after history_end, with a hash, registered the first time it is produced

Forward performance = the strategy's daily returns after history_end, valid only while (a) the run uses the
frozen rules and (b) the targets it trades are the registered ones. Anything else is reported as invalid.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd

from .config import CFG, REPORTS

FWD = REPORTS / "forward"
RULES = FWD / "frozen_rules.json"
REGISTRY = FWD / "registry.csv"
# the code that produces targets and returns; a change here means the forward sample is no longer the frozen strategy
CORE_CODE = ["src/mf/panel.py", "src/mf/dataset.py", "src/mf/factors.py", "src/mf/factor_test.py", "src/mf/models.py",
             "src/mf/risk.py", "src/mf/optimizer.py", "src/mf/portfolio.py", "src/mf/backtest.py",
             "scripts/03_factors.py", "scripts/04_models.py", "scripts/05_portfolio_backtest.py"]
FACTOR_RULE = {"t_min": 2.0, "icir_min": 0.5, "corr_max": 0.6, "min_months": 24, "walk_forward": True}


def load_frozen() -> dict | None:
    return json.load(open(RULES, encoding="utf-8")) if RULES.exists() else None


def code_hash() -> str:
    from .config import ROOT
    h = hashlib.sha256()
    for f in CORE_CODE:
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def current_rules(model: str, lam: float) -> dict:
    port = {k: v for k, v in CFG["portfolio"].items() if k != "risk_aversion"}
    return {"model": model, "lambda": float(lam), "portfolio": port, "costs": CFG["costs"], "split": CFG["split"],
            "risk_free": CFG.get("risk_free"), "factor_rule": FACTOR_RULE, "model_params": CFG["model"],
            "core_code_sha256": code_hash()}


def freeze(model: str, lam: float, history_end: str, commit: str) -> dict:
    if RULES.exists():
        raise FileExistsError(f"{RULES} already exists: frozen rules are never overwritten")
    FWD.mkdir(parents=True, exist_ok=True)
    rules = {"frozen_on": str(date.today()), "history_end": history_end, "commit": commit,
             **current_rules(model, lam)}
    json.dump(rules, open(RULES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rules


def rule_diff(frozen: dict, model: str, lam: float) -> list[str]:
    """Differences between the frozen rules and the rules this run used (empty = compliant)."""
    cur = current_rules(model, lam)
    return [k for k in cur if json.dumps(cur[k], sort_keys=True, default=str) !=
            json.dumps(frozen.get(k), sort_keys=True, default=str)]


def _hash(w: pd.Series) -> str:
    w = w[w > 0].sort_index().round(8)
    return hashlib.sha256(";".join(f"{k}:{v:.8f}" for k, v in w.items()).encode()).hexdigest()[:16]


def register(targets: dict, history_end: str, compliant: bool) -> pd.DataFrame:
    """Append targets whose holding period starts after history_end (the last signal on or before
    history_end, and every later one). Existing rows are never changed; a recomputed target that differs
    from its registered hash is flagged (e.g. after a data revision)."""
    he = pd.Timestamp(history_end)
    sigs = sorted(targets)
    first = max([s for s in sigs if s <= he], default=None)
    fwd = [s for s in sigs if first is not None and s >= first]
    reg = pd.read_csv(REGISTRY, dtype=str) if REGISTRY.exists() else pd.DataFrame(
        columns=["signal_date", "registered_on", "n_holdings", "hash", "rules_compliant"])
    known = dict(zip(reg.signal_date, reg.hash))
    new_rows, mismatch = [], []
    for s in fwd:
        key, h = str(s.date()), _hash(targets[s])
        if key in known:
            if known[key] != h:
                mismatch.append(key)
        elif compliant:
            new_rows.append({"signal_date": key, "registered_on": str(date.today()),
                             "n_holdings": int((targets[s] > 0).sum()), "hash": h, "rules_compliant": "yes"})
    if new_rows:
        reg = pd.concat([reg, pd.DataFrame(new_rows)], ignore_index=True)
        FWD.mkdir(parents=True, exist_ok=True)
        reg.to_csv(REGISTRY, index=False)
        for s in fwd:                                   # keep the registered weights themselves too
            if str(s.date()) in {r["signal_date"] for r in new_rows}:
                targets[s][targets[s] > 0].rename("weight").to_csv(FWD / f"target_{s.date()}.csv")
    reg.attrs["mismatch"] = mismatch
    return reg


def forward_summary(ret: pd.Series, bench: pd.Series, history_end: str, reg: pd.DataFrame, diff: list) -> dict:
    """Forward-period statistics from daily returns after history_end (only reported, never used to tune)."""
    he = pd.Timestamp(history_end)
    r = ret[ret.index > he]
    b = bench.reindex(r.index).fillna(0)
    out = {"history_end": history_end, "days": int(len(r)), "rules_compliant": not diff, "rule_diff": diff,
           "targets_match_registry": not reg.attrs.get("mismatch"), "mismatch": reg.attrs.get("mismatch", []),
           "registered": reg.to_dict(orient="records")}
    if len(r):
        nav, bnav = (1 + r).prod(), (1 + b).prod()
        out.update({"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "cum_return": nav - 1,
                    "bench_cum_return": bnav - 1, "cum_excess": nav / bnav - 1,
                    "dates": [d.strftime("%Y-%m-%d") for d in r.index],
                    "strat": [float(x) for x in r.values], "bench": [float(x) for x in b.values]})
        if len(r) >= 60:
            act = r - b
            out["ir"] = float(act.mean() / act.std() * (252 ** 0.5))
    out["valid"] = out["rules_compliant"] and out["targets_match_registry"]
    return out


def write_forward_md(fwd: dict) -> None:
    pct = lambda v: f"{v:.2%}"
    reg = pd.DataFrame(fwd["registered"])
    lines = ["# 前瞻样本跟踪 (自动生成)", "",
             f"- 规则冻结于 **{fwd['frozen_on']}** (代码版本 {fwd['frozen_commit']}), 规则见 [frozen_rules.json](frozen_rules.json)。",
             f"- 开发过程中看过的最后一个数据日是 **{fwd['history_end']}**: 在此之前的所有结果 (含 2022-01 起的'样本外') 都被反复查看过, 只能算历史回测。",
             "- 只有此后的收益才算前瞻样本; 每期目标仓位在第一次生成时登记 (registry.csv, 只追加不修改), 前瞻收益必须来自已登记的目标。", "",
             f"**本次运行状态**: 规则与冻结版本一致: {'是' if fwd['rules_compliant'] else '否 (' + ', '.join(fwd['rule_diff']) + ')'}; "
             f"交易的目标与登记一致: {'是' if fwd['targets_match_registry'] else '否 (' + ', '.join(fwd['mismatch']) + ')'}; "
             f"前瞻结果有效: **{'是' if fwd['valid'] else '否'}**", ""]
    if fwd["days"]:
        lines += [f"**前瞻区间** {fwd['start']} ~ {fwd['end']} ({fwd['days']} 个交易日): 组合 {pct(fwd['cum_return'])}, "
                  f"基准 {pct(fwd['bench_cum_return'])}, 累计超额 {pct(fwd['cum_excess'])}"
                  + (f", 年化 IR {fwd['ir']:.2f}" if "ir" in fwd else " (不足 60 个交易日, 不计算 IR)"), ""]
    else:
        lines += ["**前瞻区间**: 还没有冻结之后的数据。下一次更新数据 (把 config.yaml 的 data.end 往后改并重新下载) 后, 这里会出现第一批前瞻结果。", ""]
    lines += ["## 已登记的目标仓位", "", reg.to_markdown(index=False) if len(reg) else "(无)", ""]
    (FWD / "README.md").write_text("\n".join(lines), encoding="utf-8")
