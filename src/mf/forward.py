"""Frozen rules (versioned) and the strictly pre-registered forward sample.

Everything up to `history_end` was looked at repeatedly during development (v1-v6), so it is HISTORY. A
forward return only counts if it comes from a target portfolio that was registered BEFORE it could be traded,
under a rules version that was frozen before the registration, and whose registration is provable by an
independent timestamp (the git commit that added the registry row).

Files (reports/forward/):
  rules/v{n}.json          frozen rule versions; never edited; a new version names the one it supersedes and why
  registry.csv             append-only event log: one row per target the first time it is produced
  registry_v1_legacy.csv   the v1 registry, kept as it was (its only entry was registered after execution)
  target_<date>.csv        the registered weights
  env_v{n}.txt             `pip freeze` of the environment a version was frozen with

Validity is decided per row and per day:
  row  valid_preregistered  registered_at_utc < execution bound (09:30 Asia/Shanghai on the next weekday
                            after the signal) AND rules frozen_at_utc <= registered_at_utc
       late_registration    otherwise (kept, never counts; back-filling cannot make it valid)
  day  strict forward day   every target that drives the day's return is a valid row, its hash matches the
                            target actually traded, its git commit time is before the execution bound, the
                            run uses the frozen rules, and the data passed the coverage checks
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import CFG, REPORTS, ROOT

FWD = REPORTS / "forward"
RULES_DIR = FWD / "rules"
REGISTRY = FWD / "registry.csv"
CST = timezone(timedelta(hours=8))
REG_COLS = ["signal_date", "signal_time", "execution_bound", "registered_at_utc", "rules_version", "rules_sha",
            "target_sha", "n_holdings", "status", "note"]
FACTOR_RULE = {"t_min": 2.0, "icir_min": 0.5, "corr_max": 0.6, "min_months": 24, "walk_forward": True}
# everything that produces data, targets, returns and the reported numbers
CORE_GLOBS = ["src/mf/*.py", "src/mf/report_template.html", "src/mf/xval_codes.txt", "scripts/*.py",
              "config.yaml", "requirements.txt", "run_all.sh"]


# ---------------------------------------------------------------- versions
def _versions() -> list:
    return sorted(RULES_DIR.glob("v*.json"), key=lambda p: int(p.stem[1:])) if RULES_DIR.exists() else []


def load_frozen() -> dict | None:
    """The current (latest) frozen rules version, with its file sha."""
    v = _versions()
    if not v:
        return None
    rules = json.load(open(v[-1], encoding="utf-8"))
    rules["version"] = v[-1].stem
    rules["rules_sha"] = hashlib.sha256(v[-1].read_bytes()).hexdigest()[:16]
    return rules


def code_files() -> list:
    files = set()
    for g in CORE_GLOBS:
        files |= set(ROOT.glob(g))
    return sorted(files)


def code_hash() -> str:
    """Fingerprint of all code and configuration. config.yaml is hashed WITHOUT its `data` section, because
    moving data.end forward every month is the normal forward workflow, not a rule change."""
    import yaml
    h = hashlib.sha256()
    for f in code_files():
        h.update(str(f.relative_to(ROOT)).encode())
        if f.name == "config.yaml":
            cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
            cfg.pop("data", None)
            h.update(json.dumps(cfg, sort_keys=True).encode())
        else:
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def current_rules(model: str, lam: float) -> dict:
    port = {k: v for k, v in CFG["portfolio"].items() if k != "risk_aversion"}
    return {"model": model, "lambda": float(lam), "portfolio": port, "costs": CFG["costs"], "split": CFG["split"],
            "filters": CFG["filters"], "risk_free": CFG.get("risk_free"), "factor_rule": FACTOR_RULE,
            "model_params": CFG["model"], "code_sha256": code_hash()}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()


def freeze_new_version(model: str, lam: float, history_end: str, reason: str, hist_metrics: dict,
                       data_hash: str | None, allow_dirty: bool = False) -> dict:
    """Write rules/v{n+1}.json. Refuses on a dirty working tree (outside data/ and reports/) so the frozen
    code is exactly a commit; records the full SHA, the Python environment and the data content hash."""
    dirty = [l for l in _git("status", "--porcelain").splitlines()
             if not l[3:].startswith(("data/", "reports/"))]
    if dirty and not allow_dirty:
        raise RuntimeError("refusing to freeze on a dirty working tree; commit first:\n" + "\n".join(dirty))
    v = _versions()
    n = int(v[-1].stem[1:]) + 1 if v else 1
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    env = subprocess.run([str(ROOT / ".venv/bin/pip"), "freeze"], capture_output=True, text=True).stdout
    (FWD / f"env_v{n}.txt").write_text(env, encoding="utf-8")
    import platform
    rules = {"version": f"v{n}", "supersedes": v[-1].stem if v else None, "reason": reason,
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "git_sha": _git("rev-parse", "HEAD"), "python": platform.python_version(),
             "env_sha256": hashlib.sha256(env.encode()).hexdigest()[:16], "data_content_sha256": data_hash,
             "history_end": history_end, "historical_metrics_at_freeze": hist_metrics,
             **current_rules(model, lam)}
    p = RULES_DIR / f"v{n}.json"
    json.dump(rules, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rules


def rule_diff(frozen: dict, model: str, lam: float) -> list:
    cur = current_rules(model, lam)
    return [k for k in cur if json.dumps(cur[k], sort_keys=True, default=str) !=
            json.dumps(frozen.get(k), sort_keys=True, default=str)]


# ---------------------------------------------------------------- registration
def target_sha(w: pd.Series) -> str:
    w = w[w > 0].sort_index().round(8)
    return hashlib.sha256(";".join(f"{k}:{v:.8f}" for k, v in w.items()).encode()).hexdigest()[:16]


def execution_bound(signal_date) -> datetime:
    """Earliest possible execution: 09:30 Beijing time on the next weekday (holidays only make it later)."""
    d = pd.Timestamp(signal_date) + pd.offsets.BDay(1)
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=CST)


def _utc(s) -> datetime:
    return datetime.fromisoformat(str(s)).astimezone(timezone.utc)


def load_registry() -> pd.DataFrame:
    return pd.read_csv(REGISTRY, dtype=str, keep_default_na=False) if REGISTRY.exists() \
        else pd.DataFrame(columns=REG_COLS)


def register(targets: dict, rules: dict, compliant: bool, now: datetime | None = None) -> pd.DataFrame:
    """Append (never modify) a row for every target whose holding period lies after history_end and that is
    not registered yet. Its status is fixed at registration time and cannot be changed by back-filling."""
    now = now or datetime.now(timezone.utc)
    he = pd.Timestamp(rules["history_end"])
    sigs = sorted(targets)
    first = max([s for s in sigs if s <= he], default=None)
    fwd = [s for s in sigs if first is not None and s >= first]
    reg = load_registry()
    # rows are keyed by (signal date, rules version): a new rules version registers its own targets and never
    # touches rows of older versions; integrity is checked only against rows of the current version
    cur_rows = reg[reg.rules_version == rules["version"]]
    known = dict(zip(cur_rows.signal_date, cur_rows.target_sha))
    frozen_at = _utc(rules["frozen_at_utc"]) if rules.get("frozen_at_utc") else None
    new, mismatch = [], []
    for s in fwd:
        key, h = str(s.date()), target_sha(targets[s])
        if key in known:
            if known[key] != h:
                mismatch.append(key)
            continue
        if not compliant:
            continue
        bound = execution_bound(s)
        ok = frozen_at is not None and frozen_at <= now < bound.astimezone(timezone.utc)
        new.append({"signal_date": key, "signal_time": datetime(s.year, s.month, s.day, 15, 0, tzinfo=CST).isoformat(),
                    "execution_bound": bound.isoformat(), "registered_at_utc": now.isoformat(timespec="seconds"),
                    "rules_version": rules["version"], "rules_sha": rules["rules_sha"], "target_sha": h,
                    "n_holdings": str(int((targets[s] > 0).sum())),
                    "status": "valid_preregistered" if ok else "late_registration",
                    "note": "" if ok else ("registered after the execution bound" if frozen_at and now >= bound
                                           else "rules not frozen before registration")})
    if new:
        reg = pd.concat([reg, pd.DataFrame(new)], ignore_index=True)[REG_COLS]
        FWD.mkdir(parents=True, exist_ok=True)
        reg.to_csv(REGISTRY, index=False)
        for r in new:
            s = pd.Timestamp(r["signal_date"])
            targets[s][targets[s] > 0].rename("weight").to_csv(FWD / f"target_{r['signal_date']}.csv")
    reg.attrs["mismatch"] = mismatch
    return reg


def git_commit_time(target_hash: str) -> datetime | None:
    """Time of the earliest commit whose diff of registry.csv contains this target hash (independent timestamp)."""
    out = _git("log", "--format=%cI", "-S", target_hash, "--", str(REGISTRY.relative_to(ROOT)))
    times = [datetime.fromisoformat(t) for t in out.splitlines() if t]
    return min(times).astimezone(timezone.utc) if times else None


# ---------------------------------------------------------------- strict forward performance
def strict_days(ret: pd.Series, targets: dict, reg: pd.DataFrame, rules_ok: bool, data_ok: bool,
                history_end: str, commit_time=git_commit_time, rules_version: str | None = None) -> pd.DataFrame:
    """Per day after history_end: which signal drives it and whether it counts as strict forward.
    Non-execution days are driven by the last executed target; an execution day mixes the old target
    (open -> VWAP) and the new one (VWAP -> close), so both must be valid."""
    days = ret.index[ret.index > pd.Timestamp(history_end)]
    if not len(days):
        return pd.DataFrame(columns=["signal", "strict", "reason"])
    allidx = ret.index
    exec_of = {s: allidx[allidx.searchsorted(s, side="right")] for s in targets
               if allidx.searchsorted(s, side="right") < len(allidx)}
    sig_by_exec = {e: s for s, e in exec_of.items()}
    row = {r.signal_date: r for r in reg.itertuples()
           if rules_version is None or r.rules_version == rules_version} if len(reg) else {}

    def ok(s):
        if s is None:
            return False, "no target"
        r = row.get(str(s.date()))
        if r is None:
            return False, f"{s.date()} not registered"
        if r.status != "valid_preregistered":
            return False, f"{s.date()} {r.status}"
        if r.target_sha != target_sha(targets[s]):
            return False, f"{s.date()} traded target differs from registration"
        ct = commit_time(r.target_sha)
        if ct is None or ct >= _utc(r.execution_bound):
            return False, f"{s.date()} no git timestamp before execution"
        return True, ""

    cur = max([s for s, e in exec_of.items() if e < days[0]], default=None)
    out = []
    for d in days:
        reasons = []
        if not rules_ok:
            reasons.append("rules differ from frozen version")
        if not data_ok:
            reasons.append("data coverage check failed")
        if d in sig_by_exec:                       # execution day: old target until VWAP, new one after
            s_new = sig_by_exec[d]
            for s in (cur, s_new):
                good, why = ok(s)
                if not good:
                    reasons.append(why)
            cur = s_new
        else:
            good, why = ok(cur)
            if not good:
                reasons.append(why)
        out.append({"date": d, "signal": None if cur is None else str(cur.date()), "strict": not reasons,
                    "reason": "; ".join(dict.fromkeys(reasons))})
    return pd.DataFrame(out).set_index("date")


def forward_summary(ret: pd.Series, bench: pd.Series, targets: dict, rules: dict, reg: pd.DataFrame,
                    diff: list, data_ok: bool, commit_time=git_commit_time) -> dict:
    sd = strict_days(ret, targets, reg, not diff, data_ok, rules["history_end"], commit_time, rules["version"])
    strict = sd[sd.strict.astype(bool)] if len(sd) else sd
    integrity = not reg.attrs.get("mismatch")
    out = {"rules_version": rules["version"], "frozen_at_utc": rules.get("frozen_at_utc"),
           "history_end": rules["history_end"], "rules_compliant": not diff, "rule_diff": diff,
           "registry_integrity": integrity, "mismatch": reg.attrs.get("mismatch", []), "data_ok": data_ok,
           "days_after_history": int(len(sd)), "strict_days": int(len(strict)),
           "excluded": sd[~sd.strict.astype(bool)].reason.value_counts().to_dict() if len(sd) else {},
           "registered": reg[REG_COLS].to_dict(orient="records") if len(reg) else []}
    if len(strict) == 0:
        out["performance_status"] = "尚未开始 / 无可验收的前瞻绩效"
        return out
    r = ret.reindex(strict.index)
    b = bench.reindex(strict.index)
    if b.isna().any():
        out["performance_status"] = "基准缺失, 不可评价"
        return out
    nav, bnav = (1 + r).prod(), (1 + b).prod()
    out.update({"performance_status": f"严格前瞻 {len(strict)} 个交易日", "start": str(strict.index[0].date()),
                "end": str(strict.index[-1].date()), "cum_return": nav - 1, "bench_cum_return": bnav - 1,
                "cum_excess": nav / bnav - 1, "dates": [d.strftime("%Y-%m-%d") for d in strict.index],
                "strat": [float(x) for x in r.values], "bench": [float(x) for x in b.values]})
    return out


def write_forward_md(fwd: dict) -> None:
    pct = lambda v: f"{v:.2%}"
    reg = pd.DataFrame(fwd["registered"])
    yes = lambda b: "是" if b else "否"
    lines = ["# 前瞻样本跟踪 (自动生成)", "",
             f"- 当前规则版本 **{fwd['rules_version']}**, 冻结于 {fwd['frozen_at_utc']} (UTC), 见 [rules/](rules/)。旧版本保留不改。",
             f"- 开发中看过的最后一个数据日是 **{fwd['history_end']}**: 此前的全部结果 (含 2022 年起的历史验证期) 只算历史回测。",
             "- 只有满足以下全部条件的交易日才算**严格前瞻**: 驱动当天收益的目标仓位在执行前 (下一工作日 09:30 北京时间) 已登记、"
             "登记时规则已冻结、登记行有早于执行时间的 git 提交时间、实际交易的目标与登记哈希一致、本次运行使用冻结规则、数据覆盖检查通过。"
             "执行日同时受新旧两期目标影响, 两期都必须有效。", "",
             "## 本次运行状态", "",
             f"| 项目 | 状态 |\n|---|---|\n| 规则与冻结版本一致 | {yes(fwd['rules_compliant'])}{'' if fwd['rules_compliant'] else ' (' + ', '.join(fwd['rule_diff']) + ')'} |\n"
             f"| 登记表完整 (交易的目标与登记一致) | {yes(fwd['registry_integrity'])}{'' if fwd['registry_integrity'] else ' (' + ', '.join(fwd['mismatch']) + ')'} |\n"
             f"| 数据覆盖检查 | {'通过' if fwd['data_ok'] else '未通过'} |\n"
             f"| 历史结束日之后的交易日 | {fwd['days_after_history']} |\n| 其中严格前瞻交易日 | {fwd['strict_days']} |\n"
             f"| 前瞻绩效 | **{fwd['performance_status']}** |", ""]
    if fwd.get("excluded"):
        lines += ["不计入严格前瞻的原因 (交易日数):", "", *[f"- {k}: {v}" for k, v in fwd["excluded"].items()], ""]
    if "cum_excess" in fwd:
        lines += [f"**严格前瞻区间** {fwd['start']} ~ {fwd['end']}: 组合 {pct(fwd['cum_return'])}, 基准 {pct(fwd['bench_cum_return'])}, "
                  f"累计超额 {pct(fwd['cum_excess'])}", ""]
    lines += ["## 登记表 (只追加)", "", reg.to_markdown(index=False) if len(reg) else "(无)", "",
              "`registry_v1_legacy.csv` 是 v1 的登记表, 原样保留: 其中 2026-10-30 的目标在 2026-11-15 登记, 晚于理论执行日 2026-11-02, "
              "所以只能算历史 / 演示, 不计入前瞻。", "",
              "## 每月操作流程", "",
              "1. 月末收盘后更新数据并运行 `bash run_all.sh` (使用冻结规则, 新目标自动登记);",
              "2. **在下一个交易日 09:30 之前** 提交并推送 `reports/forward/` (git 提交时间是独立时间戳);",
              "3. 之后的运行会核对登记时间、git 时间和哈希, 只有全部通过的交易日计入前瞻绩效。",
              "4. 修改规则或核心代码必须用 `scripts/freeze_rules.py --new-version --reason ...` 生成新版本, 旧版本和旧登记不改。"]
    (FWD / "README.md").write_text("\n".join(lines), encoding="utf-8")
