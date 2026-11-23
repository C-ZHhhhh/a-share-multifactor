"""Independent check of the published numbers (acceptance 3 and 5): recompute the headline statistics from the
daily returns embedded in reports/report.html and compare them with results.md and the holdings CSV.
Exits non-zero on any mismatch."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

problems = []
html = (ROOT / "reports/report.html").read_text(encoding="utf-8")
m = re.search(r"const D = (\{.*?\});\nconst NS", html, re.S)
D = json.loads(m.group(1).replace("<\\/", "</"))
M = D["meta"]
d = pd.DataFrame({k: D["daily"][k] for k in ("strat", "bench")}, index=pd.to_datetime(D["daily"]["dates"]))
if d.index.max() > pd.Timestamp(M["oos_end"]):
    problems.append("HTML history series extends past the OOS end (forward data mixed into history)")


def stats(r, b):
    n = len(r)
    nav, bnav = (1 + r).prod(), (1 + b).prod()
    a = r - b
    te = a.std(ddof=1) * np.sqrt(252)
    path = (1 + r).cumprod()
    mdd = (path / np.maximum(path.cummax(), 1.0) - 1).min()
    return {"超额年化": (nav / bnav) ** (252 / n) - 1, "信息比率": a.mean() * 252 / te, "跟踪误差": te,
            "年化收益": nav ** (252 / n) - 1, "最大回撤": mdd}


oos = d.loc[M["oos_start"]:M["oos_end"]]
s = stats(oos.strat, oos.bench)
perf = D["perf"][[k for k in D["perf"] if "OOS" in k][0]]
row = next(r for r in perf["rows"] if r[0] == M["main"])
val = dict(zip(perf["columns"], row))
parse = lambda x: float(x.strip("%")) / 100 if isinstance(x, str) and x.endswith("%") else float(x)
for k in ("超额年化", "信息比率", "跟踪误差", "年化收益", "最大回撤"):
    tol = 0.006 if k == "信息比率" else 6e-5
    if abs(parse(val[k]) - s[k]) > tol:
        problems.append(f"OOS {k}: table {val[k]} vs recomputed {s[k]:.4%}")

res_md = (ROOT / "reports/results.md").read_text(encoding="utf-8")
if f"| {M['main']}" not in res_md:
    problems.append("results.md does not list the main portfolio")

h = pd.read_csv(ROOT / "reports/holdings/optimized_monthly.csv")
ex = h[h.executed_weight.notna()]
tot = ex.groupby("signal_date").executed_weight.sum()
if not np.allclose(tot.values, 1.0, atol=1e-5):
    problems.append(f"holdings CSV: executed weights + cash != 1 on {int((abs(tot - 1) > 1e-5).sum())} dates")
pend = h[h.executed_weight.isna()]
if len(pend) and pend.signal_date.nunique() != 1:
    problems.append("holdings CSV: more than one signal date is pending execution")

F = D.get("forward")
if F and F.get("strict_days", 0) == 0 and "cum_excess" in F:
    problems.append("forward section reports performance with 0 strict forward days")

print("recomputed OOS:", {k: round(v, 6) for k, v in s.items()})
if problems:
    print("VERIFY FAILED:\n- " + "\n- ".join(problems))
    sys.exit(1)
print("verify_report: all checks passed")
