"""One-off: freeze the model and evaluation rules after the last development change (writes
reports/forward/frozen_rules.json, never overwrites it). Everything up to the last data day seen now is
the 'historical, repeatedly tested' sample; only returns after it count as the forward sample."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.config import PROC
from mf.forward import freeze
from mf.html_report import _git_commit

if __name__ == "__main__":
    bt = pickle.load(open(PROC / "backtests.pkl", "rb"))
    model = json.load(open(PROC / "model_choice.json"))["best_model"]
    history_end = str(bt["results"][f"组合优化({bt['best']})"]["ret"].index[-1].date())
    rules = freeze(model, bt["lam"], history_end, _git_commit())
    print(json.dumps(rules, ensure_ascii=False, indent=2))
