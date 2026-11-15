"""Step 8: build the interactive HTML report (reports/report.html + a timestamped copy in reports/runs/)
from the data saved by step 6 and, if present, the robustness results of step 7."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf.config import PROC
from datetime import datetime

from mf.html_report import _git_commit, write_html_report

if __name__ == "__main__":
    payload = json.load(open(PROC / "report_payload.json"))
    rob = PROC / "robustness.json"
    payload["robustness"] = json.load(open(rob)) if rob.exists() else None
    payload["meta"].update(generated=datetime.now().strftime("%Y-%m-%d %H:%M"), commit=_git_commit())
    for p in write_html_report(payload):
        print("HTML report written to", p)
