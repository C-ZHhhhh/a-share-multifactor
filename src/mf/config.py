from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()
RAW = ROOT / CFG["data"]["raw_dir"]
PROC = ROOT / CFG["data"]["processed_dir"]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
for p in (RAW, PROC, FIG):
    p.mkdir(parents=True, exist_ok=True)
