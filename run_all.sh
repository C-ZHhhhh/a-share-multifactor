#!/usr/bin/env bash
# Reproduce the full research pipeline end to end.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY -m pytest -q tests              # unit + synthetic end-to-end tests (no downloads)
$PY scripts/01_fetch_data.py          # incremental download up to config data.end; writes data/raw/manifest.json; fails if coverage is incomplete
$PY scripts/02_build_panel_audit.py   # clean panel + data-quality audit
$PY scripts/03_factors.py             # factors, preprocessing, single-factor tests, selection (IS)
$PY scripts/04_models.py              # walk-forward ICIR / Ridge / GBDT models
$PY scripts/05_portfolio_backtest.py  # risk model, optimisation, backtests
$PY scripts/06_attribution_report.py  # IS/OOS evaluation, attribution + reconciliation, constraint check, holdings, results.md
$PY scripts/07_robustness.py          # costs, parameters, constraints, splits, data sensitivity, bootstrap, capacity
$PY scripts/08_html_report.py         # interactive visual report -> reports/report.html
$PY scripts/verify_report.py          # independent recomputation of the published numbers; fails on mismatch
echo "Done. Open reports/report.html in a browser for the visual report (a timestamped copy is in reports/runs/)."
