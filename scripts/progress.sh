#!/usr/bin/env bash
# Live progress bar for the Sina daily-bar download (data/raw/sina); ~80 delisted stocks never arrive (Ctrl+C to quit).
cd "$(dirname "$0")/.."
TOTAL=$(tail -n +2 data/raw/universe_monthly.csv | cut -d, -f2 | sort -u | wc -l | tr -d ' ')
start_n=$(ls data/raw/sina | wc -l | tr -d ' '); start_t=$(date +%s)
while true; do
  n=$(ls data/raw/sina | wc -l | tr -d ' ')
  el=$(( $(date +%s) - start_t )); done_new=$(( n - start_n ))
  pct=$(( n * 100 / TOTAL )); fill=$(( pct / 2 ))
  bar=$(printf '%*s' "$fill" '' | tr ' ' '#')$(printf '%*s' $((50 - fill)) '' | tr ' ' '-')
  if [ $done_new -gt 0 ] && [ $el -gt 0 ]; then
    rate=$(( done_new * 60 / el )); eta=$(( (TOTAL - n) * el / done_new / 60 ))
    info="${rate}/min  ETA ~${eta} min"
  else info="measuring speed..."; fi
  rep=$(ls data/raw/reports 2>/dev/null | grep -c parquet); printf "\r[%s] %d/%d (%d%%)  %s  | reports %d/51   " "$bar" "$n" "$TOTAL" "$pct" "$info" "$rep"
  [ "$n" -ge "$TOTAL" ] && { echo; echo "download complete"; break; }
  sleep 5
done
