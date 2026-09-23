#!/bin/bash
# PART 24 item 3. Wait for both rater columns, then run rater_stats.py and its verifier.
#
# Every 5 minutes:
#   1. counts non empty cells in rater1 and rater2 of the local sheet
#      <local data dir>/release/edaic_rerun/rater_sheet_heard80.csv
#   2. if POLL_LIVE=1 (default), also pulls a read only csv export of the live Google Sheet
#      <rater sheet id> with rclone (remote gdrive:) and counts it
# When a source has 80 non empty cells in both columns (local checked first), it
#   saves a dated copy of that source here (ditto), runs rater_stats.py on it, then
#   verify_rater_stats.py on the output. PASS -> writes DONE and exits.
#   A failed run (unreadable cell, check failure, verifier FAIL) is logged, that file
#   version is not retried, and polling goes on, so a fixed sheet runs by itself.
# Stops by itself after MAX_HOURS (default 168). One copy at a time (lock dir).
# Plain bash 3.2, no timeout command, no associative arrays.
#
# Start:                nohup /bin/bash "<this file>" >/dev/null 2>&1 &
# Options by env:        QUESTION=auto|away|depressed  POLL_LIVE=0|1  MAX_HOURS=N  POLL_S=300
#                        P24_OUT / P24_LOCAL override the folder and the sheet (tests only)
# Watch:                 cat "<this dir>/WAIT_STATUS.txt"; tail "<this dir>/wait_and_run.log"

OUT="${P24_OUT:-<local data dir>/part24/rater}"
LOCAL="${P24_LOCAL:-<local data dir>/release/edaic_rerun/rater_sheet_heard80.csv}"
SHEET_ID="<rater sheet id>"
PY=/usr/local/bin/python3
RCLONE=/opt/homebrew/bin/rclone
QUESTION="${QUESTION:-auto}"
POLL_LIVE="${POLL_LIVE:-1}"
MAX_HOURS="${MAX_HOURS:-168}"
POLL_S="${POLL_S:-300}"
N=80

LOG="$OUT/wait_and_run.log"
STATUS="$OUT/WAIT_STATUS.txt"
TRIED="$OUT/.tried_hashes"
LOCK="$OUT/.wait_lock"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" >> "$LOG"; }

if [ ! -d "$OUT" ]; then
  echo "$(ts) G-Drive folder not found: $OUT" >> "$HOME/part24_rater_wait_error.log"
  exit 1
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  OLDPID=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
    echo "$(ts) another wait_and_run.sh is running (pid $OLDPID); exiting" >> "$LOG"
    exit 0
  fi
  echo "$(ts) stale lock from pid $OLDPID removed" >> "$LOG"
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
TMPD=$(mktemp -d -t part24rater)
trap 'rm -rf "$LOCK" "$TMPD"' EXIT
touch "$TRIED"

# prints "n_rater1 n_rater2 n_rows" (NA when a column is missing)
count() {
  "$PY" - "$1" <<'PYEOF'
import sys
import pandas as pd
try:
    d = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
except Exception:
    print("NA NA NA"); sys.exit(0)
cols = {c.strip().lower().replace(" ", "").replace("_", ""): c for c in d.columns}
out = []
for r in ("rater1", "rater2"):
    c = cols.get(r)
    out.append(str(int((d[c].str.strip() != "").sum())) if c else "NA")
out.append(str(len(d)))
print(" ".join(out))
PYEOF
}

# run stats + verifier on a saved copy; returns 0 only on a verified run
run_on() {
  src="$1"; label="$2"
  h=$(shasum -a 256 "$src" | cut -d' ' -f1)
  if grep -q "$h" "$TRIED"; then return 1; fi
  echo "$h $label $(ts)" >> "$TRIED"
  stamp=$(date -u +%Y%m%dT%H%MZ)
  copy="$OUT/rater_sheet_${label}_${stamp}.csv"
  ditto "$src" "$copy"
  log "RUN on $label, saved copy $copy (sha256 $h), QUESTION=$QUESTION"
  "$PY" "$OUT/rater_stats.py" --sheet "$copy" --question "$QUESTION" --out "$OUT" > "$OUT/rater_stats_run.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "rater_stats.py exit $rc on $copy, see rater_stats_run.log; polling continues"
    return 1
  fi
  "$PY" "$OUT/verify_rater_stats.py" --stats "$OUT/rater_stats.csv" > "$OUT/verify_rater_stats.log" 2>&1
  vrc=$?
  if [ $vrc -ne 0 ]; then
    log "VERIFY FAIL (exit $vrc), see verify_rater_stats.log; polling continues"
    return 1
  fi
  {
    echo "DONE $(ts) source $label copy $copy"
    tail -1 "$OUT/verify_rater_stats.log"
    cat "$OUT/rater_stats_paste.txt"
  } > "$OUT/DONE"
  log "DONE, verified. Paste lines in $OUT/rater_stats_paste.txt"
  return 0
}

START=$(date +%s)
log "start pid $$ poll ${POLL_S}s live=$POLL_LIVE question=$QUESTION max ${MAX_HOURS}h"
while :; do
  set -- $(count "$LOCAL"); L1=$1; L2=$2
  V1=NA; V2=NA
  LIVEF=""
  if [ "$POLL_LIVE" = "1" ]; then
    rm -f "$TMPD"/*.csv
    "$RCLONE" backend copyid gdrive: "$SHEET_ID" "$TMPD/" --drive-export-formats csv \
      --contimeout 30s --timeout 120s >> "$LOG" 2>&1
    LIVEF=$(ls "$TMPD"/*.csv 2>/dev/null | head -1)
    if [ -n "$LIVEF" ]; then set -- $(count "$LIVEF"); V1=$1; V2=$2; fi
  fi
  echo "$(ts) local rater1=$L1 rater2=$L2 | live rater1=$V1 rater2=$V2 | pid $$" > "$STATUS"

  if [ "$L1" = "$N" ] && [ "$L2" = "$N" ]; then
    if run_on "$LOCAL" local; then exit 0; fi
  fi
  if [ -n "$LIVEF" ] && [ "$V1" = "$N" ] && [ "$V2" = "$N" ]; then
    if run_on "$LIVEF" live; then exit 0; fi
  fi

  NOW=$(date +%s)
  if [ $(( (NOW - START) / 3600 )) -ge "$MAX_HOURS" ]; then
    log "MAX_HOURS reached, exiting without a run"
    exit 0
  fi
  sleep "$POLL_S"
done
