#!/bin/bash
# PART 24 item 4: wait for the final tex (and its PDF) in ~/Desktop/paper1_final/, then run final_check.sh.
#   wait_final_tex.sh            (start it in the background)
# Every 2 minutes:
#   - the tex counts as new when its mtime differs from OLD_MTIME (the Sep 23 06:38:06 save)
#   - a PDF counts when it is any *.pdf in ~/Desktop/paper1_final/ newer than OLD_MTIME and saved
#     within 10 minutes of the new tex (before or after it); the newest such PDF is used
# Then:
#   tex + PDF          -> final_check.sh TEX PDF, exit 0
#   tex, no PDF yet    -> final_check.sh TEX - once (Part 12 only, prints that the PDF is still missing),
#                         keep polling until 10 minutes after the tex mtime (+ one poll);
#                         a PDF in that window -> final_check.sh - PDF (PDF part only), exit 0;
#                         none                -> exit 3 with "PDF still missing"
#   tex saved again while waiting -> start over on the newer tex
# Gives up after MAX_HOURS (default 72) with exit 4. Mac only, no pods, no money.
# Env overrides: TEX, DIR, OLD_MTIME, POLL_S, WINDOW_S, MAX_HOURS, CHECK, WLOG (tests only).
set -u
P24="<local data dir>/part24/part12"
DIR="${DIR:-$HOME/Desktop/paper1_final}"
TEX="${TEX:-$DIR/AudioLLMHealthUtilityGap.tex}"
OLD_MTIME="${OLD_MTIME:-1790170686}"        # 2026-09-23 06:38:06 PDT, the tex Part 12 last ran on
POLL_S="${POLL_S:-120}"
WINDOW_S="${WINDOW_S:-600}"
MAX_HOURS="${MAX_HOURS:-72}"
CHECK="${CHECK:-$P24/final_check.sh}"
WLOG="${WLOG:-$P24/wait_final_tex.log}"
[ -w "$P24" ] || WLOG="${TMPDIR:-/tmp}/wait_final_tex.log"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WLOG"; }
mt() { stat -f '%m' "$1" 2>/dev/null || echo 0; }
hm() { date -r "$1" '+%Y-%m-%d %H:%M:%S'; }

# newest *.pdf in DIR with mtime > OLD_MTIME and |mtime - $1| <= WINDOW_S
find_pdf() {
  local texm="$1" best="" bestm=0 f m d
  for f in "$DIR"/*.pdf "$DIR"/*.PDF; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in ._*) continue;; esac
    m=$(mt "$f")
    [ "$m" -gt "$OLD_MTIME" ] || continue
    d=$(( m - texm )); [ "$d" -lt 0 ] && d=$(( -d ))
    [ "$d" -le "$WINDOW_S" ] || continue
    if [ "$m" -gt "$bestm" ]; then best="$f"; bestm="$m"; fi
  done
  echo "$best"
}

# wait until the file stops changing (Overleaf download or editor save in progress)
settle() {
  local a b
  a="$(stat -f '%m %z' "$1" 2>/dev/null)"; sleep 10; b="$(stat -f '%m %z' "$1" 2>/dev/null)"
  while [ "$a" != "$b" ]; do a="$b"; sleep 10; b="$(stat -f '%m %z' "$1" 2>/dev/null)"; done
}

[ -x "$CHECK" ] || { say "final_check.sh missing or not executable: $CHECK"; exit 2; }
say "waiting for a new $TEX (old mtime $(hm "$OLD_MTIME")) and a PDF in $DIR within ${WINDOW_S} s of it; poll ${POLL_S} s; give up after ${MAX_HOURS} h"
START=$(date +%s)
P12_DONE_FOR=""        # tex mtime that Part 12 has already run on
while :; do
  NOW=$(date +%s)
  if [ $(( NOW - START )) -gt $(( MAX_HOURS * 3600 )) ]; then
    say "gave up after ${MAX_HOURS} h; tex mtime still $(hm "$(mt "$TEX")")"; exit 4
  fi
  TM=$(mt "$TEX")
  if [ "$TM" != "0" ] && [ "$TM" != "$OLD_MTIME" ]; then
    settle "$TEX"; TM=$(mt "$TEX")
    PDF=$(find_pdf "$TM")
    if [ -n "$PDF" ]; then
      settle "$PDF"
      if [ "$P12_DONE_FOR" = "$TM" ]; then
        say "PDF arrived for the tex already checked: $PDF ($(hm "$(mt "$PDF")")); running the PDF part"
        bash "$CHECK" - "$PDF"; RC=$?
      else
        say "new tex ($(hm "$TM")) and PDF $PDF ($(hm "$(mt "$PDF")")); running final_check.sh"
        bash "$CHECK" "$TEX" "$PDF"; RC=$?
      fi
      say "final_check.sh exit $RC; done"; exit 0
    fi
    if [ "$P12_DONE_FOR" != "$TM" ]; then
      say "new tex ($(hm "$TM")) but no PDF within ${WINDOW_S} s of it yet; running Part 12 now. PDF STILL MISSING."
      bash "$CHECK" "$TEX" -; RC=$?
      say "Part 12 part exit $RC; still waiting for a PDF saved by $(hm $(( TM + WINDOW_S )))"
      P12_DONE_FOR="$TM"
    fi
    if [ "$(date +%s)" -gt $(( TM + WINDOW_S + POLL_S )) ]; then
      say "no PDF saved within ${WINDOW_S} s of the tex ($(hm "$TM")); PDF STILL MISSING. Part 12 output is above. Exiting."
      exit 3
    fi
  fi
  sleep "$POLL_S"
done
