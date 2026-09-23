#!/bin/bash
# Part 12: tex vs result files, one command.
#   run_part12.sh REAL_TEX [BASELINE_TEX] [OUT_CSV]
# REAL_TEX      the saved Overleaf tex (e.g. <local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex)
# BASELINE_TEX  what "added" is measured against. Default: pasted_tex_dryrun.tex (the tex pasted on
#               2026-09-23 05:58 UTC = 22:58 PDT 22 Sep), so "added" = sentences written after the paste.
#               If REAL_TEX is identical to the paste, the Sep 14 tex is used instead (and the summary says so).
# OUT_CSV       default scores/part17/paper_vs_omni.csv
# Steps: handcheck -> value_index -> B extract -> F diff -> C match -> E red spans -> paper_vs_omni.csv
# Writes only under part12_prep/ and the OUT_CSV path. Never writes under omni_final.
set -euo pipefail
PY=/usr/local/bin/python3
PREP="<local data dir>/release/edaic_rerun/part17/part12_prep"
SEP14="<local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex"
TEX="${1:?usage: run_part12.sh REAL_TEX [BASELINE_TEX] [OUT_CSV]}"
BASE="${2:-$PREP/pasted_tex_dryrun.tex}"
OUT="${3:-scores/part17/paper_vs_omni.csv}"
case "$OUT" in */omni_final/*) echo "refusing to write under omni_final" >&2; exit 2;; esac
[ -f "$TEX" ] || { echo "no such tex: $TEX" >&2; exit 2; }

body() { sed -n '/^[[:space:]]*\\documentclass/,$p' "$1"; }
BASIS="sentences in $(basename "$TEX") not in $(basename "$BASE")"
if [ "$(body "$TEX" | shasum | cut -c1-40)" = "$(body "$BASE" | shasum | cut -c1-40)" ]; then
  echo "note: $TEX is identical to the baseline; using the Sep 14 tex as baseline (superset of 23 Sep additions)"
  BASE="$SEP14"
  BASIS="added since Sep 14 tex (superset of the 23 Sep additions; the input equals the pasted tex)"
fi
STAMP=$(date +%Y%m%d_%H%M%S)
WORK="$PREP/runs/$(basename "$TEX" .tex)_$STAMP"
mkdir -p "$WORK"
echo "work dir: $WORK"
echo "tex: $TEX ($(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$TEX"))  baseline: $BASE"

cd "$PREP"
$PY part12_handcheck.py
$PY build_value_index.py
$PY extract_tex_numbers.py "$TEX" --out "$WORK/numbers.csv"
$PY diff_tex.py "$TEX" "$BASE" --out "$WORK/added_vs_baseline.csv"
if [ "$BASE" != "$SEP14" ] && [ -f "$SEP14" ]; then
  $PY diff_tex.py "$TEX" "$SEP14" --out "$WORK/added_vs_sep14.csv"
fi
$PY match_tex.py "$WORK/numbers.csv" "$PREP/value_index.csv" --tex "$TEX" --added "$WORK/added_vs_baseline.csv" \
    --red-spec "$PREP/red_spans_spec.csv" --out "$WORK/matches.csv"
$PY red_span_check.py "$WORK/numbers.csv" "$WORK/matches.csv" "$PREP/value_index.csv" --spec "$PREP/red_spans_spec.csv" \
    --out "$WORK/red_spans.csv"
$PY assemble_part12.py "$WORK/matches.csv" "$WORK/red_spans.csv" --tex "$TEX" --added-basis "$BASIS" \
    --out "$OUT" --summary "$WORK/summary.txt"
cp "$PREP/value_index.csv" "$WORK/value_index_snapshot.csv"
cp "$OUT" "$WORK/paper_vs_omni.csv"
echo "done: $OUT ; summary $WORK/summary.txt ; red spans $WORK/red_spans.csv"
