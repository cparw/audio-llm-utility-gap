#!/bin/bash
# PART 24 item 4: final check of the saved tex and its compiled PDF.
#   final_check.sh TEX PDF
#     TEX  the saved Overleaf tex, or "-" to skip Part 12 (PDF checks only)
#     PDF  the compiled PDF, or "-" (or a missing file) to skip the PDF checks; says the PDF is still missing
# Prints:
#   1. Part 12 on TEX: status counts, every MISMATCH row (tex line, sentence, hand triage from
#      triage_last_run.csv) and every UNMATCHED row (tex line, sentence, reason)
#   2. page count and the full text of page 5 (pdftotext)
#   3. smallest and largest font size in the PDF (PyMuPDF from the part24 venv), plus pdffonts embedding summary
# Writes: one folder per run under part24/part12/final_runs/ on the G-Drive (log, flagged csv, page5.txt,
#   fonts.json, Part 12 work dir copied with ditto, sidecar.json). Never writes under omni_final.
# Part 12 itself also refreshes value_index.csv and handcheck_values.csv in part12_prep/ on the Mac, as it always has.
set -uo pipefail
P24="<local data dir>/part24/part12"
PREP="<local data dir>/release/edaic_rerun/part17/part12_prep"
PY=/usr/local/bin/python3
VPY="$P24/venv/bin/python"
TEX="${1:?usage: final_check.sh TEX PDF   (TEX or PDF may be -)}"
PDF="${2:--}"

STAMP=$(date +%Y%m%d_%H%M%S)
if [ "$TEX" != "-" ]; then TAG="$(basename "$TEX" .tex)"; else TAG="pdfonly"; fi
OUTD="$P24/final_runs/${TAG}_${STAMP}"
if ! mkdir -p "$OUTD" 2>/dev/null; then
  OUTD="${TMPDIR:-/tmp}/part24_final_${TAG}_${STAMP}"; mkdir -p "$OUTD"
  echo "WARNING: G-Drive folder not writable; writing this run to $OUTD"
fi
case "$OUTD" in */omni_final/*) echo "refusing to write under omni_final" >&2; exit 2;; esac
LOG="$OUTD/final_check.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== final_check.sh  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "run folder: $OUTD"
P12_RC=na; PDF_STATE=missing; NP=na
TEX_SHA=na; PDF_SHA=na

# ------------------------------------------------------------------ 1. Part 12
if [ "$TEX" != "-" ]; then
  if [ ! -f "$TEX" ]; then echo "no such tex: $TEX"; exit 2; fi
  TEX_SHA=$(shasum -a 256 "$TEX" | cut -d' ' -f1)
  echo "tex: $TEX"
  echo "     mtime $(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$TEX")  size $(stat -f '%z' "$TEX") B  sha256 ${TEX_SHA:0:16}"
  cp "$TEX" "$OUTD/input_tex_copy.tex"
  echo
  echo "=== 1. Part 12 (run_part12.sh; takes a few minutes)"
  T0=$(date +%s)
  bash "$PREP/run_part12.sh" "$TEX" "$PREP/pasted_tex_dryrun.tex" "$OUTD/paper_vs_omni.csv" > "$OUTD/part12_stdout.log" 2>&1
  P12_RC=$?
  echo "run_part12.sh exit $P12_RC after $(( $(date +%s) - T0 )) s; full stdout in part12_stdout.log"
  grep -E '^(note:|work dir:)' "$OUTD/part12_stdout.log" | sed 's/^/  /'
  if [ "$P12_RC" -ne 0 ] || [ ! -f "$OUTD/paper_vs_omni.csv" ]; then
    echo "Part 12 FAILED; last lines:"; tail -n 25 "$OUTD/part12_stdout.log"
  else
    WORK=$(sed -n 's/^work dir: //p' "$OUTD/part12_stdout.log" | tail -1)
    [ -f "$WORK/summary.txt" ] && grep '^added basis' "$WORK/summary.txt" | sed 's/^/  /'
    "$PY" "$P24/report_part12.py" "$OUTD/paper_vs_omni.csv" "$P24/triage_last_run.csv" \
        --tex "$TEX" --out "$OUTD/flagged_rows_triaged.csv"
    [ -n "$WORK" ] && [ -d "$WORK" ] && ditto "$WORK" "$OUTD/part12_workdir"
  fi
else
  echo "tex: skipped (-)"
fi

# ------------------------------------------------------------------ 2 and 3. PDF
echo
if [ "$PDF" = "-" ] || [ ! -f "$PDF" ]; then
  echo "=== 2-3. PDF: STILL MISSING (${PDF}). Page 5 and font sizes not checked."
else
  PDF_STATE=present
  PDF_SHA=$(shasum -a 256 "$PDF" | cut -d' ' -f1)
  echo "=== 2. PDF: $PDF"
  echo "     mtime $(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$PDF")  size $(stat -f '%z' "$PDF") B  sha256 ${PDF_SHA:0:16}"
  NP=$(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages:/{print $2}')
  echo "page count: ${NP:-unknown}"
  if [ -n "$NP" ] && [ "$NP" -ge 5 ]; then
    pdftotext -f 5 -l 5 "$PDF" "$OUTD/page5.txt"
    echo "----- page 5 text (pdftotext, reading order) -----"
    cat "$OUTD/page5.txt"
    echo "----- end of page 5 ($(wc -w < "$OUTD/page5.txt" | tr -d ' ') words) -----"
    # ICASSP 2027 paper kit (as fetched 11 Sep 2026): the 5th page may hold only references,
    # acknowledgements and the ethics statement. Report any text above the first such heading.
    awk '/^([0-9]+\. *)?(Acknowledg|ACKNOWLEDG|Compliance [Ww]ith Ethical|COMPLIANCE WITH ETHICAL|REFERENCES|References)/{exit} {print}' "$OUTD/page5.txt" > "$OUTD/page5_before_allowed.txt"
    NB5=$(wc -w < "$OUTD/page5_before_allowed.txt" | tr -d ' ')
    if [ "$NB5" -gt 0 ]; then
      echo "PAGE 5 CHECK: $NB5 words sit above the first Acknowledgments / Ethics / References heading (body text on page 5):"
      sed 's/^/    | /' "$OUTD/page5_before_allowed.txt" | head -n 15
    else
      echo "PAGE 5 CHECK: page 5 starts with an Acknowledgments, Ethics or References heading"
    fi
  else
    echo "the PDF has fewer than 5 pages; no page 5"
  fi
  [ -n "$NP" ] && [ "$NP" -gt 5 ] && echo "NOTE: the PDF has $NP pages, more than 5"
  echo
  echo "=== 3. font sizes (PyMuPDF $("$VPY" -c 'import pymupdf;print(pymupdf.__version__)' 2>/dev/null))"
  "$VPY" "$P24/pdf_fonts.py" "$PDF" --json "$OUTD/fonts.json"
  pdffonts "$PDF" > "$OUTD/pdffonts.txt" 2>&1
  awk 'NR>2{n++; if($(NF-4)!="yes")ne++; if($0~/Type 3/)t3++} END{printf "pdffonts: %d fonts, %d not embedded, %d Type 3\n", n, ne+0, t3+0}' "$OUTD/pdffonts.txt"
fi

# ------------------------------------------------------------------ sidecar
"$PY" - "$OUTD" "$TEX" "$PDF" "$TEX_SHA" "$PDF_SHA" "$P12_RC" "$PDF_STATE" "${NP:-na}" <<'EOF'
import json, sys, platform, datetime, subprocess
o, tex, pdf, tsha, psha, rc, pst, np_ = sys.argv[1:9]
def v(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True).stderr.split("\n")[0] or ""
    except Exception: return ""
json.dump(dict(result="PART 24 item 4 final check", run_folder=o, tex=tex, tex_sha256=tsha, pdf=pdf, pdf_sha256=psha,
               part12_exit=rc, pdf_state=pst, pdf_pages=np_, command="final_check.sh " + tex + " " + pdf,
               seed="none (no sampling here; Part 12 reads saved values)", gpu="none (Mac CPU)", model_id="none",
               n="see paper_vs_omni.csv", n_speakers="n/a", bootstrap="n/a (Part 12 compares saved values)",
               versions=dict(python=platform.python_version(), pdftotext=v(["pdftotext", "-v"])),
               date=datetime.datetime.now().isoformat(timespec="seconds")),
          open(o + "/final_check.sidecar.json", "w"), indent=1)
EOF
echo
echo "=== done. log: $LOG"
