#!/usr/bin/env python3
"""Merge part16/rows/*.tsv fragments + SEED_ROWS.md into PART16_RESULTS.md. Idempotent."""
import glob, os, csv

ROOT = "<local data dir>/Desktop/release/edaic_rerun/part16"
HEAD = """# PART 16, running results

Started 2026-09-22. One row per value. `file` is the per-clip csv or source the value was computed from.
Bootstrap everywhere: 2000 draws, `numpy.random.default_rng(0)`, speakers resampled (never clips), percentile 2.5/97.5.
PAIRED = the speaker list is drawn once per replicate and both quantities recomputed inside that draw.
AUC recomputed with the rank formula before writing. Zero writes under `omni_final`.
Every value is recomputed by an independent verifier; a value is final only when both agree to 4 decimals.
"""

seed = [l.rstrip("\n") for l in open(os.path.join(ROOT, "SEED_ROWS.md")) if l.startswith("|")]

rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(ROOT, "rows", "*.tsv"))):
    for r in csv.reader(open(f), delimiter="\t"):
        if not r or r[0].strip().lower() == "id":
            continue
        r = (r + [""] * 8)[:8]
        if (r[0], r[1]) in seen:
            continue
        seen.add((r[0], r[1]))
        ci = f"[{r[3]}, {r[4]}]" if r[3] and r[4] and r[3].replace('-','').replace('.','').isdigit() else "NA"
        rows.append(f"| {r[0]} | {r[1]} | {r[2]} | {ci} | {r[5]} | {r[6]} | {r[7]} |")

out = [HEAD, "| id | what | value | 95% CI | n | n_spk | file |",
       "|----|------|-------|--------|---|-------|------|"] + seed + rows
open(os.path.join(ROOT, "PART16_RESULTS.md"), "w").write("\n".join(out) + "\n")
print(f"{len(seed)} seed + {len(rows)} fragment = {len(seed)+len(rows)} rows, "
      f"{len(glob.glob(os.path.join(ROOT,'rows','*.tsv')))} fragment files")
