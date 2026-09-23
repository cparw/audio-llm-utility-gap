#!/usr/bin/env python3
"""Verifier rows writer for PART 22.
  rows/P22_verified.tsv : verifier-only file (id, what, value_computed, value_verified, lo, hi, n, n_spk, file, agree_4dp);
                          no other job writes it.
  rows/P22.tsv          : shared fragment. Rewritten with the UNION of all columns already present plus the verifier
                          columns, so no other job's column is ever dropped. For verifier rows, `value` = value_verified.
usage: python3 P22_rows.py rows.json   (list of dicts; same id replaces the earlier verifier row)
"""
import csv, json, os, sys

R = "scores/part22/rows"
VER = f"{R}/P22_verified.tsv"
SHARED = f"{R}/P22.tsv"
VCOLS = ["id", "what", "value_computed", "value_verified", "lo", "hi", "n", "n_spk", "file", "agree_4dp"]


def read(p):
    if not os.path.exists(p):
        return [], []
    with open(p, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        return list(rd), list(rd.fieldnames or [])


def write(p, rows, cols):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in cols})


new = json.load(open(sys.argv[1]))
ids = {r["id"] for r in new}
old, _ = read(VER)
write(VER, [r for r in old if r["id"] not in ids] + new, VCOLS)

sh, shcols = read(SHARED)
cols = list(shcols) if shcols else []
for c in ["id", "what", "value"] + VCOLS:
    if c not in cols:
        cols.append(c)
by_id = {r["id"]: i for i, r in enumerate(sh)}
for r in new:
    r2 = dict(r)
    r2.setdefault("value", r.get("value_verified", ""))
    if r["id"] in by_id:
        # a row another job owns: keep its what/value/file, add the verifier columns only
        base = sh[by_id[r["id"]]]
        for k in ("value_computed", "value_verified", "agree_4dp"):
            base[k] = r2.get(k, "")
        for k, v in r2.items():
            if not base.get(k):
                base[k] = v
    else:
        sh.append(r2)
write(SHARED, sh, cols)
print(f"{VER}: {len(read(VER)[0])} rows | {SHARED}: {len(sh)} rows, columns {cols}")
