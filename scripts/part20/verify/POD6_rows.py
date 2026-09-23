#!/usr/bin/env python3
"""Append FINAL (agree_4dp == yes) rows from a POD6 check json to rows/POD6.tsv; skip ids already there.
usage: POD6_rows.py CHECK.json ROWS.tsv"""
import csv, json, os, sys
COLS = ["id", "what", "value_computed", "value_verified", "lo", "hi", "n", "n_spk", "file", "two_dp", "vs_half", "agree_4dp"]
rows = json.load(open(sys.argv[1])); tsv = sys.argv[2]
have = set()
if os.path.exists(tsv):
    have = {r["id"] for r in csv.DictReader(open(tsv), delimiter="\t")}
new = not os.path.exists(tsv)
with open(tsv, "a", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    if new: w.writerow(COLS)
    for r in rows:
        if r["agree_4dp"] != "yes":
            print("NOT FINAL (disagree):", r["id"], r["value_computed"], r["value_verified"]); continue
        if r["id"] in have: continue
        w.writerow([r[c] for c in COLS])
        print("\t".join(str(r[c]) for c in COLS))
