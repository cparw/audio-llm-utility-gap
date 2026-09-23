#!/usr/bin/env python3
import json, glob, os, csv
D = "<local data dir>/Desktop/release/edaic_rerun/part16/POD4"
all_ = []
for f in sorted(glob.glob(f"{D}/VERIFY_POD4*.json")):
    for o in json.load(open(f)):
        o["_src"] = os.path.basename(f); all_.append(o)
seen = {}
for o in all_: seen[o["id"]] = o
rows = list(seen.values())
rows.sort(key=lambda r: r["id"])
with open(f"{D}/VERIFY_POD4_SUMMARY.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["id", "value_run1_rank_formula", "value_run2_pairwise_MW", "agree_4dp", "lo", "hi", "usable_draws"])
    for r in rows:
        w.writerow([r["id"], f"{r['reported']:.4f}", f"{r['verify']:.4f}",
                    "YES" if r["match"] else "NO", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r.get("usable_draws", "")])
n = len(rows); ok = sum(1 for r in rows if r["match"])
print(f"VERIFY SUMMARY: {ok}/{n} agree to 4 decimals")
for r in rows:
    if not r["match"]: print("  DISAGREE", r["id"], r["reported"], r["verify"])
print(f"wrote {D}/VERIFY_POD4_SUMMARY.tsv")
