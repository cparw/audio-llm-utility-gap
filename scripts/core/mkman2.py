"""Manifest (path,label,speaker) for the pod from any per-clip CSV: resolve basenames under ROOT. usage: mkman2.py SRC ROOT OUT"""
import csv, os, sys, collections
SRC, ROOT, OUT = sys.argv[1:4]
idx = collections.defaultdict(list)
for d, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(".wav"): idx[f].append(os.path.join(d, f))
rows = list(csv.DictReader(open(SRC)))
pcol = next(c for c in ("path", "filepath", "clip_path") if c in rows[0]); scol = next(c for c in ("speaker", "speaker_id", "spk") if c in rows[0])
out, miss, dup = [], [], []
for r in rows:
    b = os.path.basename(r[pcol]); c = idx.get(b, [])
    if not c: miss.append(b); continue
    if len(c) > 1: dup.append(b)
    out.append((c[0], r["label"], r[scol]))
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"]); w.writerows(out)
print(f"{OUT}: {len(out)} resolved, {len(miss)} missing {miss[:3]}, {len(dup)} ambiguous {dup[:3]}; speakers {len(set(o[2] for o in out))}")
