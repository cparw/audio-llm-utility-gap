"""Verify every interviewer-free clip against manifest sha256, build the scoring manifest.
arm = 'set' of pitt_conflict_manifest.csv keyed by basename(segment_path) == orig_clip_id."""
import csv, hashlib, os, sys
D = "/workspace/cut"
man = list(csv.DictReader(open(f"{D}/manifest.csv")))
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open("/workspace/refs/pitt_conflict_manifest.csv"))}
bad = []
with open(f"{D}/score_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip_id", "orig_clip_id", "path", "label", "speaker", "arm", "dur_par_s"])
    for r in man:
        p = f"{D}/clips/{r['clip']}"
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != r["sha256"]: bad.append(r["clip"])
        w.writerow([r["clip"], r["orig_clip_id"], p, r["label"], r["spk"], ARM[r["orig_clip_id"]], r["dur_par_s"]])
print(f"clips {len(man)} sha256_ok {len(man)-len(bad)} bad {len(bad)} {bad[:5]}")
from collections import Counter
print("arms", Counter(ARM[r["orig_clip_id"]] for r in man), "labels", Counter(r["label"] for r in man), "speakers", len({r["spk"] for r in man}))
sys.exit(1 if bad or len(man) != 468 else 0)
