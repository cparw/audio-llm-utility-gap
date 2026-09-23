"""Part 20 POD3: GroupKFold(5) by speaker over the 966 rows of part14_manifest_new.csv, POD tie rule
(numpy.argsort(counts, kind="stable"), reversed as GroupKFold reverses it). Also writes the training manifest."""
import csv, os, sys, numpy as np
MAN = "<local data dir>/release/edaic_rerun/part14_manifest_new.csv"
CLIPS = "<local data dir>/release/edaic_rerun/part14_clips"
OUTD = "scores/part20/POD3"
os.makedirs(OUTD, exist_ok=True)
rows = list(csv.DictReader(open(MAN)))
spk = np.array([str(r["speaker_id"]) for r in rows])
uniq, gidx = np.unique(spk, return_inverse=True)
cnt = np.bincount(gidx)
order = np.argsort(cnt, kind="stable")[::-1]
per_fold = np.zeros(5); g2f = np.zeros(len(uniq), dtype=int)
for gi in order:
    f = int(np.argmin(per_fold)); per_fold[f] += cnt[gi]; g2f[gi] = f
fold = g2f[gidx]
with open(f"{OUTD}/part14_groupkfold5_pod.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip_id", "speaker_id", "fold"])
    for r, f in zip(rows, fold): w.writerow([r["seg_uid"], r["speaker_id"], int(f)])
miss = 0
with open(f"{OUTD}/p14_ft966_manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker", "set", "seg_uid", "pair_id", "fold"])
    for r, f in zip(rows, fold):
        name = f"{r['set']}_{r['speaker_id']}_{r['seg_uid']}.wav"
        if not os.path.exists(f"{CLIPS}/{name}"): miss += 1
        w.writerow([f"/workspace/p20/part14_clips/{name}", r["label"], r["speaker_id"], r["set"], r["seg_uid"], r["pair_id"], int(f)])
print("rows", len(rows), "speakers", len(uniq), "missing clips", miss, "fold sizes", np.bincount(fold).tolist(),
      "speakers per fold", [int((g2f == k).sum()) for k in range(5)])
for s in ("conflict", "agreement"):
    m = np.array([r["set"] == s for r in rows])
    print(s, m.sum(), "per fold", np.bincount(fold[m], minlength=5).tolist(), "pos", int(sum(int(r["label"]) for r, k in zip(rows, m) if k)))
# conflict and agreement of one speaker same fold (by construction; check)
d = {}
for r, f in zip(rows, fold): d.setdefault(r["speaker_id"], set()).add(int(f))
print("speakers in >1 fold:", sum(len(v) > 1 for v in d.values()))
ag = [r for r in rows if r["set"] == "agreement"]
print("agreement distinct seg_uid", len(set(r["seg_uid"] for r in ag)), "max repeat", max(__import__("collections").Counter(r["seg_uid"] for r in ag).values()))
