"""Fold check on this machine: the split GroupKFold(5) gives here, compared clip by clip with the released
pod-order and Mac-order fold files; and whether numpy's default argsort equals the stable argsort on the
per-speaker clip counts (the tie order GroupKFold depends on).  usage: t5_foldcheck.py JOB STATES.npz FOLDDIR DS OUTDIR"""
import sys, json, csv, numpy as np
from sklearn.model_selection import GroupKFold
job, npz, fdir, ds, outd = sys.argv[1:6]
z = np.load(npz, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); names = [str(v) for v in z["name"]]
native = np.full(len(y), -1)
for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, groups=spk)):
    native[te] = f
uniq, gidx = np.unique(spk, return_inverse=True); counts = np.bincount(gidx)
a_def = np.argsort(counts); a_stab = np.argsort(counts, kind="stable")
res = dict(job=job, ds=ds, n=len(y), n_groups=int(len(uniq)), argsort_default_equals_stable=bool(np.array_equal(a_def, a_stab)),
           argsort_default_first20=a_def[:20].tolist(), argsort_stable_first20=a_stab[:20].tolist())
with open(f"{outd}/{job}_folds_native.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip_id", "speaker_id", "fold"]); w.writerows(zip(names, spk, native.tolist()))
for tag in ("pod", "mac"):
    fp = f"{fdir}/{ds}_groupkfold5_{tag}.csv"
    try:
        rows = list(csv.DictReader(open(fp)))
    except FileNotFoundError:
        res[f"vs_{tag}"] = "no file"; continue
    m = {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in rows}
    missing = [n for n in names if n not in m]
    same = [m[n][1] == native[i] for i, n in enumerate(names) if n in m]
    spk_same = [m[n][0] == spk[i] for i, n in enumerate(names) if n in m]
    res[f"vs_{tag}"] = dict(file=fp, n_file=len(rows), missing=len(missing), fold_equal=int(sum(same)), speaker_equal=int(sum(spk_same)))
json.dump(res, open(f"{outd}/{job}_fold_check.json", "w"), indent=1)
print("FOLDCHECK", json.dumps(res))
