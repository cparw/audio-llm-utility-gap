#!/usr/bin/env python3
"""PART 20 POD5 verifier: fold-use check for the nested probes.

Written from scratch; does not import or exec any compute-job script.
usage: POD5_nested_refit.py STATES.npz SUMMARY.json FOLDS.json REFDIR OUT.json

1. Every outer fold array in FOLDS.json is compared clip by clip with the release fold file it names
   (read from REFDIR, the verifier's own copy of the release folds).
2. For every stream x fold set in SUMMARY.json, each outer fold is refitted at the layer the run chose
   (StandardScaler + LogisticRegression(max_iter=2000, class_weight=balanced), float32 features) using
   the RELEASE fold file, and the out-of-fold predictions are compared with the run's oof csv column.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, json, csv
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NPZ, SUM, FJ, REF, OUT = sys.argv[1:6]
z = np.load(NPZ, allow_pickle=True)
names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
S = json.load(open(SUM)); F = json.load(open(FJ))
res = {"fold_json_vs_release": {}, "refit": {}}
fpos = {c: i for i, c in enumerate(F["clip_id"])}
for sname, sd in F["sets"].items():
    rel = os.path.join(REF, os.path.basename(sd["file"]))
    fm = {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in csv.DictReader(open(rel))}
    mism = sum(1 for c, f, s in zip(F["clip_id"], sd["outer"], F["speaker_id"]) if fm[c] != (s, int(f)))
    res["fold_json_vs_release"][sname] = {"file": os.path.basename(sd["file"]), "mismatch": mism, "n": len(F["clip_id"])}


yv = y.copy()
def fit(X, tr, te):
    sc = StandardScaler().fit(X[tr])
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), yv[tr]).predict_proba(sc.transform(X[te]))[:, 1]


jobs = []
for tag, r in S["results"].items():
    key = r["stream"]
    for sname, ffile in zip(r["fold_sets"], r["fold_files"]):
        rel = os.path.join(REF, os.path.basename(ffile))
        fm = {q["clip_id"]: int(q["fold"]) for q in csv.DictReader(open(rel))}
        fold = np.array([fm[n] for n in names])
        for k in range(5):
            jobs.append((tag, key, sname, k, int(r["chosen_layers"][sname][k]), fold))
Xc = {key: np.asarray(z[key], dtype=np.float32) for key in set(j[1] for j in jobs)}
def run(tag, key, sname, k, l, fold):
    X = Xc[key][:, l, :]
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    return tag, sname, te, fit(X, tr, te)
out = Parallel(n_jobs=int(os.environ.get("NJV", "60")))(delayed(run)(*j) for j in jobs)
pred = {}
for tag, sname, te, p in out:
    pred.setdefault((tag, sname), np.full(len(y), np.nan))[te] = p
for tag, r in S["results"].items():
    rows = list(csv.DictReader(open(r["oof_csv"])))
    rid = [q["clip_id"] for q in rows]; assert rid == list(names), "row order differs from states"
    for sname in r["fold_sets"]:
        col = np.array([float(q[f"p_{sname}"]) for q in rows])
        mine = pred[(tag, sname)]
        res["refit"][f"{tag}/{sname}"] = {"max_abs_diff": float(np.nanmax(np.abs(col - mine))),
                                         "mean_abs_diff": float(np.nanmean(np.abs(col - mine))),
                                         "layers": r["chosen_layers"][sname]}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
