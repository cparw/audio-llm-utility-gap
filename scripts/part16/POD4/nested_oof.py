#!/usr/bin/env python3
"""Nested layer-selection probe that SAVES the per-clip out-of-fold predictions.

FOLDS: no saved fold file exists for the Pitt 468 (logged MEDIUM in
DISCREPANCIES.md), so outer folds are sklearn GroupKFold(n_splits=5) grouped by
speaker over the clips in the source csv's own order -- deterministic, and
therefore IDENTICAL across arms, which is what makes the paired contrast exact.
Inner layer choice: GroupKFold(4) on the training speakers.

usage: nested_oof.py STATES.npz OUTPREFIX [KEYS=enc]
writes OUTPREFIX_{key}_nested_oof.csv and OUTPREFIX_nested.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, sys, json, csv, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

NPZ, OUTP = sys.argv[1:3]
KEYS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["enc"])
z = np.load(NPZ, allow_pickle=True)
y = z["label"].astype(int); spk = z["spk"].astype(str); name = z["name"].astype(str)
NJ = max(2, (os.cpu_count() or 8) - 4)

def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    rr = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        rr[i:j + 1] = rr[i:j + 1].mean(); i = j + 1
    r = np.empty(len(s)); r[o] = rr
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

def inner(X, tr, l, g):
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups=g[tr]):
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4

# group ids in the csv's OWN order: first appearance order of each speaker
seen = {}
for s in spk:
    if s not in seen: seen[s] = len(seen)
g = np.array([seen[s] for s in spk])

res = {}
for key in KEYS:
    X = z[key].astype(np.float32); nl = X.shape[1]
    oof = np.zeros(len(y)); chosen = []; foldid = np.zeros(len(y), int)
    for fi, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
        best = 0 if nl == 1 else int(np.argmax(Parallel(n_jobs=min(nl, NJ))(delayed(inner)(X, tr, l, g) for l in range(nl))))
        chosen.append(best)
        oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best])
        foldid[te] = fi
    a = float(auc_rank(y, oof))
    res[key] = {"auc_nested_oof": round(a, 6), "chosen_layers": chosen, "n": int(len(y)),
                "n_speakers": int(len(set(spk))), "n_points": nl,
                "fold_source": "GroupKFold(5) by speaker, csv order (no saved fold file found)"}
    with open(f"{OUTP}_{key}_nested_oof.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "speaker", "label", "p_probe", "fold"])
        for i in range(len(y)):
            w.writerow([name[i], spk[i], int(y[i]), float(oof[i]), int(foldid[i])])
    print(f"{key}: nested OOF AUC {a:.4f}  layers {chosen}  -> {OUTP}_{key}_nested_oof.csv", flush=True)

json.dump(res, open(f"{OUTP}_nested.json", "w"), indent=1)
print("NESTED OOF DONE", flush=True)
