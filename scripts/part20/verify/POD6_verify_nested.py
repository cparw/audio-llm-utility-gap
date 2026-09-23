#!/usr/bin/env python3
"""POD6 verifier: own nested encoder probe refit from a states file.
Outer: speakers renumbered by position in default_rng(seed).permutation(unique str ids),
GroupKFold(5) assignment with a stable tie order (own code). Inner: the same stable
GroupKFold(4) on the training speakers' renumbered ids; layer = argmax of the mean inner
AUC (0.5 for a one-class inner fold); refit StandardScaler + LogisticRegression(max_iter=2000,
class_weight='balanced') on the outer training clips at that layer.
usage: POD6_verify_nested.py STATES.npz KEY OUT.csv [seeds]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import gkf_stable, renumber, auc_mw

NPZ, KEY, OUT = sys.argv[1:4]
SEEDS = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3, 4]
z = np.load(NPZ, allow_pickle=True)
y = z["label"].astype(int); spk = z["spk"].astype(str); names = [str(v) for v in z["name"]]
X = z[KEY].astype(np.float32)
nl = X.shape[1]


def fp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


def inner_score(tr, g, l):
    gi = gkf_stable(g[tr], 4); s = 0.0
    for f in range(4):
        ite = gi == f; itr = ~ite
        p = fp(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        yy = y[tr][ite]
        s += auc_mw(yy, p) if len(set(yy)) > 1 else 0.5
    return s / 4


cols = {"clip": names, "speaker": spk, "label": y}
res = {}
for seed in SEEDS:
    g = renumber(spk, seed) if seed >= 0 else np.unique(spk, return_inverse=True)[1]; fo = gkf_stable(spk if seed < 0 else g, 5)
    oof = np.zeros(len(y)); layers = []
    for f in range(5):
        te = np.where(fo == f)[0]; tr = np.where(fo != f)[0]
        sc = Parallel(n_jobs=min(nl, int(os.environ.get("VJOBS", "32"))))(delayed(inner_score)(tr, g, l) for l in range(nl))
        best = int(np.argmax(sc)); layers.append(best)
        oof[te] = fp(X[tr][:, best], y[tr], X[te][:, best])
    a = auc_mw(y, oof)
    res[seed] = dict(auc=a, layers=layers)
    tag = f"s{seed}" if seed >= 0 else "single"; cols[f"fold_{tag}"] = fo; cols[f"oof_{tag}"] = oof
    print(f"seed {seed}: AUC {a:.6f} layers {layers}", flush=True)
import pandas as pd
pd.DataFrame(cols).to_csv(OUT, index=False)
print("mean of five", np.mean([r["auc"] for r in res.values()]))
json.dump({str(k): v for k, v in res.items()}, open(OUT.replace(".csv", ".json"), "w"), indent=1)
