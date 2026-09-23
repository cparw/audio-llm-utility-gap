import os; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
"""Nested layer selection on saved states: outer GroupKFold(5) by speaker, inner GroupKFold(4) on the training
speakers picks the layer by mean inner AUC, refit at that layer, out-of-fold AUC. Same probe as extract_probe_layers.py
(StandardScaler + LogisticRegression l2 C=1 balanced 2000 iter). usage: nested_probe.py STATES.npz OUTPREFIX"""
import numpy as np, json, sys, csv
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
NPZ, OUTP = sys.argv[1:3]
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"]
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
def inner_auc_layer(X, y, spk, tr, l):
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups=spk[tr]):
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4
res = {}
for key in [k for k in ("enc", "proj", "llm", "ans") if k in z.files]:
    X = z[key].astype(np.float32); nl = X.shape[1]; oof = np.zeros(len(y)); chosen = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=spk):
        inner = Parallel(n_jobs=min(nl, max(2, os.cpu_count() - 2)))(delayed(inner_auc_layer)(X, y, spk, tr, l) for l in range(nl))
        best = int(np.argmax(inner)); chosen.append(best)
        oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best])
    res[key] = {"chosen_layers": chosen, "auc_nested_oof": float(roc_auc_score(y, oof)), "n": int(len(y))}
    with open(f"{OUTP}_{key}_nested_oof.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_probe"]); w.writerows(zip(names, spk, y, oof))
    print(key, res[key], flush=True)
json.dump(res, open(f"{OUTP}_nested.json", "w"), indent=1)
