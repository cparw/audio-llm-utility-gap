"""Nested layer selection, averaged over R repeated speaker-disjoint 5-fold splits.
Outer GroupKFold(5), inner GroupKFold(4) on the training speakers picks the layer, refit, out-of-fold AUC.
usage: nested_repeats_all.py STATES.npz OUTPREFIX [R]"""
import os; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np, sys, json, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
NPZ, OUTP = sys.argv[1:3]; R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str)
NJ = max(2, (os.cpu_count() or 8) - 4)
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
def inner(X, tr, l, g):
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups=g[tr]):
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4
res = {}
for key in [k for k in ("enc", "proj", "llm", "ans") if k in z.files]:
    X = z[key].astype(np.float32); nl = X.shape[1]; aucs = []
    for seed in range(1 if nl == 1 else R):
        rng = np.random.default_rng(seed); u = np.unique(spk)
        perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
            best = 0 if nl == 1 else int(np.argmax(Parallel(n_jobs=min(nl, NJ))(delayed(inner)(X, tr, l, g) for l in range(nl))))
            oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best])
        aucs.append(float(roc_auc_score(y, oof)))
    res[key] = {"mean": round(float(np.mean(aucs)), 4), "repeats": len(aucs), "per_repeat": [round(a, 4) for a in aucs], "n": int(len(y)), "n_points": nl}
    print(f"{key}: mean {res[key]['mean']:.4f} over {len(aucs)} splits  {res[key]['per_repeat']}", flush=True)
json.dump(res, open(f"{OUTP}_nested_repeats.json", "w"), indent=1)
print("NESTED REPEATS DONE", flush=True)
