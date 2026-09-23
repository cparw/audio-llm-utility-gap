"""lo-m7m11-1: ADReSS-2020 near twin. Refit the Table 1 five-repeat nested encoder probe
(same code path as scripts/nested_repeats_all.py: speakers renumbered with default_rng(seed).permutation,
outer GroupKFold(5), inner GroupKFold(4) layer choice by mean inner AUC, StandardScaler + LogisticRegression
(max_iter=2000, class_weight=balanced), seeds 0..4) on the PART 16 POD4B re-extracted Qwen2.5-Omni
ADReSS-2020 'orig' states, and SAVE the per-clip OOF for every repeat, the fold ids and the chosen layers.
The Table 1 states themselves are gone (pod gv9jvm57b21mcz deleted), so this is a near twin, not the Table 1 fit.
usage: python3 pod1_adress2020_rep5_refit.py data/adress2020_orig_enc.npz out/adress2020_pod4b"""
import os; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np, sys, json, time, warnings; warnings.filterwarnings("ignore")
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import sklearn, scipy
NPZ, OUTP = sys.argv[1:3]; R = 5
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); name = z["name"].astype(str)
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
X = z["enc"].astype(np.float32); nl = X.shape[1]
t0 = time.time(); res = {"per_repeat": [], "chosen_layers": {}, "n": int(len(y)), "n_points": int(nl)}
df = pd.DataFrame({"clip": name, "speaker": spk, "label": y})
for seed in range(R):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    oof = np.zeros(len(y)); fold = np.full(len(y), -1); layers = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
        best = int(np.argmax(Parallel(n_jobs=min(nl, NJ))(delayed(inner)(X, tr, l, g) for l in range(nl))))
        oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best]); fold[te] = k; layers.append(best)
    a = float(roc_auc_score(y, oof)); res["per_repeat"].append(a); res["chosen_layers"][f"seed{seed}"] = layers
    df[f"p_seed{seed}"] = oof; df[f"fold_seed{seed}"] = fold
    print(f"seed {seed}: AUC {a:.6f} layers {layers}  {time.time()-t0:.0f}s", flush=True)
res["mean5"] = float(np.mean(res["per_repeat"]))
res["versions"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}
res["runtime_s"] = round(time.time()-t0, 1)
df.to_csv(f"{OUTP}_enc_rep5_oof.csv", index=False)
json.dump(res, open(f"{OUTP}_enc_rep5.json", "w"), indent=1)
print(f"mean5 {res['mean5']:.6f}  per_repeat {[round(v,4) for v in res['per_repeat']]}", flush=True)
