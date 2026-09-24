"""Extra independent evidence (Mac, sklearn as installed): refit StandardScaler + LogisticRegression(max_iter=2000,
class_weight='balanced') on each outer training set at the claimed layer (mean-pooled encoder states from the staged npz),
score the held-out speakers, compare to the per-clip OOF file. Also re-derive the inner layer choice for seed0 outer fold 0
with my own inner GroupKFold(4) greedy fill on the renumbered speaker ids."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, sys, numpy as np, sklearn
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
CELL = "scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
R = list(csv.DictReader(open(f"{CELL}/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv")))
S = np.load(f"{CELL}/stage/q3o_neurovoz_encstates.npz")
X = S["enc"]; y = S["label"].astype(int); spk = S["spk"].astype(str)
assert list(S["name"].astype(str)) == [r["clip"] for r in R]
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in R] for s in range(5)]); FO = np.array([[int(r[f"fold_seed{s}"]) for r in R] for s in range(5)])
L = {"seed0": [31, 29, 31, 31, 19], "seed1": [29, 31, 31, 28, 29], "seed2": [17, 26, 17, 14, 21], "seed3": [30, 27, 26, 17, 21], "seed4": [21, 20, 29, 26, 25]}
def auc(yy, s):
    r = rankdata(s); n1 = int(yy.sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def fp(tr, te, l):
    sc = StandardScaler().fit(X[tr, l])
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr]).predict_proba(sc.transform(X[te, l]))[:, 1]
jobs = [(s, k) for s in range(5) for k in range(5)]
res = Parallel(n_jobs=int(sys.argv[2]))(delayed(fp)(np.flatnonzero(FO[s] != k), np.flatnonzero(FO[s] == k), L[f"seed{s}"][k]) for s, k in jobs)
Q = np.full((5, len(y)), np.nan)
for (s, k), p in zip(jobs, res): Q[s, FO[s] == k] = p
out = {"sklearn": sklearn.__version__, "numpy": np.__version__}
out["refit_per_repeat_auc"] = [auc(y, Q[s]) for s in range(5)]
out["oof_per_repeat_auc"] = [auc(y, P[s]) for s in range(5)]
out["refit_mean5"] = float(np.mean(out["refit_per_repeat_auc"]))
out["max_abs_auc_diff"] = float(max(abs(a - b) for a, b in zip(out["refit_per_repeat_auc"], out["oof_per_repeat_auc"])))
out["max_abs_pred_diff"] = float(np.max(np.abs(Q - P)))
out["median_abs_pred_diff"] = float(np.median(np.abs(Q - P)))
out["rank_corr_min"] = float(min(np.corrcoef(rankdata(Q[s]), rankdata(P[s]))[0, 1] for s in range(5)))
# inner re-derivation, seed0 outer fold 0
rng = np.random.default_rng(0); u = np.unique(spk); perm = {a: i for i, a in enumerate(rng.permutation(u))}; g = np.array([perm[a] for a in spk])
def fill(groups, k):
    uu, inv = np.unique(groups, return_inverse=True); c = np.bincount(inv); o = np.argsort(c, kind="stable")[::-1]
    load = np.zeros(k); fg = np.empty(len(uu), int)
    for gi in o:
        j = int(np.argmin(load)); fg[gi] = j; load[j] += c[gi]
    return fg[inv]
tr = np.flatnonzero(FO[0] != 0); f4 = fill(g[tr], 4)
def inner(l):
    s = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]; s += auc(y[b], fp(a, b, l))
    return s / 4
sc = Parallel(n_jobs=int(sys.argv[2]))(delayed(inner)(l) for l in range(X.shape[1]))
J = json.load(open(f"{CELL}/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5.json"))
out["inner_seed0_fold0_argmax"] = int(np.argmax(sc)); out["inner_seed0_fold0_claimed"] = L["seed0"][0]
out["inner_seed0_fold0_max_abs_diff_vs_pod"] = float(np.max(np.abs(np.array(sc) - np.array(J["tags"]["seed0"]["inner"][0]))))
srt = sorted(sc, reverse=True); out["inner_seed0_fold0_top2"] = srt[:2]
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
