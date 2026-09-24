"""Information-only Mac refit: do the saved fold files explain the pod OOF scores?
Refit StandardScaler + LR(max_iter=2000, balanced) on each outer training set at the layer the pod saved,
compare with the pod OOF scores. Then repeat with a wrong split (Mac tie order single split) as a negative control.
Also re-derive the inner layer choice (inner GroupKFold(4), stable tie rule on renumbered ids, mean of inner AUCs)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import csv, json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CELL = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
pc = list(csv.DictReader(open(CELL + "/per_clip/p26_q3o_adress2020_perclip.csv")))
pj = json.load(open(CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5.json"))
z = np.load(STATES); X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str)
assert list(z["name"].astype(str)) == [r["clip"] for r in pc]
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
F = np.array([[int(r[f"fold_seed{s}"]) for s in range(5)] for r in pc])
CLAIM_LAYERS = {"seed0": [19, 18, 31, 25, 29], "seed1": [18, 19, 29, 26, 25], "seed2": [28, 18, 30, 19, 19],
                "seed3": [30, 22, 22, 29, 19], "seed4": [19, 29, 19, 20, 30]}

def fp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

def auc(yy, s):
    p = s[yy == 1][:, None]; n = s[yy == 0][None, :]; return float(((p > n) + 0.5 * (p == n)).mean())

def stable(groups, k):
    u, inv = np.unique(groups, return_inverse=True); c = np.bincount(inv)
    o = np.argsort(c, kind="stable")[::-1]; fg = np.zeros(len(u), int); ld = np.zeros(k)
    for gi in o:
        f = int(np.argmin(ld)); fg[gi] = f; ld[f] += c[gi]
    return fg[inv]

def inner_score(tr, l, g):
    f4 = stable(g[tr], 4); s = 0.0
    for j in range(4):
        a, b = np.where(f4 != j)[0], np.where(f4 == j)[0]
        yy = y[tr][b]; s += auc(yy, fp(X[tr][a, l], y[tr][a], X[tr][b, l])) if yy.min() != yy.max() else 0.5
    return s / 4

res = {"layers_pod_json_equal_claim": all(pj["tags"][t]["layers"] == CLAIM_LAYERS[t] for t in CLAIM_LAYERS)}
maxd = 0.0; lay_mis = []; per_refit = []
for s in range(5):
    t = f"seed{s}"; rng = np.random.default_rng(s); u = np.unique(spk)
    m = {q: i for i, q in enumerate(rng.permutation(u))}; g = np.array([m[q] for q in spk])
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(F[:, s] != k)[0], np.where(F[:, s] == k)[0]
        sc = Parallel(n_jobs=8)(delayed(inner_score)(tr, l, g) for l in range(X.shape[1]))
        best = int(np.argmax(sc))
        if best != CLAIM_LAYERS[t][k]: lay_mis.append([t, k, best, CLAIM_LAYERS[t][k], float(sorted(sc)[-1] - sorted(sc)[-2])])
        oof[te] = fp(X[tr][:, CLAIM_LAYERS[t][k]], y[tr], X[te][:, CLAIM_LAYERS[t][k]])
    maxd = max(maxd, float(np.abs(oof - P[:, s]).max())); per_refit.append(auc(y, oof))
res.update(refit_at_saved_layers_max_abs_pred_diff=maxd, refit_per_repeat_auc=per_refit,
           refit_mean5=float(np.mean(per_refit)), inner_layer_mismatches=lay_mis)
# negative control: seed0 scores vs refit on the Mac-order single split at the same per-fold layers
mac = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open("<local data dir>/release/folds/adress2020_groupkfold5_mac.csv"))}
Fm = np.array([mac[r["clip"]] for r in pc]); oof = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(Fm != k)[0], np.where(Fm == k)[0]
    oof[te] = fp(X[tr][:, CLAIM_LAYERS["seed0"][k]], y[tr], X[te][:, CLAIM_LAYERS["seed0"][k]])
res["negative_control_wrong_split_max_abs_pred_diff_seed0"] = float(np.abs(oof - P[:, 0]).max())
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
