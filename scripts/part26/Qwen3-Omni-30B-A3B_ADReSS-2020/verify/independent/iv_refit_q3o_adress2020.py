"""Independent refit (information only, Mac sklearn differs from the pod):
1) outer fits at the pod's chosen layers on the saved release fold files -> compare with the saved OOF;
2) same, but with deliberately wrong folds (single-split pod file, and the Mac tie-order file) to show the check has power;
3) my own inner GroupKFold(4) layer selection on the renumbered ids -> compare layers."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

CELL = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
FOLDDIR = "<local data dir>/release/folds"
LAY = {0: [19, 18, 31, 25, 29], 1: [18, 19, 29, 26, 25], 2: [28, 18, 30, 19, 19], 3: [30, 22, 22, 29, 19], 4: [19, 29, 19, 20, 30]}

z = np.load(STATES, allow_pickle=True)
X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
R = list(csv.DictReader(open(CELL + "/per_clip/p26_q3o_adress2020_perclip.csv")))
assert [r["clip"] for r in R] == list(names)
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in R])

def auc(yy, ss):
    r = rankdata(ss); a = int(yy.sum()); b = len(yy) - a
    return (r[yy == 1].sum() - a * (a + 1) / 2) / (a * b)

def fit(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]

def readfold(fn):
    d = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(FOLDDIR, fn)))}
    return np.array([d[c] for c in names])

def outer_at_layers(fold, layers):
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        oof[te] = fit(X[tr, layers[k]], y[tr], X[te, layers[k]])
    return oof

def gkf_stable(groups, k):
    u, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); fg = np.empty(len(u), int)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += cnt[gi]
    return fg[inv]

res = {"fixed_layer_refit": {}, "wrong_fold_control": {}, "inner_selection": {}}
for s in range(5):
    fold = readfold(f"adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")
    oof = outer_at_layers(fold, LAY[s])
    res["fixed_layer_refit"][f"seed{s}"] = dict(max_abs_diff=float(np.abs(oof - P[:, s]).max()),
                                              auc_mac=float(auc(y, oof)), auc_saved=float(auc(y, P[:, s])))
    for wrong in ("adress2020_groupkfold5_mac.csv", "adress2020_groupkfold5_pod.csv"):
        wf = readfold(wrong); o2 = outer_at_layers(wf, LAY[s])
        res["wrong_fold_control"][f"seed{s}_{wrong}"] = dict(n_clips_fold_differs=int((wf != fold).sum()),
                                                            max_abs_diff=float(np.abs(o2 - P[:, s]).max()))
print(json.dumps(res["fixed_layer_refit"], indent=1)); print(json.dumps(res["wrong_fold_control"], indent=1), flush=True)

def inner_score(tr, g, l):
    f4 = gkf_stable(g[tr], 4); sc = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        sc += auc(y[b], fit(X[a, l], y[a], X[b, l])) if len(set(y[b])) > 1 else 0.5
    return sc / 4

for s in range(5):
    rng = np.random.default_rng(s); u = np.unique(spk); perm = {q: i for i, q in enumerate(rng.permutation(u))}
    g = np.array([perm[q] for q in spk])
    fold = readfold(f"adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")
    assert np.array_equal(gkf_stable(g, 5), fold)
    mine = []; margins = []
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = Parallel(n_jobs=-1)(delayed(inner_score)(tr, g, l) for l in range(X.shape[1]))
        b = int(np.argmax(sc)); mine.append(b); srt = sorted(sc, reverse=True); margins.append(srt[0] - srt[1])
        oof[te] = fit(X[tr, b], y[tr], X[te, b])
    res["inner_selection"][f"seed{s}"] = dict(layers_mine=mine, layers_pod=LAY[s], equal=mine == LAY[s],
                                            auc_mine=float(auc(y, oof)), max_abs_diff_vs_saved=float(np.abs(oof - P[:, s]).max()),
                                            top_margins=margins)
    print(s, res["inner_selection"][f"seed{s}"], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iv_refit_q3o_adress2020.json"), "w"), indent=1)
print("DONE")
