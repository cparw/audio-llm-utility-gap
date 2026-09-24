"""Extra independent check: rebuild the nested encoder probe from the saved states on the Mac (own code), outer folds
from the release fold files, inner 4-fold greedy split (stable ties) on the default_rng(seed) renumbered ids, layer by
argmax of mean inner AUC, StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced').
Compares layers and per-clip OOF with the cell's per-clip csv. Read only on every input."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import csv, json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CELL = "scores/part26/Qwen2-Audio_MDVR-KCL"
z = np.load("<local data dir>/paper1_local_runs/probe2/kcl_states.npz", allow_pickle=True)
names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int); X = z["enc"].astype(np.float32)
pc = {r["clip"]: r for r in csv.DictReader(open(f"{CELL}/Q2A_kcl_perclip.csv"))}
side = json.load(open(f"{CELL}/Q2A_kcl_perclip.sidecar.json"))["probe"]["layers"]
P = np.array([[float(pc[n][f"p_probe_seed{s}"]) for s in range(5)] for n in names])

def greedy(groups, k):
    u, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); fg = np.zeros(len(u), int)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += cnt[gi]
    return fg[inv]

def fit(Xa, ya, Xb):
    sc = StandardScaler().fit(Xa)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xa), ya).predict_proba(sc.transform(Xb))[:, 1]

def inner_score(tr, l, g):
    f4 = greedy(g[tr], 4); tot = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        tot += roc_auc_score(y[b], fit(X[a, l], y[a], X[b, l])) if len(set(y[b])) > 1 else 0.5
    return tot / 4

res = {}
for s in range(5):
    fr = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(
        f"<local data dir>/release/folds/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv"))}
    fold = np.array([fr[n] for n in names])
    u = np.unique(spk); perm = {v: i for i, v in enumerate(np.random.default_rng(s).permutation(u))}
    g = np.array([perm[v] for v in spk])
    oof = np.zeros(len(y)); oof_saved_layer = np.zeros(len(y)); layers = []; margins = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = np.array(Parallel(n_jobs=8)(delayed(inner_score)(tr, l, g) for l in range(X.shape[1])))
        best = int(np.argmax(sc)); layers.append(best)
        srt = np.sort(sc)[::-1]; margins.append(float(srt[0] - srt[1]))
        oof[te] = fit(X[tr, best], y[tr], X[te, best])
        L = side[f"seed{s}"][k]
        oof_saved_layer[te] = fit(X[tr, L], y[tr], X[te, L])
    res[f"seed{s}"] = dict(layers_own=layers, layers_sidecar=side[f"seed{s}"], top_margins=margins,
                           auc_own_nested=roc_auc_score(y, oof), auc_own_at_saved_layers=roc_auc_score(y, oof_saved_layer),
                           auc_cell=roc_auc_score(y, P[:, s]),
                           maxabs_own_nested_vs_cell=float(np.max(np.abs(oof - P[:, s]))),
                           maxabs_saved_layers_vs_cell=float(np.max(np.abs(oof_saved_layer - P[:, s]))))
    print(s, json.dumps({k: (np.round(v, 6).tolist() if isinstance(v, (list, float)) else v) for k, v in res[f'seed{s}'].items()}), flush=True)
res["mean_own_nested"] = float(np.mean([res[f"seed{s}"]["auc_own_nested"] for s in range(5)]))
res["mean_own_at_saved_layers"] = float(np.mean([res[f"seed{s}"]["auc_own_at_saved_layers"] for s in range(5)]))
print("MEAN own nested", round(res["mean_own_nested"], 6), "own at saved layers", round(res["mean_own_at_saved_layers"], 6))
json.dump(res, open(sys.argv[1], "w"), indent=1)
