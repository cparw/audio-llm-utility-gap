"""Extra independent check: rebuild the nested encoder probe from the states on the Mac with my own code.
Outer folds from the release fold files; inner 4 folds by greedy fill of the training speakers
(group ids sorted descending, each to the lightest fold, first on ties); every layer scored by mean inner AUC
kept as an exact fraction; tie set at the max; refit at the saved layer and compare with the pod OOF."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import csv, json, sys, warnings
from fractions import Fraction
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

ROOT = "scores/part26/Kimi-Audio_MDVR-KCL"
ST = ROOT + "/pull/p26-kimi-kcl-gpu/out/p26_kimi_kcl_states.npz"
PODJ = ROOT + "/pull/p26-kimi-kcl-cpu/out/p26_kimi_kcl_enc_nested5.json"
PODOOF = ROOT + "/pull/p26-kimi-kcl-cpu/out/p26_kimi_kcl_enc_nested5_oof.csv"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"

z = np.load(ST, allow_pickle=False)
X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
L = X.shape[1]
podj = json.load(open(PODJ))
pod = {r["clip"]: r for r in csv.DictReader(open(PODOOF))}
zs = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(ZS))}
Z = np.array([zs[n] for n in names])

def fauc(yy, ss):
    pos = [s for s, t in zip(ss, yy) if t == 1]; neg = [s for s, t in zip(ss, yy) if t == 0]
    if not pos or not neg:
        return Fraction(1, 2)
    c = Fraction(0)
    for a in pos:
        for b in neg:
            c += 1 if a > b else (Fraction(1, 2) if a == b else 0)
    return c / (len(pos) * len(neg))

def greedy(groups, k):
    ug = sorted(set(groups.tolist()))
    cnt = {g: int((groups == g).sum()) for g in ug}
    order = list(reversed(ug))  # all counts are 1 here; stable sort by count then reversed
    order = sorted(order, key=lambda g: -cnt[g]) if len(set(cnt.values())) > 1 else order
    load = [0] * k; f = {}
    for g in order:
        j = load.index(min(load)); f[g] = j; load[j] += cnt[g]
    return np.array([f[g] for g in groups])

def fit(Xtr, ytr, Xte):
    s = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(Xtr), ytr)
    return m.predict_proba(s.transform(Xte))[:, 1]

def layer_score(tr, l, g):
    f4 = greedy(g[tr], 4); tot = Fraction(0)
    for j in range(4):
        a = tr[f4 != j]; b = tr[f4 == j]
        tot += fauc(y[b], fit(X[a, l], y[a], X[b, l]))
    return tot / 4

tags = ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]
res = {}
for t in tags:
    if t == "single":
        g = spk.copy(); ff = f"{FD}/kcl_groupkfold5_pod.csv"
    else:
        s = int(t[4:]); perm = np.random.default_rng(s).permutation(np.unique(spk))
        m = {p: i for i, p in enumerate(perm)}; g = np.array([m[x] for x in spk])
        ff = f"{FD}/kcl_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fm = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(ff))}
    fold = np.array([fm[n] for n in names])
    saved = podj["tags"][t]["layers"]
    oof_saved_layer = np.zeros(len(y)); info = []
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        sc = Parallel(n_jobs=8)(delayed(layer_score)(tr, l, g) for l in range(L))
        mx = max(sc); ties = [l for l in range(L) if sc[l] == mx]
        oof_saved_layer[te] = fit(X[tr, saved[k]], y[tr], X[te, saved[k]])
        podinner = podj["tags"][t]["inner"][k]
        info.append(dict(fold=k, saved_layer=saved[k], my_argmax_first=int(np.argmax([float(v) for v in sc])),
                         tie_set=ties, max_inner=str(mx), saved_in_tie_set=saved[k] in ties,
                         my_inner_vs_pod_maxabs=float(max(abs(float(a) - b) for a, b in zip(sc, podinner)))))
    podp = np.array([float(pod[n][f"p_{t}"]) for n in names])
    res[t] = dict(folds=info, mac_refit_auc=float(fauc(y, oof_saved_layer)), pod_auc=podj["tags"][t]["auc"],
                  pod_auc_exact=str(fauc(y, podp)), mac_refit_auc_exact=str(fauc(y, oof_saved_layer)),
                  max_abs_pred_diff_vs_pod=float(np.max(np.abs(oof_saved_layer - podp))))
    print(t, res[t]["mac_refit_auc"], res[t]["pod_auc"], res[t]["max_abs_pred_diff_vs_pod"],
          [(d["saved_layer"], d["tie_set"]) for d in info], flush=True)

# tie swap range: for each fold with more than one tied layer, swap in each other tied layer (one fold at a time)
base_per = [fauc(y, np.array([float(pod[n][f"p_seed{s}"]) for n in names])) for s in range(5)]
ans = fauc(y, Z)
mean_vals = []; gap_vals = []
nt = 0
for t in tags:
    for d in res[t]["folds"]:
        if len(d["tie_set"]) > 1:
            nt += 1
            if t == "single":
                continue
            s = int(t[4:])
            perm = np.random.default_rng(s).permutation(np.unique(spk)); m = {p: i for i, p in enumerate(perm)}
            fm = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FD}/kcl_groupkfold5_pod_{t}_UNVERIFIED.csv"))}
            fold = np.array([fm[n] for n in names]); k = d["fold"]
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            for alt in d["tie_set"]:
                if alt == d["saved_layer"]:
                    continue
                p = np.array([float(pod[n][f"p_{t}"]) for n in names])
                p[te] = fit(X[tr, alt], y[tr], X[te, alt])
                per = list(base_per); per[s] = fauc(y, p)
                mv = sum(per) / 5; mean_vals.append(float(mv)); gap_vals.append(float(mv - ans))
out = dict(tags=res, n_folds_with_ties=nt,
           swap_mean5_range=[min(mean_vals), max(mean_vals)] if mean_vals else None,
           swap_gap_range=[min(gap_vals), max(gap_vals)] if gap_vals else None,
           all_saved_in_tie_set=all(d["saved_in_tie_set"] for t in tags for d in res[t]["folds"]))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "myrefit_out.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "tags"}, indent=1))
