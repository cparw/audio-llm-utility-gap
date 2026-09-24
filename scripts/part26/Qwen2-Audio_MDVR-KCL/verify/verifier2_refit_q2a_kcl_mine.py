# Extra independent check: refit the nested encoder probe from the states on the Mac with my own fold code,
# and compare to the claimed per-clip OOF scores and chosen folds. Mac sklearn differs from the pod (1.7.2 vs 1.9.1),
# so exact equality is not expected; agreement of layers and AUCs to 4 dp is the target.
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from fractions import Fraction
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ST = "<local data dir>/paper1_local_runs/probe2/kcl_states.npz"
PERCLIP = "scores/part26/Qwen2-Audio_MDVR-KCL/Q2A_kcl_perclip.csv"
FOLDDIR = "<local data dir>/release/folds"
z = np.load(ST, allow_pickle=True)
X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
pc = {r["clip"]: r for r in csv.DictReader(open(PERCLIP))}
assert set(pc) == set(names)
order_ok = [r for r in pc]  # per-clip order
P_claim = np.array([[float(pc[n][f"p_probe_seed{s}"]) for s in range(5)] for n in names])
zs_states_eq = bool(np.array_equal(z["p_yes"].astype(float), np.array([float(pc[n]["p_yes_zeroshot"]) for n in names])))
lab_eq = all(int(pc[n]["label"]) == int(l) for n, l in zip(names, y)); spk_eq = all(pc[n]["speaker"] == s for n, s in zip(names, spk))

def greedy_folds(groups, k):
    # my own GroupKFold greedy fill: groups sorted by size, biggest first; ties broken by descending group id
    # (the pod rule: stable argsort of counts reversed); each goes to the fold with the least clips, lowest index first
    ug = sorted(set(groups.tolist()))
    cnt = {g: int((groups == g).sum()) for g in ug}
    seq = sorted(ug, key=lambda g: (cnt[g], ug.index(g)), reverse=True)
    load = [0] * k; fg = {}
    for g in seq:
        f = min(range(k), key=lambda j: (load[j], j)); fg[g] = f; load[f] += cnt[g]
    return np.array([fg[g] for g in groups.tolist()])

def fit_score(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]

def inner_auc(tr, l, g):
    f4 = greedy_folds(g[tr], 4); vals = []; fr = []
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        p = fit_score(X[a, l], y[a], X[b, l])
        vals.append(roc_auc_score(y[b], p) if len(set(y[b])) > 1 else 0.5)
        yb = y[b]; npos = int(yb.sum()); nneg = len(yb) - npos
        # exact fraction for tie inspection
        wins = sum((1.0 if pi > pj else 0.5 if pi == pj else 0.0) for pi, yi in zip(p, yb) if yi == 1 for pj, yj in zip(p, yb) if yj == 0)
        fr.append(Fraction(int(round(2 * wins)), 2 * npos * nneg) if npos and nneg else Fraction(1, 2))
    return sum(vals) / 4, sum(fr) / 4

res = {"states_label_equal": lab_eq, "states_speaker_equal": spk_eq, "states_pyes_equal_perclip_zeroshot": zs_states_eq, "seeds": {}}
for s in range(5):
    rng = np.random.default_rng(s); perm = rng.permutation(np.unique(spk)); pos = {u: i for i, u in enumerate(perm)}
    g = np.array([pos[u] for u in spk])
    outer = greedy_folds(g, 5)
    ff = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDDIR}/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv"))}
    saved = np.array([ff[n] for n in names])
    oof = np.zeros(len(y)); layers = []; ties = []
    for k in range(5):
        tr, te = np.where(saved != k)[0], np.where(saved == k)[0]
        sc = Parallel(n_jobs=8)(delayed(inner_auc)(tr, l, g) for l in range(X.shape[1]))
        fl = np.array([a for a, _ in sc]); ex = [b for _, b in sc]
        best = int(np.argmax(fl)); layers.append(best)
        mx = max(ex); tied = [l for l in range(len(ex)) if ex[l] == mx]
        if len(tied) > 1: ties.append({"fold": k, "exact_tied_layers": tied, "exact_value": str(mx), "float_argmax": best})
        oof[te] = fit_score(X[tr][:, best], y[tr], X[te][:, best])
    auc = float(roc_auc_score(y, oof)); auc_claim = float(roc_auc_score(y, P_claim[:, s]))
    res["seeds"][f"seed{s}"] = {"my_outer_rule_equals_saved_file": bool(np.array_equal(outer, saved)),
                                "layers": layers, "ties": ties, "auc_refit": auc, "auc_claim": auc_claim,
                                "max_abs_score_diff": float(np.max(np.abs(oof - P_claim[:, s])))}
    print(s, res["seeds"][f"seed{s}"], flush=True)
res["mean5_refit"] = float(np.mean([v["auc_refit"] for v in res["seeds"].values()]))
res["mean5_claim"] = float(np.mean([v["auc_claim"] for v in res["seeds"].values()]))
import sklearn; res["versions"] = {"numpy": np.__version__, "sklearn": sklearn.__version__}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refit_q2a_kcl_mine.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "seeds"}, indent=1))
