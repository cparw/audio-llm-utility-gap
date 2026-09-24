"""Independent fold-usage check for PART26 Qwen3-Omni NeuroVoz.
1) rebuild the outer folds from scratch (seeded speaker renumbering + greedy largest-group-first fill) and compare to the saved files
2) refit the outer model at the chosen layer on each outer training set and compare to the saved OOF scores
3) redo the inner layer search for two outer folds and compare with the pod's recorded inner scores."""
import csv, json, sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

B = "scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
z = np.load(B + "/stage/q3o_neurovoz_encstates.npz")
X = z["enc"]; y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
pc = list(csv.DictReader(open(B + "/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv")))
assert [r["clip"] for r in pc] == names.tolist()
pj = json.load(open(B + "/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5.json"))
out = {}

def groups_for(tag):
    if tag == "single": return spk
    u = np.unique(spk); p = np.random.default_rng(int(tag[4:])).permutation(u)
    m = dict(zip(p.tolist(), range(len(p)))); return np.array([m[s] for s in spk])

def greedy(g, k):
    u = np.unique(g); cnt = np.array([(g == v).sum() for v in u])
    # largest first; among equal counts the later group in sorted order goes first
    order = sorted(range(len(u)), key=lambda i: (-cnt[i], -i))
    load = [0] * k; fold_of = {}
    for i in order:
        f = min(range(k), key=lambda j: (load[j], j)); fold_of[u[i]] = f; load[f] += cnt[i]
    return np.array([fold_of[v] for v in g])

def fit(tr, te, l):
    sc = StandardScaler().fit(X[tr, l])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
    return m.predict_proba(sc.transform(X[te, l]))[:, 1]

tags = [f"seed{s}" for s in range(5)] + ["single"]
col = {t: ("p_probe_single_rerun" if t == "single" else f"p_probe_{t}") for t in tags}
fcol = {t: ("fold_single" if t == "single" else f"fold_{t}") for t in tags}
jobs = []
for t in tags:
    g = groups_for(t); f = np.array([int(r[fcol[t]]) for r in pc])
    out[f"rebuilt_outer_equals_saved_{t}"] = bool(np.array_equal(greedy(g, 5), f))
    for k in range(5):
        jobs.append((t, k, np.where(f != k)[0], np.where(f == k)[0], pj["tags"][t]["layers"][k]))
res = Parallel(n_jobs=6)(delayed(fit)(tr, te, l) for (_, _, tr, te, l) in jobs)
refit = {t: np.full(len(y), np.nan) for t in tags}
for (t, k, tr, te, l), p in zip(jobs, res): refit[t][te] = p
for t in tags:
    saved = np.array([float(r[col[t]]) for r in pc])
    out[f"refit_{t}"] = {"max_abs_pred_diff": float(np.max(np.abs(refit[t] - saved))),
                         "rank_corr": float(np.corrcoef(np.argsort(np.argsort(refit[t])), np.argsort(np.argsort(saved)))[0, 1]),
                         "auc_refit": float(roc_auc_score(y, refit[t])), "auc_saved": float(roc_auc_score(y, saved))}
    print(t, out[f"rebuilt_outer_equals_saved_{t}"], out[f"refit_{t}"], flush=True)

# inner layer search, own inner folds, for seed0 fold 0 and seed2 fold 4
def inner_score(tr, g, l):
    f4 = greedy(g[tr], 4); s = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        s += roc_auc_score(y[b], fit(a, b, l))
    return s / 4
for t, k in (("seed0", 0), ("seed2", 4)):
    g = groups_for(t); f = np.array([int(r[fcol[t]]) for r in pc]); tr = np.where(f != k)[0]
    sc = Parallel(n_jobs=6)(delayed(inner_score)(tr, g, l) for l in range(X.shape[1]))
    podsc = np.array(pj["tags"][t]["inner"][k])
    out[f"inner_{t}_fold{k}"] = {"argmax_mine": int(np.argmax(sc)), "argmax_pod": int(np.argmax(podsc)),
                                 "layer_pod": pj["tags"][t]["layers"][k],
                                 "max_abs_inner_diff": float(np.max(np.abs(np.array(sc) - podsc)))}
    print(t, k, out[f"inner_{t}_fold{k}"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
