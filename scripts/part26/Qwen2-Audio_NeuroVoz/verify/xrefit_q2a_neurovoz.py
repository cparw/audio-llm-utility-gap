"""PART26 independent refit check, Qwen2-Audio x NeuroVoz, on the Mac (own code, not the producer's scripts).
For each of the 5 repeats and each outer fold taken from the SAVED release fold file:
  inner selection on the outer training rows only (own rebuild of the inner 4-fold speaker split),
  score all 33 encoder stages by mean inner AUC (own rank AUC), argmax, refit StandardScaler +
  LogisticRegression(max_iter=2000, class_weight='balanced') on the outer training rows, predict held-out rows.
Compares chosen layers with the claimed layers and predictions with the per-clip OOF file.
Mac sklearn/numpy differ from the pod, so predictions are compared with a tolerance and AUCs to 4 dp.
Writes only xrefit_q2a_neurovoz.json next to this script."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, sys, time, warnings
import numpy as np
from scipy.stats import rankdata
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CELL = "scores/part26/Qwen2-Audio_NeuroVoz"
FOLDDIR = "<local data dir>/release/folds"
CLAIM_LAYERS = {"seed0": [24, 25, 18, 24, 20], "seed1": [20, 20, 30, 22, 25], "seed2": [17, 21, 25, 19, 30],
                "seed3": [20, 25, 20, 17, 20], "seed4": [20, 30, 24, 21, 28]}
OUT = f"{CELL}/verify/xrefit_q2a_neurovoz.json"

z = np.load(f"{CELL}/stage/q2a_neurovoz_encstates.npz", allow_pickle=True)
X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
pc = {r["clip"]: r for r in csv.DictReader(open(f"{CELL}/per_clip/p26_q2a_neurovoz_perclip.csv", newline=""))}
NL = X.shape[1]

def auc(yy, s):
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return 0.5
    r = rankdata(s); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def greedy(g, k):
    ids = sorted(set(g.tolist())); cnt = {i: int((g == i).sum()) for i in ids}
    load = [0] * k; fg = {}
    for i in sorted(ids, key=lambda i: (-cnt[i], -i)):
        f = min(range(k), key=lambda j: (load[j], j)); fg[i] = f; load[f] += cnt[i]
    return np.array([fg[i] for i in g])

def fit_predict(XX, yv, tr, te, l):
    sc = StandardScaler().fit(XX[tr, l])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(XX[tr, l]), yv[tr])
    return m.predict_proba(sc.transform(XX[te, l]))[:, 1]

def inner_score(XX, yv, tr, g, l):
    f4 = greedy(g[tr], 4); s = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        s += auc(yv[b], fit_predict(XX, yv, a, b, l))
    return s / 4

t0 = time.time(); out = {"tags": {}}
allok = True
for seed in range(5):
    tag = f"seed{seed}"
    u = np.unique(spk); perm = np.random.default_rng(seed).permutation(u)
    g = np.array([{s: i for i, s in enumerate(perm)}[s] for s in spk])
    fr = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDDIR}/neurovoz_groupkfold5_pod_{tag}_UNVERIFIED.csv"))}
    fold = np.array([fr[n] for n in names])
    oof = np.full(len(y), np.nan); layers = []; margins = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        assert not set(spk[tr]) & set(spk[te])
        sc = Parallel(n_jobs=16)(delayed(inner_score)(X, y, tr, g, l) for l in range(NL))
        best = int(np.argmax(sc)); srt = sorted(sc, reverse=True)
        layers.append(best); margins.append(srt[0] - srt[1])
        oof[te] = fit_predict(X, y, tr, te, best)
        print(f"{tag} fold {k} layer {best} (claimed {CLAIM_LAYERS[tag][k]}) margin {srt[0]-srt[1]:.2e} {time.time()-t0:.0f}s", flush=True)
    claimed = np.array([float(pc[n][f"p_probe_{tag}"]) for n in names])
    d = float(np.abs(oof - claimed).max()); a_mine = auc(y, oof); a_claim = auc(y, claimed)
    ok = layers == CLAIM_LAYERS[tag] and f"{a_mine:.4f}" == f"{a_claim:.4f}"
    allok &= ok
    out["tags"][tag] = {"layers": layers, "claimed_layers": CLAIM_LAYERS[tag], "layers_equal": layers == CLAIM_LAYERS[tag],
                        "inner_margin_top_vs_runnerup": margins, "max_abs_pred_diff": d,
                        "auc_refit": a_mine, "auc_claimed_oof": a_claim, "auc_4dp_equal": f"{a_mine:.4f}" == f"{a_claim:.4f}"}
    print(f"{tag} layers_equal {layers == CLAIM_LAYERS[tag]} max|dp| {d:.3g} auc refit {a_mine:.6f} claimed {a_claim:.6f}", flush=True)
import numpy, sklearn, scipy, platform
out["versions"] = {"numpy": numpy.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "platform": platform.platform()}
out["seconds"] = round(time.time() - t0, 1); out["all_pass"] = bool(allok)
json.dump(out, open(OUT, "w"), indent=1)
print("REFIT ALL_PASS", allok)
