"""PART26 Qwen2-Audio PC-GITA addendum to the independent check. The Mac refit reproduced every pod layer choice but
per-clip scores differ from the pod by up to 0.037 (the verify json keeps that check as failed, threshold 1e-4, set
before the run). The two Linux pods with identical sklearn/numpy/scipy (this run on AVX2 EPYC 7702P, PART25 C p25C-2 on
AVX-512 EPYC 9655P, same folds, same layers) differ by up to 0.031 too. This script measures what that means for the
reported numbers: it refits the 25 outer folds on the Mac at the pod layers (saving the scores and the LBFGS
iteration counts), and runs the same 2000-draw paired speaker bootstrap (default_rng(0)) on (a) the pod OOF,
(b) the Mac refit OOF and (c) the PART25 C OOF. Writes verify/sensitivity26.json and verify/mac_refit_oof.csv."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, numpy as np, warnings, sklearn; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
C = "scores/part26/Qwen2-Audio_PC-GITA"
S = json.load(open(f"{C}/Qwen2-Audio_PC-GITA_perclip.sidecar.json"))
rows = list(csv.DictReader(open(f"{C}/Qwen2-Audio_PC-GITA_perclip.csv"))); names = [r["clip"] for r in rows]
z = np.load("<local data dir>/paper1_local_runs/probe2/pcgita_states.npz", allow_pickle=True)
assert list(z["name"].astype(str)) == names
X = z["enc"].astype(np.float32); y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
Ppod = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)])
Pmac = np.zeros_like(Ppod); iters = []
for s in range(5):
    fo = np.array([int(r[f"fold_seed{s}"]) for r in rows]); L = S["layers"][f"seed{s}"]
    for k in range(5):
        tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
        sc = StandardScaler().fit(X[tr][:, L[k]])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr][:, L[k]]), y[tr])
        Pmac[s, te] = m.predict_proba(sc.transform(X[te][:, L[k]]))[:, 1]; iters.append(int(m.n_iter_[0]))
B = {r["clip"]: r for r in csv.DictReader(open("<local data dir>/release_from_mac/scores/part25/C/pull/p25C-2/out/q2a_pcgita_enc_rep5_oof.csv"))}
Pc25 = np.array([[float(B[c][f"p_seed{s}"]) for c in names] for s in range(5)])
with open(f"{C}/verify/mac_refit_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label"] + [f"p_mac_seed{s}" for s in range(5)])
    for i, c in enumerate(names): w.writerow([c, spk[i], y[i]] + [repr(float(Pmac[s, i])) for s in range(5)])
def boot(P):
    u = np.unique(spk); rows_ = [np.where(spk == q)[0] for q in u]; rng = np.random.default_rng(0); d = []
    for b in range(2000):
        ii = np.concatenate([rows_[i] for i in rng.choice(len(u), size=len(u), replace=True)])
        d.append(np.mean([roc_auc_score(y[ii], P[r][ii]) for r in range(5)]) - roc_auc_score(y[ii], zs[ii]))
    d = np.array(d); m5 = float(np.mean([roc_auc_score(y, P[r]) for r in range(5)]))
    return {"mean5": m5, "gap": m5 - float(roc_auc_score(y, zs)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
out = {"lbfgs_iterations_outer_refits_mac": {"min": min(iters), "max": max(iters), "n_at_max_iter_2000": int(sum(i >= 2000 for i in iters))},
       "max_abs_pred_diff": {"mac_vs_pod": float(np.max(np.abs(Pmac - Ppod))), "c25_vs_pod": float(np.max(np.abs(Pc25 - Ppod))), "mac_vs_c25": float(np.max(np.abs(Pmac - Pc25)))},
       "pod": boot(Ppod), "mac_refit": boot(Pmac), "part25C_avx512": boot(Pc25), "mac_sklearn": sklearn.__version__}
for k in ("pod", "mac_refit", "part25C_avx512"):
    out[k]["4dp"] = [round(out[k]["mean5"], 4), round(out[k]["gap"], 4), round(out[k]["ci"][0], 4), round(out[k]["ci"][1], 4)]
out["gap_and_ci_same_at_3dp_all_three"] = len({tuple(round(v, 3) for v in (out[k]["gap"], *out[k]["ci"])) for k in ("pod", "mac_refit", "part25C_avx512")}) == 1
json.dump(out, open(f"{C}/verify/sensitivity26.json", "w"), indent=1); print(json.dumps(out, indent=1))
