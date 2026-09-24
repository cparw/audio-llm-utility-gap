"""PART26 Qwen2-Audio PC-GITA: clip-by-clip comparison of this run's per-repeat OOF (AVX2 EPYC 7702P pod, saved fold
files, pod tie rule inner) with the PART25 C refit (p25C-2, AVX-512 EPYC 9655P pod, pod sklearn GroupKFold outer and
inner, same sklearn/numpy/scipy). Writes verify/compare_c25.json."""
import csv, json, numpy as np
from sklearn.metrics import roc_auc_score
C = "scores/part26/Qwen2-Audio_PC-GITA"
A = list(csv.DictReader(open(f"{C}/Qwen2-Audio_PC-GITA_perclip.csv")))
B = {r["clip"]: r for r in csv.DictReader(open("<local data dir>/release_from_mac/scores/part25/C/pull/p25C-2/out/q2a_pcgita_enc_rep5_oof.csv"))}
assert set(B) == {r["clip"] for r in A}
y = np.array([int(r["label"]) for r in A]); out = {}
for s in range(5):
    pa = np.array([float(r[f"p_probe_seed{s}"]) for r in A]); pb = np.array([float(B[r["clip"]][f"p_seed{s}"]) for r in A])
    fa = np.array([int(r[f"fold_seed{s}"]) for r in A]); fb = np.array([int(B[r["clip"]][f"fold_seed{s}"]) for r in A])
    out[f"seed{s}"] = {"folds_equal": bool(np.array_equal(fa, fb)), "max_abs_pred_diff": float(np.max(np.abs(pa - pb))),
                       "median_abs_pred_diff": float(np.median(np.abs(pa - pb))), "n_clips_diff_gt_1e-6": int((np.abs(pa - pb) > 1e-6).sum()),
                       "auc_this_run": float(roc_auc_score(y, pa)), "auc_c25": float(roc_auc_score(y, pb)),
                       "folds_with_diff": sorted({int(f) for f in fa[np.abs(pa - pb) > 1e-6]})}
json.dump(out, open(f"{C}/verify/compare_c25.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
