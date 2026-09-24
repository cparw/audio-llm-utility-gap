"""PART 26 Kimi-Audio E-DAIC independent check. Written separately from p26_kimi_edaic_gap.py and shares no code with it.
- AUC by the Mann-Whitney rank formula (scipy.stats.rankdata, ties averaged), not sklearn.
- joins by python dicts on the clip name, not a pandas merge; reads the pod OOF file and the zero-shot file directly.
- the fold columns of the OOF file are compared with the release fold files, read again here by clip.
- bootstrap re-implemented: sorted unique speakers, fresh numpy default_rng(0), idx = rng.choice(n_spk, n_spk, replace=True),
  all clips of each drawn speaker, mean of the five per-repeat AUCs minus the zero-shot AUC, 2.5/97.5 percentiles.
- compares every number with the sidecar and the saved draws (max abs difference).
- states checks: the re-extracted p_yes against the canonical zero-shot file; the re-extracted Kimi LM states against the
  earlier Kimi E-DAIC LM states (overnight2/kimi/kimi_edaic_states.npz, the run behind the zero-shot file); encoder states
  shape, finite, frames kept; the enc array the CPU pod probed equals the enc array in the full GPU states file.
- a Mac refit of the five seed0 outer folds at the pod's chosen layers (StandardScaler + LogisticRegression(max_iter=2000,
  class_weight='balanced'), Mac sklearn), compared with the pod OOF column.
usage: /usr/local/bin/python3 p26_kimi_edaic_verify.py OOF_CSV STATES_NPZ ENCSTATES_NPZ"""
import os, sys, csv, json, datetime
import numpy as np
from scipy.stats import rankdata
C = "scores/part26/Kimi-Audio_E-DAIC"
ZS = "<local data dir>/release/overnight2/part3/kimi_edaic_zeroshot_scores.csv"
OLD = "<local data dir>/release/overnight2/kimi/kimi_edaic_states.npz"
FOLDS = "<local data dir>/release/folds"
OOF, NPZ, ENPZ = sys.argv[1:4]
S = json.load(open(f"{C}/p26_kimi_edaic_enc_perclip.sidecar.json"))

def mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

rec = {}
zd = {r["clip"]: (int(r["label"]), float(r["p_yes"]), r["speaker"]) for r in csv.DictReader(open(ZS))}
rows = list(csv.DictReader(open(OOF)))
names = [r["clip"] for r in rows]
assert set(names) == set(zd) and len(names) == 275
y = np.array([zd[n][0] for n in names]); rec["labels_agree"] = all(int(r["label"]) == zd[r["clip"]][0] for r in rows)
rec["speakers_agree"] = all(r["speaker"] == zd[r["clip"]][2] for r in rows)
spk = np.array([r["speaker"] for r in rows])
Sx = np.array([[float(r[f"p_seed{k}"]) for k in range(5)] for r in rows]); zs = np.array([zd[n][1] for n in names])
# fold columns vs release fold files
fold_ok = {}
for k in range(5):
    f = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDS}/edaic_groupkfold5_pod_seed{k}_UNVERIFIED.csv"))}
    fold_ok[f"seed{k}"] = all(int(r[f"fold_seed{k}"]) == f[r["clip"]] for r in rows)
f = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDS}/edaic_groupkfold5_pod.csv"))}
fold_ok["single"] = all(int(r["fold_single"]) == f[r["clip"]] for r in rows)
rec["fold_columns_equal_release_files"] = fold_ok
per = [mw(y, Sx[:, k]) for k in range(5)]; probe = float(np.mean(per)); ans = mw(y, zs); point = probe - ans
rec.update(per_repeat_mw=per, probe_mw=probe, answer_mw=ans, gap_mw=point)
u = sorted(set(spk)); pos = {s: [i for i in range(len(spk)) if spk[i] == s] for s in u}
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    ii = np.array([i for j in rng.choice(len(u), size=len(u), replace=True) for i in pos[u[j]]])
    if y[ii].min() == y[ii].max(): d.append(np.nan); continue
    d.append(float(np.mean([mw(y[ii], Sx[ii, k]) for k in range(5)])) - mw(y[ii], zs[ii]))
d = np.array(d); ok = ~np.isnan(d); lo, hi = [float(v) for v in np.percentile(d[ok], [2.5, 97.5])]
rec.update(gap_lo_v=lo, gap_hi_v=hi, usable_v=int(ok.sum()))
rec["agree_4dp"] = {"probe": bool(round(probe, 4) == round(S["probe_auc_mean5"], 4)), "answer": bool(round(ans, 4) == round(S["answer_auc"], 4)),
                    "gap": bool(round(point, 4) == round(S["gap"], 4)), "lo": bool(round(lo, 4) == round(S["gap_ci95"][0], 4)),
                    "hi": bool(round(hi, 4) == round(S["gap_ci95"][1], 4)),
                    "per_repeat": [round(a, 4) for a in per] == S["per_repeat_auc_4dp"]}
Dz = np.load(S["files"]["draws_npz"]); rec["draws_maxabs"] = float(np.nanmax(np.abs(Dz["diff"] - d)))
rec["draws_nan_same"] = bool(np.array_equal(np.isnan(Dz["diff"]), np.isnan(d)))
# per-clip file written by the build equals the pod OOF columns exactly
pc = {r["clip"]: r for r in csv.DictReader(open(S["files"]["perclip_csv"]))}
rec["perclip_equals_pod_oof"] = all(float(pc[r["clip"]][f"p_seed{k}"]) == float(r[f"p_seed{k}"]) for r in rows for k in range(5)) and \
    all(float(pc[n]["p_yes_zeroshot"]) == zd[n][1] for n in names)
# states checks
z = np.load(NPZ, allow_pickle=True); ze = np.load(ENPZ, allow_pickle=True); old = np.load(OLD, allow_pickle=True)
zn = [str(v) for v in z["name"]]
assert zn == [str(v) for v in old["name"]] == [str(v) for v in ze["name"]] == names
E = z["enc"]; rec["enc_shape"] = list(E.shape); rec["enc_dtype"] = str(E.dtype); rec["enc_finite"] = bool(np.isfinite(E).all())
rec["enc_frames"] = sorted(set(int(v) for v in z["enc_frames"]))
rec["enc_equal_probed_array"] = bool(np.array_equal(E, ze["enc"]))
rec["enc_layer_distinct"] = bool(all(not np.allclose(E[:, i], E[:, i + 1]) for i in range(E.shape[1] - 1)))
dp = np.abs(z["p_yes"].astype(float) - zs); rec["pyes_vs_zeroshot"] = {"max_abs": float(dp.max()), "median_abs": float(np.median(dp)),
    "pearson": float(np.corrcoef(z["p_yes"].astype(float), zs)[0, 1]), "auc_rerun_mw": mw(y, z["p_yes"].astype(float))}
L, Lo = z["llm"].astype(np.float64), old["llm"].astype(np.float64)
rel = np.linalg.norm(L - Lo, axis=2) / np.linalg.norm(Lo, axis=2)
cos = (L * Lo).sum(2) / (np.linalg.norm(L, axis=2) * np.linalg.norm(Lo, axis=2))
rec["llm_vs_earlier_run"] = {"max_rel_l2": float(rel.max()), "median_rel_l2": float(np.median(rel)), "min_cosine": float(cos.min()),
                             "max_abs": float(np.abs(L - Lo).max())}
# Mac refit of seed0 outer folds at the chosen layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X = E.astype(np.float32); yy = z["label"].astype(int)
lay = S["layers_chosen"]["seed0"]; fs0 = np.array([int(r["fold_seed0"]) for r in rows]); ref = np.zeros(len(yy))
for k in range(5):
    tr, te = np.where(fs0 != k)[0], np.where(fs0 == k)[0]
    sc = StandardScaler().fit(X[tr][:, lay[k]])
    ref[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr][:, lay[k]]), yy[tr]).predict_proba(sc.transform(X[te][:, lay[k]]))[:, 1]
rec["mac_refit_seed0"] = {"layers": lay, "max_abs_vs_pod_oof": float(np.abs(ref - Sx[:, 0]).max()), "auc_mac_refit": mw(y, ref)}
allok = (rec["labels_agree"] and rec["speakers_agree"] and all(fold_ok.values()) and all(rec["agree_4dp"].values()) and
         rec["draws_maxabs"] < 1e-9 and rec["draws_nan_same"] and rec["perclip_equals_pod_oof"] and rec["enc_finite"] and
         rec["enc_equal_probed_array"] and rec["enc_shape"] == [275, 32, 1280] and rec["enc_frames"] == [1500])
rec["ALL_CHECKS_PASS"] = bool(allok)
rec["created_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); rec["script"] = os.path.abspath(__file__); rec["method"] = __doc__
os.makedirs(f"{C}/verify", exist_ok=True)
json.dump(rec, open(f"{C}/verify/p26_kimi_edaic_verify.json", "w"), indent=1, default=float)
print(json.dumps({k: rec[k] for k in ("agree_4dp", "draws_maxabs", "fold_columns_equal_release_files", "pyes_vs_zeroshot", "llm_vs_earlier_run",
                                      "mac_refit_seed0", "enc_shape", "enc_frames", "ALL_CHECKS_PASS")}, indent=1, default=float))
print(f"VERIFY probe {probe:.4f} answer {ans:.4f} gap {point:+.4f} [{lo:+.4f}, {hi:+.4f}] ALL_CHECKS_PASS {allok}")
