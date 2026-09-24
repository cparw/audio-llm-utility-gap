"""PART26 Qwen2-Audio on Pitt: paired probe-minus-answer gap on the paper's estimator (mean of five per-repeat nested
encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod run (p25a_nested5.py, unchanged
PART25 A script, saved fold files pitt_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv).
Bootstrap (same code path as part25/A/scripts/A_gaps.py): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order, 228 labels) with replacement, idx = rng.choice(n_spk, size=n_spk,
replace=True), and takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn
roc_auc_score) are averaged, the zero-shot answer AUC is computed on the same rows, and the difference is taken.
2.5 and 97.5 percentiles. Zero-shot: paper1_local_runs/probe2/pitt_zeroshot_scores.csv (read only).
Writes per_clip/q2a_pitt_perclip.csv, per_clip/q2a_pitt_perclip.sidecar.json, draws/q2a_pitt_draws.npz."""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

C = "scores/part26/Qwen2-Audio_Pitt"
POD = f"{C}/pull/p26-q2a-pitt"
OOF = f"{POD}/out/p26_q2a_pitt_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_q2a_pitt_enc_nested5.json"
PV = f"{POD}/out/p26_q2a_pitt_podverify.json"
ZS = "<local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv"
MJ = "<local data dir>/paper1_local_runs/omni_final/q2a_pitt_nested_repeats.json"
C25 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q2a_pitt_enc_rep5_oof.csv"
STAGE = f"{C}/stage/stage_check.json"
NB = 2000
os.makedirs(f"{C}/per_clip", exist_ok=True); os.makedirs(f"{C}/draws", exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot(y, spk, P, zs):
    u = np.unique(spk); rows = [np.where(spk == s)[0] for s in u]; nsp = len(u)
    rng = np.random.default_rng(0)
    rep = np.full((NB, P.shape[0]), np.nan); zb = np.full(NB, np.nan); idxs = np.zeros((NB, nsp), np.int32)
    for b in range(NB):
        idx = rng.choice(nsp, size=nsp, replace=True); idxs[b] = idx
        ii = np.concatenate([rows[i] for i in idx]); yy = y[ii]
        if yy.min() == yy.max(): continue
        rep[b] = [roc_auc_score(yy, P[r][ii]) for r in range(P.shape[0])]
        zb[b] = roc_auc_score(yy, zs[ii])
    ok = ~np.isnan(zb)
    m5 = rep.mean(axis=1); d = m5 - zb
    pc = lambda v: (float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5)))
    return dict(rep=rep, zs=zb, mean5=m5, diff=d, idx=idxs, usable=int(ok.sum()), speakers=u,
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb), p_le0=float(np.mean(d[ok] <= 0)))

J = json.load(open(PJ)); V = json.load(open(PV)); M = json.load(open(MJ))["enc"]; S = json.load(open(STAGE))
rows = list(csv.DictReader(open(OOF)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and zmap[c]["speaker"] == s for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
assert all(abs(a - b) < 1e-12 for a, b in zip(per, J["per_repeat_auc"]))
B = boot(y, spk, P, zs)
DR = f"{C}/draws/q2a_pitt_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_diff_master=np.array(M["mean"] - zauc))
# PART25 C Linux refit (pod sklearn GroupKFold, not the saved fold files), for comparison only
c25 = {r["clip"]: r for r in csv.DictReader(open(C25))}
P25 = np.array([[float(c25[c][f"p_seed{s}"]) for c in names] for s in range(5)])
F25 = np.array([[int(c25[c][f"fold_seed{s}"]) for c in names] for s in range(5)])
F26 = np.array([[int(r[f"fold_seed{s}"]) for r in rows] for s in range(5)])
PC = f"{C}/per_clip/q2a_pitt_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
rec = {"task": "PART26 probe grid: Qwen2-Audio encoder nested probe on Pitt, paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z", "model": "Qwen2-Audio (Qwen2-Audio-7B-Instruct)", "dataset": "Pitt",
       "condition": "AD", "n": n, "n_pos": int(y.sum()), "n_speakers": int(len(np.unique(spk))),
       "speaker_note": "228 speaker labels cover 227 people: participant 172 is labeled both Control172 and Dementia172, kept as the runs did",
       "probe": {"estimator": "nested encoder probe, outer folds from the saved fold files, inner GroupKFold(4) with the stable tie rule picks the layer by mean inner AUC, StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), mean of the five per-repeat out-of-fold AUCs",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc": J["single_split_auc"], "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files": {t: J["tags"][t]["fold_file"] for t in J["tags"]},
                 "fold_file_sha256": {t: J["tags"][t]["fold_file_sha256"] for t in J["tags"]},
                 "fold_file_equals_pod_tie_rule": all(J["tags"][t]["fold_file_equals_pod_tie_rule"] for t in J["tags"]),
                 "fold_file_equals_this_pod_sklearn": {t: J["tags"][t]["fold_file_equals_this_pod_sklearn"] for t in J["tags"]},
                 "n_clips_sklearn_differs": {t: J["tags"][t]["n_clips_sklearn_differs"] for t in J["tags"]},
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "pod_n_jobs": J["n_jobs"],
                 "pod_cpu": "AMD EPYC 7702P, cpu3c 32 vCPU, avx512f absent", "podverify_pass": V["PASS"],
                 "podverify": {k: V[k] for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff")}},
       "master_lookup_probe": {"file": MJ, "per_repeat": M["per_repeat"], "mean": M["mean"],
                               "mean5_minus_master": m5 - M["mean"], "matches_master_4dp": round(m5, 4) == M["mean"],
                               "per_repeat_abs_diff_4dp": [round(abs(round(a, 4) - b), 4) for a, b in zip(per, M["per_repeat"])]},
       "part25C_refit_comparison": {"file": C25, "per_repeat_auc": [float(roc_auc_score(y, P25[s])) for s in range(5)],
                                    "folds_equal_part26_per_seed": [bool(np.array_equal(F25[s], F26[s])) for s in range(5)],
                                    "max_abs_pclip_diff_per_seed": [float(np.max(np.abs(P25[s] - P[s]))) for s in range(5)]},
       "zeroshot": {"file": ZS, "sha256": sha(ZS), "auc": zauc, "auc_4dp": round(zauc, 4),
                    "equals_states_p_yes": S["zeroshot"]["p_yes_max_abs_diff_vs_states"] == 0.0},
       "gap": {"point": m5 - zauc, "point_4dp": round(m5 - zauc, 4), "ci_2.5": B["ci_diff"][0], "ci_97.5": B["ci_diff"][1],
               "ci_4dp": [round(B["ci_diff"][0], 4), round(B["ci_diff"][1], 4)], "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "p_boot_diff_le0": B["p_le0"], "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]),
               "usable_draws": B["usable"], "point_master_mean_minus_answer": M["mean"] - zauc},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "probe (mean of the five per-repeat AUCs) and zero-shot AUC on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score"},
       "states": {"source": S["source"], "source_sha256": S["source_sha256"], "probe_input": S["stage"], "probe_input_sha256": S["stage_sha256"],
                  "probe_input_arrays_equal_source": S["arrays_equal_source"], "keys": S["keys"]},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV,
                 "pod_tar": f"{C}/pull/p26-q2a-pitt.tar", "script": os.path.abspath(__file__)},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "perclip_csv": sha(PC), "draws_npz": sha(DR), "master_json": sha(MJ)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/per_clip/q2a_pitt_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen2-Audio Pitt: probe mean5 {m5:.4f} per {[round(a,4) for a in per]} (master {M['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}]  usable {B['usable']}  "
      f"P25C folds equal {rec['part25C_refit_comparison']['folds_equal_part26_per_seed']}", flush=True)
