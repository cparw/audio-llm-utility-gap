"""PART 26: Qwen2-Audio on ADReSSo. Paired probe minus zero-shot answer gap on the paper's estimator (mean of the
five per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART 26 CPU pod run of
p25a_nested5.py (byte copy of the PART 25 A script). Same bootstrap code path as part25/A/scripts/A_gaps.py:
2000 draws, a fresh numpy default_rng(0); each draw resamples the unique speakers (np.unique order) with replacement,
idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every drawn speaker. In each draw the five
per-repeat AUCs (sklearn roc_auc_score) are averaged and the zero-shot answer AUC on the same rows is subtracted.
2.5 and 97.5 percentiles over the usable draws.
Zero-shot per-clip file: paper1_local_runs/probe2/adresso_zeroshot_scores.csv (read only; C_build_cells.csv row).
Writes under part26/Qwen2-Audio_ADReSSo/: per_clip/Qwen2-Audio_ADReSSo_perclip.csv, the sidecar json next to it,
draws/Qwen2-Audio_ADReSSo_draws.npz.   usage: /usr/local/bin/python3 gap26.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

C = "scores/part26/Qwen2-Audio_ADReSSo"
POD = f"{C}/pull/p26-q2a-adresso"
OOF = f"{POD}/out/p26_q2a_adresso_enc_nested5_oof.csv"
PJ = f"{POD}/out/p26_q2a_adresso_enc_nested5.json"
ZS = "<local data dir>/paper1_local_runs/probe2/adresso_zeroshot_scores.csv"
STATES = "<local data dir>/paper1_local_runs/probe2/adresso_states.npz"
SLIM = f"{C}/podfiles/q2a_adresso_enc_states.npz"
T1 = "<local data dir>/paper1_local_runs/omni_final/q2a_adresso_nested_repeats.json"
P25C = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-3/out/q2a_adresso_enc_rep5_oof.csv"
FOLDS = "<local data dir>/release/folds"
NB = 2000
ZS_AUC_EXPECTED = 0.6649

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

J = json.load(open(PJ))
rows = list(csv.DictReader(open(OOF)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
assert np.isfinite(P).all() and np.isfinite(Ps).all()

zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and zmap[c]["speaker"] == s for c, l, s in zip(names, y, spk)), "labels or speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])

# fold ids in the pod output equal the saved fold files, clip by clip
fold_check = {}
for s in range(5):
    ff = f"{FOLDS}/adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    fr = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(ff))}
    fold_check[f"seed{s}"] = {"file": ff, "sha256": sha(ff), "equal": all(fr[r["clip"]] == int(r[f"fold_seed{s}"]) for r in rows)}
assert all(v["equal"] for v in fold_check.values()), "pod fold ids differ from the saved fold files"

T = json.load(open(T1))["enc"]
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
assert abs(zauc - ZS_AUC_EXPECTED) < 5e-5, zauc
B = boot(y, spk, P, zs)

# PART 25 C Linux refit of the same estimator (sklearn GroupKFold on the pod), for comparison only
c25 = {r["clip"]: r for r in csv.DictReader(open(P25C))}
c25_maxabs = float(max(abs(float(c25[c][f"p_seed{s}"]) - P[s][i]) for s in range(5) for i, c in enumerate(names)))
c25_folds_equal = all(int(c25[c][f"fold_seed{s}"]) == int(rows[i][f"fold_seed{s}"]) for s in range(5) for i, c in enumerate(names))

os.makedirs(f"{C}/per_clip", exist_ok=True); os.makedirs(f"{C}/draws", exist_ok=True)
DR = f"{C}/draws/Qwen2-Audio_ADReSSo_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc))
PC = f"{C}/per_clip/Qwen2-Audio_ADReSSo_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])

rec = {"task": "PART 26 probe grid: Qwen2-Audio encoder nested probe on ADReSSo, paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "model": "Qwen2-Audio", "dataset": "ADReSSo", "condition": "AD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "nested encoder probe: outer folds = the five saved repeat fold files; inner GroupKFold(4) (pod tie rule) picks the "
                              "layer by mean inner AUC inside each outer training set; StandardScaler + LogisticRegression(max_iter=2000, "
                              "class_weight='balanced') refit at that layer; Table 1 value = mean of the five per-repeat out-of-fold AUCs",
                 "script": "p25a_nested5.py (sha256 " + sha(f"{C}/podfiles/p25a_nested5.py") + ", byte identical to part25/A/podfiles/p25a_nested5.py)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc": float(roc_auc_score(y, Ps)), "single_split_layers": J["tags"]["single"]["layers"],
                 "table1_per_repeat": T["per_repeat"], "table1_mean": T["mean"],
                 "matches_table1_4dp": [round(a, 4) for a in per] == T["per_repeat"] and round(m5, 4) == T["mean"],
                 "fold_files_equal_pod_output": fold_check,
                 "fold_files_equal_pod_tie_rule": {t: J["tags"][t]["fold_file_equals_pod_tie_rule"] for t in J["tags"]},
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "pod_command": J["command"],
                 "states_used_on_pod": J["states"],
                 "p25C_refit_comparison": {"file": P25C, "folds_equal": c25_folds_equal, "max_abs_p_diff": c25_maxabs}},
       "zeroshot": {"file": ZS, "sha256": sha(ZS), "auc": zauc, "auc_4dp": round(zauc, 4), "score_column": "p_yes"},
       "gap": {"point": m5 - zauc, "ci_2p5_97p5": list(B["ci_diff"]), "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0), "p_boot_le0": B["p_le0"],
               "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"], "total_draws": NB},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC, same rows in the same draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "versions_mac": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "states_source": STATES, "states_uploaded": SLIM,
                 "table1_json": T1, "gap_script": os.path.abspath(__file__)},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "states_source": sha(STATES), "states_uploaded": sha(SLIM),
                  "table1_json": sha(T1), "perclip_csv": sha(PC), "draws_npz": sha(DR)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/per_clip/Qwen2-Audio_ADReSSo_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen2-Audio ADReSSo probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (Table 1 {T['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}  P25C max|dp| {c25_maxabs:.2e} folds_equal {c25_folds_equal}")
