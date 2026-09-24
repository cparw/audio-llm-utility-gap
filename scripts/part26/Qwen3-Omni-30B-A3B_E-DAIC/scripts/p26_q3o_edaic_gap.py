"""PART 26, Qwen3-Omni-30B-A3B on E-DAIC (30 s window, MDD): paired probe-minus-answer gap on the paper's estimator.
Probe: nested five-repeat encoder probe (layer chosen inside the training folds by mean inner AUC), run on a Linux CPU
pod by p25a_nested5.py (PART 25 A script, unchanged) on states re-extracted in PART 26 with podD2 extract_probe_layers.py.
Answer: the q3o_new zero-shot file (master_lookup row 147, AUC 0.6290), read only.
Bootstrap: 2000 draws, a fresh numpy default_rng(0); unique speakers in np.unique order; idx = rng.choice(n_spk,
size=n_spk, replace=True); every clip of every drawn speaker; in each draw the mean of the five per-repeat AUCs minus the
zero-shot AUC on the same rows (sklearn roc_auc_score); 2.5 and 97.5 percentiles. Same code path as PART 25 A_gaps.py
and PART 25 C_build.py.
Writes Q3O_EDAIC_perclip.csv, Q3O_EDAIC_perclip.sidecar.json, Q3O_EDAIC_draws.npz under the cell folder.
usage: /usr/local/bin/python3 p26_q3o_edaic_gap.py CPU_PULL_DIR GPU_PULL_DIR"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn, scipy
from sklearn.metrics import roc_auc_score
R = "scores/part26/Qwen3-Omni-30B-A3B_E-DAIC"
CPU, GPU = sys.argv[1:3]
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_edaic_zeroshot_scores.csv"
OLDREP = "<local data dir>/release/overnight2/final/q3o_edaic_nested_repeats.json"
OLDSS = "<local data dir>/release/overnight2/q3o_new/q3o_edaic_enc_nested_oof.csv"
OLDSSJ = "<local data dir>/release/overnight2/q3o_new/q3o_edaic_nested.json"
FOLDS = "<local data dir>/release/folds"
NB = 2000
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
oofp = f"{CPU}/out/p26_q3o_edaic_enc_nested5_oof.csv"; jp = f"{CPU}/out/p26_q3o_edaic_enc_nested5.json"; vp = f"{CPU}/out/p26_q3o_edaic_podverify.json"
J = json.load(open(jp)); V = json.load(open(vp))
rows = list(csv.DictReader(open(oofp))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
Z = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(Z) == n and set(Z) == set(names), "zero-shot clip set differs"
assert all(int(Z[c]["label"]) == int(l) and Z[c]["speaker"] == s for c, l, s in zip(names, y, spk)), "labels or speakers differ"
zs = np.array([float(Z[c]["p_yes"]) for c in names])
RZ = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{GPU}/out/p26_q3o_edaic_zeroshot_scores.csv"))}
zs_rerun = np.array([RZ[c] for c in names])
SS = {r["clip"]: float(r["p_probe"]) for r in csv.DictReader(open(OLDSS))}; ps_old = np.array([SS[c] for c in names])
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
u = np.unique(spk); by = [np.where(spk == s)[0] for s in u]; nsp = len(u)
rng = np.random.default_rng(0)
rep = np.full((NB, 5), np.nan); zb = np.full(NB, np.nan); idxs = np.zeros((NB, nsp), np.int32)
for b in range(NB):
    idx = rng.choice(nsp, size=nsp, replace=True); idxs[b] = idx
    ii = np.concatenate([by[i] for i in idx]); yy = y[ii]
    if yy.min() == yy.max(): continue
    rep[b] = [roc_auc_score(yy, P[r][ii]) for r in range(5)]; zb[b] = roc_auc_score(yy, zs[ii])
ok = ~np.isnan(zb); mb = rep.mean(1); d = mb - zb
pc = lambda v: [float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5))]
ci, cim, ciz = pc(d), pc(mb), pc(zb)
dr = f"{R}/Q3O_EDAIC_draws.npz"
np.savez_compressed(dr, diff=d, mean5=mb, zeroshot=zb, per_repeat=rep, speaker_idx=idxs, speakers=u, point_diff=np.array(m5 - zauc),
                    point_probe_mean5=np.array(m5), point_zeroshot=np.array(zauc), seed=np.array(0), n_boot=np.array(NB))
fold = {t: {r["clip_id"]: r["fold"] for r in csv.DictReader(open(f"{FOLDS}/{J['tags'][t]['fold_file']}"))} for t in J["tags"]}
pcf = f"{R}/Q3O_EDAIC_perclip.csv"
with open(pcf, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_rerun", "fold_single", "p_probe_single_podD2_saved", "p_yes_zeroshot", "p_yes_rerun_extraction"])
    for i, r in enumerate(rows):
        for t in J["tags"]: assert fold[t][r["clip"]] == r[f"fold_{t}"], (t, r["clip"])
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(ps_old[i])), repr(float(zs[i])), repr(float(zs_rerun[i]))])
OLD = json.load(open(OLDREP))["enc"]; OLDS = json.load(open(OLDSSJ))["enc"]
rho = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
PY = json.load(open(f"{GPU}/out/p26_pyes_check.json"))
rec = {"cell": "Qwen3-Omni-30B-A3B on E-DAIC", "model": "Qwen3-Omni-30B-A3B", "checkpoint": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
       "dataset": "E-DAIC", "condition": "MDD", "window": "first 30 s of dcaps_proc/<pid>.wav (x[:16000*30] in the extractor)",
       "n": n, "n_speakers": nsp, "n_pos": int(y.sum()),
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_4dp": [round(a, 4) for a in per], "probe_mean5_4dp": round(m5, 4),
       "zeroshot_auc": zauc, "zeroshot_auc_4dp": round(zauc, 4), "gap_point": m5 - zauc, "gap_ci": ci,
       "gap_point_4dp": round(m5 - zauc, 4), "gap_ci_4dp": [round(v, 4) for v in ci],
       "interval_above_zero": bool(ci[0] > 0), "interval_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
       "p_boot_one_sided_le0": float(np.mean(d[ok] <= 0)),
       "probe_mean5_ci": cim, "zeroshot_ci": ciz, "usable_draws": int(ok.sum()), "n_draws": NB,
       "layers": {t: J["tags"][t]["layers"] for t in J["tags"]},
       "single_split_auc_rerun": float(roc_auc_score(y, Ps)), "single_split_auc_podD2_saved": float(roc_auc_score(y, ps_old)),
       "single_split_layers_rerun": J["tags"]["single"]["layers"], "single_split_layers_podD2_saved": OLDS["chosen_layers"],
       "single_split_rank_corr_rerun_vs_podD2": rho(Ps, ps_old),
       "old_podD2_per_repeat_master_row144": OLD["per_repeat"], "old_podD2_mean5_master_row144": OLD["mean"],
       "mean5_minus_old_master": m5 - OLD["mean"],
       "zeroshot_rerun_extraction_auc": float(roc_auc_score(y, zs_rerun)),
       "zeroshot_rerun_vs_saved_max_abs_p_yes_diff": float(np.max(np.abs(zs_rerun - zs))),
       "zeroshot_rerun_vs_saved_median_abs_p_yes_diff": float(np.median(np.abs(zs_rerun - zs))),
       "pod_pyes_check": PY, "podverify": {k: V.get(k) for k in ("PASS", "max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "mean5_rank_from_oof", "abs_mean5_diff")},
       "probe_pod_versions": J["versions"], "probe_n_jobs": J["n_jobs"], "probe_command": J["command"], "states_on_pod": J["states"],
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh", "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs and the zero-shot AUC on the same draw", "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "files": {"perclip_csv": pcf, "draws_npz": dr, "pod_oof_csv": oofp, "pod_json": jp, "pod_verify_json": vp, "zeroshot": ZS,
                 "rerun_zeroshot": f"{GPU}/out/p26_q3o_edaic_zeroshot_scores.csv", "rerun_encstates": f"{GPU}/out/p26_q3o_edaic_encstates.npz",
                 "rerun_states_full": f"{GPU}/out/p26_q3o_edaic_states.npz", "old_master_json": OLDREP, "old_single_split_oof": OLDSS},
       "sha256": {k: sha(v) for k, v in {"pod_oof_csv": oofp, "pod_json": jp, "zeroshot": ZS, "perclip_csv": pcf, "draws_npz": dr,
                                         "rerun_encstates": f"{GPU}/out/p26_q3o_edaic_encstates.npz"}.items()},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join([os.path.abspath(__file__)] + sys.argv[1:]),
       "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(rec, open(f"{R}/Q3O_EDAIC_perclip.sidecar.json", "w"), indent=1)
print(f"Q3O E-DAIC probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} answer {zauc:.4f} gap {m5 - zauc:+.4f} "
      f"[{ci[0]:+.4f}, {ci[1]:+.4f}] usable {int(ok.sum())} (old master mean5 {OLD['mean']}) single {rec['single_split_auc_rerun']:.4f} vs podD2 {rec['single_split_auc_podD2_saved']:.4f}")
