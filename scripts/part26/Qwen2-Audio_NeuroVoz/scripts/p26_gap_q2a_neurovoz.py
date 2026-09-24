"""PART26 Qwen2-Audio NeuroVoz: paired probe-minus-answer gap on the paper's estimator (mean of the five per-repeat
nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod run of p25a_nested5.py
(PART25 A script, unchanged, on the saved release fold files). Same bootstrap code path as part25/A/scripts/A_gaps.py.

Bootstrap: 2000 draws, a fresh numpy default_rng(0); each draw resamples the unique speakers (np.unique order of the
states spk, 107 speakers) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of
every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged, the zero-shot answer
AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot scores: paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv (read only), the file named for this cell in
part25/C/C_build_cells.csv and C_other_models.tsv.
Writes, under part26/Qwen2-Audio_NeuroVoz/: per_clip/p26_q2a_neurovoz_perclip.csv, per_clip/p26_q2a_neurovoz_perclip.sidecar.json,
draws/p26_q2a_neurovoz_draws.npz.   usage: p26_gap_q2a_neurovoz.py PULLDIR"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen2-Audio_NeuroVoz"
PULL = sys.argv[1]
ZS = "<local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv"
ML = "<local data dir>/paper1_local_runs/omni_final/q2a_neurovoz_nested_repeats.json"
STATES = "<local data dir>/paper1_local_runs/probe2/neurovoz_states.npz"
STAGED = f"{R}/stage/q2a_neurovoz_encstates.npz"
NB = 2000

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
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb))

oofp = f"{PULL}/out/p26_q2a_neurovoz_enc_nested5_oof.csv"; jp = f"{PULL}/out/p26_q2a_neurovoz_enc_nested5.json"
vj = f"{PULL}/out/p26_q2a_neurovoz_podverify.json"
J = json.load(open(jp)); V = json.load(open(vj))
rows = list(csv.DictReader(open(oofp)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
T = json.load(open(ML))["enc"]
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
dr = f"{R}/draws/p26_q2a_neurovoz_draws.npz"
np.savez_compressed(dr, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"],
                    speaker_idx=B["idx"], speakers=B["speakers"], point_diff=np.array(m5 - zauc),
                    point_diff_master=np.array(T["mean"] - zauc))
pc = f"{R}/per_clip/p26_q2a_neurovoz_perclip.csv"
with open(pc, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], zmap[r["clip"]]["p_yes"]])
rec = {"task": "PART26 encoder probe grid, cell Qwen2-Audio x NeuroVoz: nested five-repeat encoder probe (layer chosen inside the training folds), paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "model": "Qwen2-Audio", "dataset": "NeuroVoz", "condition": "PD", "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_4dp": [round(a, 4) for a in per], "probe_mean5_4dp": round(m5, 4),
       "probe_per_repeat_master": T["per_repeat"], "probe_mean5_master": T["mean"],
       "matches_master_4dp": round(m5, 4) == T["mean"] and [round(a, 4) for a in per] == T["per_repeat"],
       "zeroshot_auc": zauc, "zeroshot_auc_4dp": round(zauc, 4),
       "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_point_master": T["mean"] - zauc,
       "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
       "single_split_auc": float(roc_auc_score(y, Ps)),
       "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)}, "layers_single": J["tags"]["single"]["layers"],
       "fold_files": {t: {"file": J["tags"][t]["fold_file"], "sha256": J["tags"][t]["fold_file_sha256"],
                          "equals_pod_tie_rule": J["tags"][t]["fold_file_equals_pod_tie_rule"],
                          "equals_pod_sklearn_GroupKFold": J["tags"][t]["fold_file_equals_this_pod_sklearn"]} for t in J["tags"]},
       "fold_note": "outer folds read from release/folds/neurovoz_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv (the Table 1 five-repeat splits; UNVERIFIED means no per-clip OOF ever existed to confirm them clip by clip)",
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order of states spk), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot AUC, same rows in each draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score", "code_path": "part25/A/scripts/A_gaps.py boot()"},
       "estimator": {"script": "part25/A/podfiles/p25a_nested5.py (unchanged, md5 615ced0380cca2636ed7cc30c71810e6)",
                     "outer": "saved fold files, GroupKFold(5) on speakers renumbered by default_rng(seed)",
                     "inner": "GroupKFold(4) pod tie rule on the outer training speakers, layer = argmax mean inner AUC over 33 encoder stages",
                     "model": "StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced')", "blas_threads_per_fit": 1},
       "pod": {"name": "p26-q2a-neurovoz", "id": "md42nsos0qf0ei", "cpuFlavor": "cpu3c", "cpu": "AMD EPYC 9654 96-Core Processor (avx512f present)",
               "versions": J["versions"], "seconds": J["seconds"], "n_jobs": J["n_jobs"]},
       "podverify": {"PASS": V.get("PASS"), "max_abs_pred_diff": V.get("max_abs_pred_diff"), "max_abs_inner_score_diff": V.get("max_abs_inner_score_diff"),
                     "layer_mismatches": V.get("layer_mismatches"), "abs_mean5_diff": V.get("abs_mean5_diff")},
       "files": {"perclip_csv": pc, "draws_npz": dr, "pod_oof_csv": oofp, "pod_json": jp, "pod_verify_json": vj, "zeroshot": ZS,
                 "master_repeats_json": ML, "states_source": STATES, "states_staged": STAGED},
       "sha256": {"pod_oof_csv": sha(oofp), "pod_json": sha(jp), "zeroshot": sha(ZS), "master_repeats_json": sha(ML),
                  "states_staged": sha(STAGED), "perclip_csv": sha(pc), "draws_npz": sha(dr)},
       "versions_mac": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{R}/per_clip/p26_q2a_neurovoz_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen2-Audio NeuroVoz probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (master {T['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}  podverify {V.get('PASS')}", flush=True)
