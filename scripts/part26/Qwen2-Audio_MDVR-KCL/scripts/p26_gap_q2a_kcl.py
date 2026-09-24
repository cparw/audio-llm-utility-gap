"""PART26: Qwen2-Audio, MDVR-KCL. Paired probe minus zero-shot answer gap on the paper's estimator (mean of five
per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors the PART26 CPU pod saved (p25a_nested5.py,
the PART25 A script, unchanged) on the five saved pod fold files.
Bootstrap (same as part25/A/scripts/A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw resamples the
unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every
clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged, the zero-shot
answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot: paper1_local_runs/probe2/kcl_zeroshot_scores.csv (read only).
Writes Q2A_kcl_perclip.csv, Q2A_kcl_perclip.sidecar.json, Q2A_kcl_draws.npz in the cell folder."""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen2-Audio_MDVR-KCL"
POD = f"{R}/pull/p26-q2a-kcl"
OOF = f"{POD}/out/p26_q2a_kcl_enc_nested5_oof.csv"; JS = f"{POD}/out/p26_q2a_kcl_enc_nested5.json"
PV = f"{POD}/out/p26_q2a_kcl_podverify.json"
ZS = "<local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv"
STATES = "<local data dir>/paper1_local_runs/probe2/kcl_states.npz"
T1 = "<local data dir>/release/omni_final/q2a_kcl_nested_repeats.json"
FOLDS = "<local data dir>/release/folds"
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

J = json.load(open(JS)); V = json.load(open(PV)) if os.path.exists(PV) else {}
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
T = json.load(open(T1))["enc"]
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{R}/Q2A_kcl_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_diff_table1=np.array(T["mean"] - zauc))
PC = f"{R}/Q2A_kcl_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
rec = {"task": "PART26 probe grid: Qwen2-Audio encoder nested probe (layer chosen inside the training folds), five repeats on the saved fold files, paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "model": "Qwen2-Audio-7B-Instruct", "dataset": "MDVR-KCL", "condition": "PD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()), "n_layers": J["n_layers"],
       "probe": {"estimator": "outer folds = saved pod fold files seed0..4; inner GroupKFold(4) with the pod tie rule on the outer training speakers scores every encoder stage (33: input stage + 32 layers) by mean inner AUC; argmax; refit StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'); score held-out clips. Probe = mean over the five repeats of the out-of-fold AUC.",
                 "script": "part25/A/podfiles/p25a_nested5.py (unchanged, md5 615ced0380cca2636ed7cc30c71810e6)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc": J["single_split_auc"], "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files": {t: J["tags"][t]["fold_file"] for t in J["tags"]},
                 "fold_file_sha256": {t: J["tags"][t]["fold_file_sha256"] for t in J["tags"]},
                 "table1_saved_per_repeat": T["per_repeat"], "table1_saved_mean": T["mean"],
                 "per_repeat_abs_diff_vs_table1_4dp": [abs(round(a, 4) - b) for a, b in zip(per, T["per_repeat"])],
                 "matches_table1_4dp": [round(a, 4) for a in per] == T["per_repeat"] and round(m5, 4) == T["mean"],
                 "pod_versions": J["versions"], "podverify_pass": V.get("PASS"), "n_jobs": J["n_jobs"], "pod_seconds": J["seconds"]},
       "answer": {"file": ZS, "auc": zauc, "auc_4dp": round(zauc, 4), "column": "p_yes"},
       "gap": {"point": m5 - zauc, "ci_2p5_97p5": list(B["ci_diff"]), "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
               "probe_mean5_ci": list(B["ci_mean5"]), "answer_ci": list(B["ci_zs"]), "usable_draws": B["usable"]},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC, on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score", "degenerate_draws_dropped": NB - B["usable"]},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv),
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": JS, "pod_verify_json": PV, "zeroshot": ZS,
                 "states_source": STATES, "table1_json": T1},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(JS), "zeroshot": sha(ZS), "states_source": sha(STATES), "table1_json": sha(T1),
                  "perclip_csv": sha(PC), "draws_npz": sha(DR)}}
json.dump(rec, open(f"{R}/Q2A_kcl_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen2-Audio MDVR-KCL probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (Table 1 {T['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}", flush=True)
