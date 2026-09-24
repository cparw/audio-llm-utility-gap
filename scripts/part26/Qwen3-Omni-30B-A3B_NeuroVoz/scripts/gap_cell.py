"""PART26 cell Qwen3-Omni-30B-A3B x NeuroVoz: paired probe-minus-answer gap on the paper's estimator (mean of the five
per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod
(p25a_nested5.py, unchanged PART25 A script). Same bootstrap code as part25/A/scripts/A_gaps.py (boot()).

Bootstrap: 2000 draws, a fresh numpy default_rng(0); each draw resamples the unique speakers (np.unique order of the
states `spk`) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every
drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged, the zero-shot answer AUC is
computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot answer: overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv (read only), the master_lookup file (AUC 0.6393).
Writes under the cell folder: per_clip/<cell>_perclip.csv, per_clip/<cell>_perclip.sidecar.json, draws/<cell>_draws.npz.
usage: /usr/local/bin/python3 gap_cell.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

CELL = "Qwen3-Omni-30B-A3B_NeuroVoz"
C = f"<local data dir>/release_from_mac/scores/part26/{CELL}"
REL = "<local data dir>/release"
POD = f"{C}/pull/p26-q3o-neurovoz"
OOF = f"{POD}/out/p26_q3o_neurovoz_enc_nested5_oof.csv"
PJ = f"{POD}/out/p26_q3o_neurovoz_enc_nested5.json"
PV = f"{POD}/out/p26_q3o_neurovoz_podverify.json"
ENV = f"{POD}/out/p26_env.json"
ST = f"{C}/stage/q3o_neurovoz_encstates.npz"
ZS = f"{REL}/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv"
REF_REP = f"{REL}/overnight2/q3o_new/q3o_neurovoz_nested_repeats.json"
REF_SINGLE = f"{REL}/overnight2/q3o_new/q3o_neurovoz_enc_nested_oof.csv"
REF_SINGLE_J = f"{REL}/overnight2/q3o_new/q3o_neurovoz_nested.json"
NB = 2000
os.makedirs(f"{C}/per_clip", exist_ok=True); os.makedirs(f"{C}/draws", exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot(y, spk, P, zs):
    """P: (5, n) per-repeat OOF scores; zs: (n,) zero-shot p_yes.  (copied from part25/A/scripts/A_gaps.py)"""
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

J = json.load(open(PJ)); V = json.load(open(PV)); E = json.load(open(ENV))
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
z = np.load(ST, allow_pickle=True)
assert list(z["name"].astype(str)) == names, "OOF clip order differs from the states"
assert np.array_equal(z["label"].astype(int), y) and np.array_equal(z["spk"].astype(str), spk), "labels/speakers differ from states"
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) for c, l in zip(names, y)), "zero-shot labels differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{C}/draws/{CELL}_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc))
REFR = json.load(open(REF_REP))["enc"]
ref_single = {r["clip"]: float(r["p_probe"]) for r in csv.DictReader(open(REF_SINGLE))}
ps_ref = np.array([ref_single[c] for c in names])
PC = f"{C}/per_clip/{CELL}_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_rerun", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
rho = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
rec = {"cell": CELL, "model": "Qwen3-Omni-30B-A3B (Qwen/Qwen3-Omni-30B-A3B-Instruct)", "dataset": "NeuroVoz", "condition": "PD",
       "task": "PART26 encoder nested probe (layer chosen inside the training folds) vs zero-shot answer, paired speaker bootstrap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()), "n_encoder_layers": J["n_layers"],
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_pod_json": J["per_repeat_auc"], "probe_mean5_pod_json": J["mean5"],
       "zeroshot_auc": zauc, "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]),
       "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
       "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
       "single_split_auc_rerun": float(roc_auc_score(y, Ps)), "single_split_layers_rerun": J["tags"]["single"]["layers"],
       "reference_earlier_pod_run": {"per_repeat": REFR["per_repeat"], "mean": REFR["mean"], "file": REF_REP,
                                     "abs_diff_per_repeat_4dp": [round(abs(round(a, 4) - b), 4) for a, b in zip(per, REFR["per_repeat"])],
                                     "mean5_minus_reference": m5 - REFR["mean"],
                                     "single_split_auc_saved": float(roc_auc_score(y, ps_ref)),
                                     "single_split_layers_saved": json.load(open(REF_SINGLE_J))["enc"]["chosen_layers"],
                                     "single_split_rank_corr_rerun_vs_saved": rho(Ps, ps_ref),
                                     "single_split_max_abs_pred_diff_rerun_vs_saved": float(np.max(np.abs(Ps - ps_ref)))},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order of the states spk), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC, same rows in each draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score", "code": "boot() copied from part25/A/scripts/A_gaps.py"},
       "probe": {"script": "p25a_nested5.py (PART25 A, unchanged, sha256 in inputs)", "outer_folds": "saved release fold files neurovoz_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv (plus neurovoz_groupkfold5_pod.csv for the single split)",
                 "inner": "GroupKFold(4) with the pod tie rule inside each outer training set, layer = argmax mean inner AUC over 32 encoder layers",
                 "model": "StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced')",
                 "fold_files": {t: J["tags"][t]["fold_file"] for t in J["tags"]},
                 "fold_file_equals_pod_tie_rule": {t: J["tags"][t]["fold_file_equals_pod_tie_rule"] for t in J["tags"]}},
       "pod": {"id": "fecp2sxone9uvn", "name": "p26-q3o-neurovoz", "type": "CPU cpu3c 32 vCPU", "cost_per_hr": 0.96,
               "versions": J["versions"], "blas_threads": [p["num_threads"] for p in E["threadpools"]], "threadpools": E["threadpools"],
               "n_jobs": J["n_jobs"], "seconds": J["seconds"], "podverify_pass": V.get("PASS"),
               "podverify": {k: V.get(k) for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff", "mean5_rank_from_oof")}},
       "states": {"file_used": ST, "sha256": sha(ST), "source": f"{REL}/overnight2/q3o/q3o_neurovoz_states.npz",
                  "note": "enc, label, spk, p_yes, name copied byte-identical from the source states; p_yes equals the zero-shot file exactly (max diff 0.0)"},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV, "pod_env_json": ENV,
                 "zeroshot": ZS},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "zeroshot": sha(ZS), "perclip_csv": sha(PC), "draws_npz": sha(DR)},
       "versions_mac": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/per_clip/{CELL}_perclip.sidecar.json", "w"), indent=1)
print(f"{CELL}: probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (earlier pod run {REFR['mean']:.4f} {REFR['per_repeat']})  "
      f"zs {zauc:.4f}  gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}]  usable {B['usable']}  podverify {V.get('PASS')}", flush=True)
