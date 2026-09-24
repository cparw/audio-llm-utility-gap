"""PART26 Qwen2-Audio PC-GITA: paired probe minus zero-shot answer gap on the mean-of-five nested encoder probe.
Probe: per-repeat out-of-fold scores from the PART26 CPU pod run of p25a_nested5.py (the PART25 A Qwen2.5-Omni nested
probe script, unchanged) on the saved fold files release/folds/pcgita_groupkfold5_pod_seed{0..4}.csv.
Answer: probe2/pcgita_zeroshot_scores.csv (p_yes per clip, read only). Its speaker column is the clip stem
(1100 ids), so speakers come from the OOF csv (states spk), which equal the fold files' speaker_id (100 speakers).
Bootstrap: same code path as part25/A/scripts/A_gaps.py boot(): 2000 draws, fresh numpy default_rng(0); unique
speakers in np.unique order; idx = rng.choice(n_spk, size=n_spk, replace=True); every clip of each drawn speaker; in
each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged and the zero-shot AUC on the same rows is
subtracted. 2.5 and 97.5 percentiles.
Writes Qwen2-Audio_PC-GITA_perclip.csv, Qwen2-Audio_PC-GITA_perclip.sidecar.json, draws/Qwen2-Audio_PC-GITA_draws.npz."""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np, sklearn
from sklearn.metrics import roc_auc_score
C = "scores/part26/Qwen2-Audio_PC-GITA"
POD = f"{C}/pull/p26-q2a-pcgita"
OOF = f"{POD}/out/p26_q2a_pcgita_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_q2a_pcgita_enc_nested5.json"
PV = f"{POD}/out/p26_q2a_pcgita_podverify.json"
ZS = "<local data dir>/paper1_local_runs/probe2/pcgita_zeroshot_scores.csv"
MASTER = "<local data dir>/paper1_local_runs/omni_final/q2a_pcgita_nested_repeats.json"
C25 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-2/out/q2a_pcgita_enc_rep5.json"
SINGLE_SAVED = "<local data dir>/paper1_local_runs/probe2/pcgita_enc_nested_oof.csv"
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
    ok = ~np.isnan(zb); m5 = rep.mean(axis=1); d = m5 - zb
    pc = lambda v: (float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5)))
    return dict(rep=rep, zs=zb, mean5=m5, diff=d, idx=idxs, usable=int(ok.sum()), speakers=u,
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb))
J = json.load(open(PJ)); V = json.load(open(PV))
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) for c, l in zip(names, y)), "labels differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
assert np.isfinite(P).all() and np.isfinite(zs).all()
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{C}/draws/Qwen2-Audio_PC-GITA_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc))
smap = {r["clip"]: r for r in csv.DictReader(open(SINGLE_SAVED))}
pcol = [k for k in next(iter(smap.values())).keys() if k.startswith("p")][0] if smap else None
ps_saved = np.array([float(smap[c][pcol]) for c in names]) if smap and set(smap) == set(names) else None
PC = f"{C}/Qwen2-Audio_PC-GITA_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_split", "fold_single_split", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
M = json.load(open(MASTER))["enc"]; C5 = json.load(open(C25))
rec = {"task": "PART26 probe grid: Qwen2-Audio, PC-GITA, nested encoder probe (layer chosen inside the training folds), mean of five repeats, paired gap vs zero-shot answer",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "model": "Qwen2-Audio-7B-Instruct", "dataset": "PC-GITA", "condition": "PD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "states": J["states"], "n_layers": J["n_layers"], "window": "30 s (probe2/extract2.py)",
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_4dp": [round(a, 4) for a in per], "probe_mean5_4dp": round(m5, 4),
       "probe_json_mean5": J["mean5"], "single_split_auc": float(roc_auc_score(y, Ps)),
       "single_split_auc_saved_probe2": (float(roc_auc_score(y, ps_saved)) if ps_saved is not None else None),
       "layers": {t: J["tags"][t]["layers"] for t in J["tags"]},
       "zeroshot_auc": zauc, "zeroshot_file": ZS, "zeroshot_speaker_column_note": "clip stem, 1100 ids; not used for the bootstrap",
       "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_interval_above_zero": bool(B["ci_diff"][0] > 0),
       "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "reference_master_table1": {"file": MASTER, "per_repeat": M["per_repeat"], "mean": M["mean"],
                                   "note": "master estimator: pod sklearn GroupKFold outer and inner on renumbered speakers; this run uses the saved fold files and the pod tie rule inner"},
       "reference_part25C_refit": {"file": C25, "per_repeat": C5["per_repeat"], "mean5": C5["mean5"], "layers": C5["chosen_layers"]},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh", "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot AUC, same draw", "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score",
                     "speaker_source": "OOF csv speaker = states spk = fold file speaker_id (100 speakers)"},
       "pod": {"name": "p26-q2a-pcgita", "id": "j0g9l1t6s9r3vm", "versions": J["versions"], "n_jobs": J["n_jobs"], "seconds": J["seconds"],
               "podverify_pass": V.get("PASS"), "podverify": {k: V[k] for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff")},
               "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule", "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs", "inner_sk_equal")} for t in J["tags"]}},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV, "zeroshot": ZS},
       "sha256": {"perclip_csv": sha(PC), "draws_npz": sha(DR), "pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "zeroshot": sha(ZS)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/Qwen2-Audio_PC-GITA_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen2-Audio PC-GITA probe mean5 {m5:.4f} per {[round(a, 4) for a in per]}  zs {zauc:.4f}  gap {m5 - zauc:+.4f} "
      f"[{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']}  master {M['mean']}  C25 {C5['mean5']:.4f}")
