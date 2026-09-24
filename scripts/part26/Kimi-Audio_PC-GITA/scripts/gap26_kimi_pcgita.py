"""PART26 Kimi-Audio PC-GITA: paired probe minus zero-shot answer gap on the mean-of-five nested encoder probe.
Probe: per-repeat out-of-fold scores from the PART26 CPU pod run of p25a_nested5.py (the PART25 A Qwen2.5-Omni nested
probe script, unchanged) on the saved fold files release/folds/pcgita_groupkfold5_pod_seed{0..4}.csv, over the NEW
PART26 Kimi encoder states (Whisper-large-v3 encoder inside Kimi-Audio, 32 layers x 1280, mean over the frames Kimi keeps,
first 30 s; podfiles/kimi_enc_extract.py).
Answer: the canonical Kimi zero-shot file overnight2/part3/kimi_pcgita_zeroshot_scores.csv (p_yes per clip, read only;
AUC 0.6147, the master value). Its speaker column equals the fold files' speaker_id (100 speakers).
Bootstrap: same code path as part26/Qwen2-Audio_PC-GITA/scripts/gap26_q2a_pcgita.py boot() (= part25/A/scripts/A_gaps.py):
2000 draws, fresh numpy default_rng(0); unique speakers in np.unique order; idx = rng.choice(n_spk, size=n_spk,
replace=True); every clip of each drawn speaker; in each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged
and the zero-shot AUC on the same rows is subtracted. 2.5 and 97.5 percentiles.
Secondary (sidecar only, not the reported gap): the same bootstrap with the p_yes of the PART26 extraction pod run.
Writes Kimi-Audio_PC-GITA_perclip.csv, Kimi-Audio_PC-GITA_perclip.sidecar.json, draws/Kimi-Audio_PC-GITA_draws.npz."""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np, sklearn
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_PC-GITA"
POD = f"{C}/pull/p26-kimi-pcgita-cpu"
OOF = f"{POD}/out/p26_kimi_pcgita_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_kimi_pcgita_enc_nested5.json"
PV = f"{POD}/out/p26_kimi_pcgita_podverify.json"
ZS = "<local data dir>/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv"
GPUST = f"{C}/pull/p26-kimi-pcgita-gpu/out/kimi_pcgita_encstates.npz"
GPUJ = f"{C}/pull/p26-kimi-pcgita-gpu/out/kimi_pcgita_encstates.json"
STG = f"{C}/verify/stage_cpu_check.json"
LMREF = "<local data dir>/release/overnight2/part3/kimi_pcgita_nested_repeats.json"
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
J = json.load(open(PJ)); V = json.load(open(PV)); S = json.load(open(STG)); G = json.load(open(GPUJ))
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) for c, l in zip(names, y)), "labels differ"
assert all(zmap[c]["speaker"] == s for c, s in zip(names, spk)), "speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
g = np.load(GPUST, allow_pickle=True); gmap = dict(zip(g["name"].astype(str), g["p_yes"].astype(float)))
zs_pod = np.array([gmap[c] for c in names])
assert np.isfinite(P).all() and np.isfinite(zs).all()
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
assert abs(m5 - J["mean5"]) < 1e-12, (m5, J["mean5"])
B = boot(y, spk, P, zs)
B2 = boot(y, spk, P, zs_pod); zauc_pod = float(roc_auc_score(y, zs_pod))
os.makedirs(f"{C}/draws", exist_ok=True)
DR = f"{C}/draws/Kimi-Audio_PC-GITA_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc),
                    secondary_diff_podrun_pyes=B2["diff"], secondary_zeroshot_podrun_pyes=B2["zs"])
PC = f"{C}/Kimi-Audio_PC-GITA_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_split", "fold_single_split", "p_yes_zeroshot", "p_yes_zeroshot_part26_pod_rerun"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i])), repr(float(zs_pod[i]))])
LM = json.load(open(LMREF)) if os.path.exists(LMREF) else None
rec = {"task": "PART26 probe grid: Kimi-Audio, PC-GITA, nested encoder probe (layer chosen inside the training folds), mean of five repeats, paired gap vs zero-shot answer",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "model": "Kimi-Audio-7B-Instruct", "dataset": "PC-GITA", "condition": "PD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "METHOD_NOTE": ("NEW extraction path. No earlier run extracted Kimi encoder states (the paper Method says the Kimi encoder side is skipped). "
                       "These states are the 32 layers of the Whisper-large-v3 encoder inside Kimi-Audio (the continuous-feature path into the LM), "
                       "mean-pooled over the frames Kimi keeps, first 30 s. Kimi's other audio path, the discrete GLM-4-Voice tokens, is not probed."),
       "states": J["states"], "n_layers": J["n_layers"], "window": "first 30 s (x[:16000*30]), as the Kimi zero-shot run",
       "extraction": {"script": f"{C}/podfiles/kimi_enc_extract.py", "script_sha256": sha(f"{C}/podfiles/kimi_enc_extract.py"),
                      "pod": "p26-kimi-pcgita-gpu roj4j50j5vbk1a (H100 80GB HBM3)", "sidecar": G, "stage_check_pass": S.get("PASS"),
                      "p_yes_rerun_vs_canonical": S.get("p_yes_this_run_vs_canonical"), "rerun10_enc_max_abs": S.get("rerun10_enc_max_abs_vs_full"),
                      "hookcheck_max": S.get("hookcheck_maxdiff_max"), "wav_sha256_equal_upload": S.get("wav_sha256_equal_upload_list")},
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_4dp": [round(a, 4) for a in per], "probe_mean5_4dp": round(m5, 4),
       "probe_json_mean5": J["mean5"], "single_split_auc": float(roc_auc_score(y, Ps)),
       "layers": {t: J["tags"][t]["layers"] for t in J["tags"]},
       "zeroshot_auc": zauc, "zeroshot_file": ZS,
       "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_interval_above_zero": bool(B["ci_diff"][0] > 0),
       "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "secondary_gap_vs_part26_pod_rerun_pyes": {"zeroshot_auc": zauc_pod, "gap_point": m5 - zauc_pod, "gap_ci": list(B2["ci_diff"]),
                                                   "note": "sensitivity only; the reported gap uses the canonical zero-shot file"},
       "reference_kimi_lm_nested_repeats": ({"file": LMREF, "content": LM} if LM is not None else None),
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh", "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot AUC, same draw", "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score",
                     "speaker_source": "OOF csv speaker = states spk = fold file speaker_id = zero-shot speaker (100 speakers)"},
       "pod": {"name": "p26-kimi-pcgita-cpu", "versions": J["versions"], "n_jobs": J["n_jobs"], "seconds": J["seconds"],
               "podverify_pass": V.get("PASS"), "podverify": {k: V.get(k) for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff")},
               "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule", "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs", "inner_sk_equal")} for t in J["tags"]}},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV, "zeroshot": ZS, "gpu_states": GPUST},
       "sha256": {"perclip_csv": sha(PC), "draws_npz": sha(DR), "pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "zeroshot": sha(ZS), "gpu_states": sha(GPUST)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/Kimi-Audio_PC-GITA_perclip.sidecar.json", "w"), indent=1, default=str)
print(f"Kimi-Audio PC-GITA probe mean5 {m5:.4f} per {[round(a, 4) for a in per]}  zs {zauc:.4f}  gap {m5 - zauc:+.4f} "
      f"[{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']} | secondary (pod rerun p_yes) zs {zauc_pod:.4f} "
      f"gap {m5 - zauc_pod:+.4f} [{B2['ci_diff'][0]:.4f}, {B2['ci_diff'][1]:.4f}]")
