"""PART26 cell Kimi-Audio / Pitt: paired probe-minus-answer gap on the paper's estimator (mean of the five per-repeat
nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod run of
part25/A/podfiles/p25a_nested5.py (unchanged) on the saved release fold files.

Encoder states: NEW for PART26. Re-extracted on this cell's GPU pod p26-kimi-pitt-gpu with kimi_enc_extract.py (the
Kimi-Audio_PC-GITA cell script, copied unchanged): forward hooks on the 32 Whisper-large-v3 encoder layers inside
kimia_infer (prompt_manager.whisper_model.speech_encoder.layers), first 30 s of each clip, mean over the encoder
frames Kimi keeps (all 1500 for these clips, every clip is at least 30 s).

Bootstrap (same code path as part25/A/scripts/A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True),
and takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged,
the zero-shot answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot answer: release/overnight2/part3/kimi_pitt_zeroshot_scores.csv (p_yes; the canonical file per the 23 Sep
decision and the part25/C C_build_cells zs_file, read only).
Writes, under part26/Kimi-Audio_Pitt/: Kimi-Audio_Pitt_perclip.csv, Kimi-Audio_Pitt.sidecar.json, Kimi-Audio_Pitt_draws.npz.
usage: /usr/local/bin/python3 p26_gap.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

C = "scores/part26/Kimi-Audio_Pitt"
POD = f"{C}/pull/p26-kimi-pitt-cpu"
GPU = f"{C}/pull/p26-kimi-pitt-gpu"
ZS = "<local data dir>/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv"
ZS_SHA_EXPECTED = "15a3c5c1175df984d7117cb7f9cbc401a1b6fd12a6dbecf1d68904854a069365"   # part25/C C_build_cells.csv zs_file_sha256
LMREF = "<local data dir>/release_from_mac/scores/part25/C/sidecars/C_kimi_pitt_llm.sidecar.json"
STATES = f"{GPU}/out/kimi_pitt_encstates.npz"
STAGE = f"{C}/stage/in/kimi_pitt_encstates.npz"
NB = 2000; TAG = "Kimi-Audio_Pitt"

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

oofp = f"{POD}/out/p26_kimi_pitt_enc_nested5_oof.csv"; jp = f"{POD}/out/p26_kimi_pitt_enc_nested5.json"
vj = f"{POD}/out/p26_kimi_pitt_podverify.json"
J = json.load(open(jp)); V = json.load(open(vj))
rows = list(csv.DictReader(open(oofp))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
assert sha(ZS) == ZS_SHA_EXPECTED, "zero-shot file changed since part25/C"
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
dr = f"{C}/{TAG}_draws.npz"
np.savez_compressed(dr, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_probe_mean5=np.array(m5),
                    point_zeroshot=np.array(zauc), ci_diff=np.array(B["ci_diff"]))
gz = np.load(STATES, allow_pickle=True); gmap = dict(zip(gz["name"].astype(str), gz["p_yes"].astype(float)))
pc_this = np.array([gmap[c] for c in names])
pcf = f"{C}/{TAG}_perclip.csv"
with open(pcf, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_podtie", "fold_single_podtie", "p_yes_zeroshot", "p_yes_zeroshot_rerun_on_gpu_pod"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i])), repr(float(pc_this[i]))])
SC = json.load(open(f"{C}/stage/stage_check.json")); EJ = json.load(open(f"{GPU}/out/kimi_pitt_encstates.json"))
LM = json.load(open(LMREF)) if os.path.exists(LMREF) else {}
rec = {"cell": "Kimi-Audio / Pitt", "model": "Kimi-Audio-7B-Instruct", "dataset": "Pitt", "condition": "AD",
       "part": "PART26", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "encoder nested probe, layer chosen inside each outer training fold by mean inner GroupKFold(4) AUC (pod tie rule), StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced') refit at that layer; outer folds = saved release fold files pitt_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv; value = mean of the five per-repeat out-of-fold AUCs",
                 "script": "part25/A/podfiles/p25a_nested5.py (unchanged, sha256 710e3de8d94eaa9cbc0cb3a5c30868415574ad669aa4fa6e0c728edfdc0fb68e)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_podtie_auc": float(roc_auc_score(y, Ps)), "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule",
                                                                  "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs")} for t in J["tags"]},
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "states_on_pod": J["states"]},
       "encoder_states": {"new_in_part26": True, "gpu_pod": "p26-kimi-pitt-gpu (ykow87gaqraiqs, NVIDIA H100 80GB HBM3, $3.49/h, created 01:06:02, deleted 01:18:37 UTC)",
                          "script": "gpu_stage/kimi_enc_extract.py = Kimi-Audio_PC-GITA/podfiles/kimi_enc_extract.py, sha256 fa201c10479ecb58a8641981d6903e76cf28001d5a58b709cb96d9d83debb454",
                          "setup": "gpu_stage/setup_kimi.sh (sha256 324748a00e5ee35c5fef42f23e2096e53aabaa11e7416b6c8c54f4a1a1d39adf)",
                          "encoder": EJ.get("encoder"), "pooling": EJ.get("pooling"), "enc_shape": EJ.get("enc_shape"),
                          "window": "first 30 s, x[:16000*30]; every Pitt clip is 30.01 to 55.56 s, so 480000 samples and 1500 kept frames for all 468",
                          "versions": EJ.get("versions"), "hookcheck_max": EJ.get("hookcheck_max_over_clips"),
                          "clip_list": "gpu_stage/mf_pitt.csv, zero-shot file row order", "stage_check": SC,
                          "determinism": {"rerun10_enc_max_abs": SC["rerun10_enc_max_abs"], "rerun10_p_yes_max_abs": SC["rerun10_p_yes_max_abs"]},
                          "same_audio_window_evidence": {"wav_sha256_equal_mac": SC["wav_sha256_equal_mac"],
                                                         "p_yes_rerun_vs_canonical": SC["p_yes_vs_canonical_30s"],
                                                         "p_yes_rerun_vs_18sep_fullclip_contrast": SC["p_yes_vs_18sep_fullclip_contrast"],
                                                         "note": "the answer side is the canonical 19 Sep file; the pod rerun of the answer is recorded only (different torch/flash-attn build)"}},
       "answer": {"file": ZS, "sha256": sha(ZS), "zeroshot_auc": zauc, "zeroshot_auc_4dp": round(zauc, 4), "score": "p_yes",
                  "rerun_on_gpu_pod_auc": float(roc_auc_score(y, pc_this))},
       "gap": {"point": m5 - zauc, "point_4dp": round(m5 - zauc, 4), "ci95": list(B["ci_diff"]), "ci95_4dp": [round(v, 4) for v in B["ci_diff"]],
               "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
               "probe_mean5_ci95": list(B["ci_mean5"]), "zeroshot_ci95": list(B["ci_zs"]), "usable_draws": B["usable"],
               "p_boot_diff_le0": float(np.mean(B["diff"][~np.isnan(B["diff"])] <= 0))},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat probe AUCs and the zero-shot answer AUC on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score", "code_path": "part25/A/scripts/A_gaps.py boot(), copied unchanged"},
       "comparisons": {"part25C_kimi_pitt_LM_probe_reference": {"file": LMREF, "note": "the earlier Kimi LM-stage probe (not the encoder), recorded only",
                                                                 "gap": LM.get("gap") or LM.get("diff_point"), "probe": LM.get("probe") or LM.get("probe_mean5")}},
       "podverify": {"file": vj, "PASS": V.get("PASS"), "max_abs_pred_diff": V.get("max_abs_pred_diff"),
                     "max_abs_inner_score_diff": V.get("max_abs_inner_score_diff"), "layer_mismatches": V.get("layer_mismatches"),
                     "abs_mean5_diff": V.get("abs_mean5_diff")},
       "inputs": {"states_source": STATES, "states_source_sha256": sha(STATES), "states_stage_enc_only": STAGE, "stage_sha256": sha(STAGE)},
       "files": {"perclip_csv": pcf, "draws_npz": dr, "pod_oof_csv": oofp, "pod_json": jp, "pod_verify_json": vj},
       "sha256": {"pod_oof_csv": sha(oofp), "pod_json": sha(jp), "perclip_csv": sha(pcf), "draws_npz": sha(dr)},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/{TAG}.sidecar.json", "w"), indent=1)
print(f"{TAG}: probe mean5 {m5:.4f} per {[round(a, 4) for a in per]}  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}] usable {B['usable']}  podverify {V.get('PASS')}  "
      f"layers {[J['tags'][f'seed{s}']['layers'] for s in range(5)]}")
