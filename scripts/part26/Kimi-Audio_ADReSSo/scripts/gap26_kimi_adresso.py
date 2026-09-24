"""PART 26: Kimi-Audio on ADReSSo. Paired probe minus zero-shot answer gap on the paper's estimator (mean of the
five per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART 26 CPU pod run of
p25a_nested5.py (byte copy of the PART 25 A script) on the PART 26 Kimi Whisper-encoder states.
Bootstrap is the same code path as part26/Qwen2-Audio_ADReSSo/scripts/gap26.py and part25/A/scripts/A_gaps.py:
2000 draws, a fresh numpy default_rng(0); each draw resamples the unique speakers (np.unique order) with replacement,
idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every drawn speaker. In each draw the five
per-repeat AUCs (sklearn roc_auc_score) are averaged and the zero-shot answer AUC on the same rows is subtracted.
2.5 and 97.5 percentiles over the usable draws.
Zero-shot per-clip file: release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv (read only; C_build_cells.csv row, AUC 0.7531).
Sensitivity only (not the reported gap): the same draws with the PART 26 rerun answer p_yes from the extraction pass.
Writes under part26/Kimi-Audio_ADReSSo/: per_clip/Kimi-Audio_ADReSSo_perclip.csv, the sidecar json next to it,
draws/Kimi-Audio_ADReSSo_draws.npz.   usage: /usr/local/bin/python3 gap26_kimi_adresso.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

C = "scores/part26/Kimi-Audio_ADReSSo"
POD = f"{C}/pull/p26-kimi-adresso-cpu2"
OOF = f"{POD}/out/p26_kimi_adresso_enc_nested5_oof.csv"
PJ = f"{POD}/out/p26_kimi_adresso_enc_nested5.json"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv"
GPU = f"{C}/pull/p26-kimi-adresso-gpu"
STATES = f"{GPU}/out/kimi_adresso_enc_states.npz"
STATES_JSON = f"{GPU}/out/kimi_adresso_enc_states.json"
RERUN = f"{GPU}/out/kimi_adresso_rerun_zeroshot_scores.csv"
EXTRACT = f"{GPU}/kimi_encstates26.py"
SLIM = f"{C}/stage/cpu2/in/kimi_adresso_enc_states_slim.npz"
NESTED = f"{C}/stage/cpu2/p25a_nested5.py"
NESTED_P25A = "<local data dir>/release_from_mac/scores/part25/A/podfiles/p25a_nested5.py"
STATES_CHECK = f"{C}/verify/p26_states_check.json"
LM_T1 = "<local data dir>/release/overnight2/kimi_new/kimi_adresso_nested_repeats.json"
FOLDS = "<local data dir>/release/folds"
NB = 2000
ZS_AUC_EXPECTED = 0.7531004989308624

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot(y, spk, P, zs, zs_alt):
    u = np.unique(spk); rows = [np.where(spk == s)[0] for s in u]; nsp = len(u)
    rng = np.random.default_rng(0)
    rep = np.full((NB, P.shape[0]), np.nan); zb = np.full(NB, np.nan); za = np.full(NB, np.nan); idxs = np.zeros((NB, nsp), np.int32)
    for b in range(NB):
        idx = rng.choice(nsp, size=nsp, replace=True); idxs[b] = idx
        ii = np.concatenate([rows[i] for i in idx]); yy = y[ii]
        if yy.min() == yy.max(): continue
        rep[b] = [roc_auc_score(yy, P[r][ii]) for r in range(P.shape[0])]
        zb[b] = roc_auc_score(yy, zs[ii]); za[b] = roc_auc_score(yy, zs_alt[ii])
    ok = ~np.isnan(zb)
    m5 = rep.mean(axis=1); d = m5 - zb; da = m5 - za
    pc = lambda v: (float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5)))
    return dict(rep=rep, zs=zb, mean5=m5, diff=d, idx=idxs, usable=int(ok.sum()), speakers=u, diff_alt=da,
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb), ci_diff_alt=pc(da), p_le0=float(np.mean(d[ok] <= 0)))

assert sha(NESTED) == sha(NESTED_P25A), "nested script is not a byte copy of the PART 25 A script"
SC = json.load(open(STATES_CHECK)); assert SC["STATES_OK"], "states check failed"
J = json.load(open(PJ))
assert J["versions"]["sklearn"] == "1.9.1" and J["versions"]["numpy"] == "2.1.2" and J["versions"]["scipy"] == "1.18.1", J["versions"]
assert J["states"]["sha256"] == sha(SLIM), "pod read a different states file"
rows = list(csv.DictReader(open(OOF)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
assert n == 237 and np.isfinite(P).all() and np.isfinite(Ps).all()

zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and zmap[c]["speaker"] == s for c, l, s in zip(names, y, spk)), "labels or speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
rmap = {r["clip"]: r for r in csv.DictReader(open(RERUN))}
assert set(rmap) == set(names) and all(int(rmap[c]["label"]) == int(l) for c, l in zip(names, y))
zr = np.array([float(rmap[c]["p_yes"]) for c in names])

fold_check = {}
for s in range(5):
    ff = f"{FOLDS}/adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    fr = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(ff))}
    fold_check[f"seed{s}"] = {"file": ff, "sha256": sha(ff), "equal": all(fr[r["clip"]] == int(r[f"fold_seed{s}"]) for r in rows)}
assert all(v["equal"] for v in fold_check.values()), "pod fold ids differ from the saved fold files"

per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
assert abs(zauc - ZS_AUC_EXPECTED) < 1e-12, zauc
assert all(abs(a - b) < 1e-12 for a, b in zip(per, J["per_repeat_auc"])), "per-repeat AUC differs from the pod json"
zr_auc = float(roc_auc_score(y, zr))
B = boot(y, spk, P, zs, zr)

os.makedirs(f"{C}/per_clip", exist_ok=True); os.makedirs(f"{C}/draws", exist_ok=True)
DR = f"{C}/draws/Kimi-Audio_ADReSSo_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc),
                    diff_sensitivity_rerun_answer=B["diff_alt"])
PC = f"{C}/per_clip/Kimi-Audio_ADReSSo_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot", "p_yes_rerun_part26"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i])), repr(float(zr[i]))])

SJ = json.load(open(STATES_JSON)); LM = json.load(open(LM_T1))
rec = {"task": "PART 26 probe grid: Kimi-Audio encoder nested probe on ADReSSo, paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "model": "Kimi-Audio (moonshotai/Kimi-Audio-7B-Instruct)", "dataset": "ADReSSo", "condition": "AD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "states": {"what": "NEW in PART 26: Kimi-Audio Whisper-large-v3 encoder (kimia_infer prompt_manager.whisper_model.speech_encoder), "
                          "forward hooks on all 32 layers, each layer output mean-pooled over the frames Kimi passes to the LM "
                          "(first token_len*4 of the 1500 padded frames). 30 s window x[:16000*30], the same clips, prompt and window as the zero-shot file. "
                          "No earlier Kimi run extracted encoder states; the earlier Kimi probes hooked the language model only.",
                  "extraction_script": EXTRACT, "extraction_script_sha256": sha(EXTRACT),
                  "extraction_note": "built from release/podD2_final/kimi_probe.py (the script behind the Kimi zero-shot files) with encoder hooks added",
                  "file": STATES, "sha256": sha(STATES), "sidecar": SJ, "states_check": STATES_CHECK, "states_check_sha256": sha(STATES_CHECK),
                  "rerun_answer_auc_same_pass": zr_auc, "rerun_vs_zeroshot_p_yes": SC["rerun_vs_zeroshot_p_yes"],
                  "gpu_pod": "p26-kimi-adresso-gpu yz8nyovce4nx08, H100 80GB HBM3, extraction 01:24:38 to 01:25:29 UTC 24 Sep, pulled and deleted 01:28 UTC"},
       "probe": {"estimator": "nested encoder probe: outer folds = the five saved repeat fold files; inner GroupKFold(4) (pod tie rule) picks the "
                              "layer by mean inner AUC inside each outer training set; StandardScaler + LogisticRegression(max_iter=2000, "
                              "class_weight='balanced') refit at that layer; value = mean of the five per-repeat out-of-fold AUCs",
                 "script": "p25a_nested5.py (sha256 " + sha(NESTED) + ", byte identical to part25/A/podfiles/p25a_nested5.py)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc": float(roc_auc_score(y, Ps)), "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files_equal_pod_output": fold_check,
                 "fold_files_equal_pod_tie_rule": {t: J["tags"][t]["fold_file_equals_pod_tie_rule"] for t in J["tags"]},
                 "fold_files_equal_this_pod_sklearn": {t: J["tags"][t]["fold_file_equals_this_pod_sklearn"] for t in J["tags"]},
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "pod_command": J["command"], "pod_n_jobs": J["n_jobs"],
                 "blas_threads": "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 (driver26b_cpu.sh and the script's own setdefault)",
                 "states_used_on_pod": J["states"],
                 "context_not_compared": {"table1_kimi_adresso_llm_probe": LM["llm"], "file": LM_T1}},
       "zeroshot": {"file": ZS, "sha256": sha(ZS), "auc": zauc, "auc_4dp": round(zauc, 4), "score_column": "p_yes"},
       "gap": {"point": m5 - zauc, "ci_2p5_97p5": list(B["ci_diff"]), "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0), "p_boot_le0": B["p_le0"],
               "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"], "total_draws": NB},
       "sensitivity_rerun_answer": {"note": "not the reported gap; answer p_yes from the PART 26 extraction pass instead of the zero-shot file",
                                    "answer_auc": zr_auc, "point": m5 - zr_auc, "ci_2p5_97p5": list(B["ci_diff_alt"])},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC, same rows in the same draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "versions_mac": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "states_source": STATES, "states_uploaded": SLIM,
                 "rerun_answer_csv": RERUN, "gap_script": os.path.abspath(__file__)},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "states_source": sha(STATES), "states_uploaded": sha(SLIM),
                  "rerun_answer_csv": sha(RERUN), "perclip_csv": sha(PC), "draws_npz": sha(DR)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/per_clip/Kimi-Audio_ADReSSo_perclip.sidecar.json", "w"), indent=1)
print(f"Kimi-Audio ADReSSo encoder probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} layers {[J['tags'][f'seed{s}']['layers'] for s in range(5)]}  "
      f"answer {zauc:.4f}  gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}  "
      f"(sensitivity rerun answer {zr_auc:.4f} gap {m5 - zr_auc:+.4f} [{B['ci_diff_alt'][0]:+.4f}, {B['ci_diff_alt'][1]:+.4f}])")
