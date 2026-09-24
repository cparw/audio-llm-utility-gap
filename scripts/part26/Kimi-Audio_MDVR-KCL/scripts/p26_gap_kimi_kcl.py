"""PART26: Kimi-Audio, MDVR-KCL. Paired probe minus zero-shot answer gap on the paper's estimator (mean of five
per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors the PART26 CPU pod saved (p25a_nested5.py,
the PART25 A script, unchanged) on the five saved pod fold files.
Encoder states: re-extracted in PART26 on a GPU pod (kimi_enc_probe.py = podD2_final/kimi_probe.py plus forward hooks on the
32 Whisper-large-v3 encoder layers inside kimia_infer, mean over the 1500 frames Kimi keeps), checked by
p26_states_check_kimi_kcl.py before the probe ran.
Bootstrap (same as part25/A/scripts/A_gaps.py boot() and the PART26 Qwen2-Audio KCL cell): 2000 draws, a fresh numpy
default_rng(0); each draw resamples the unique speakers (np.unique order) with replacement,
idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every drawn speaker. In each draw the five
per-repeat AUCs (sklearn roc_auc_score) are averaged, the zero-shot answer AUC is computed on the same rows, and the
difference is taken. 2.5 and 97.5 percentiles.
Zero-shot: release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv (read only), the file behind the master 0.7113.
Writes per_clip/p26_kimi_kcl_perclip.csv, per_clip/p26_kimi_kcl_perclip.sidecar.json, draws/p26_kimi_kcl_draws.npz."""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Kimi-Audio_MDVR-KCL"
POD = f"{R}/pull/p26-kimi-kcl-cpu"
GPOD = f"{R}/pull/p26-kimi-kcl-gpu"
OOF = f"{POD}/out/p26_kimi_kcl_enc_nested5_oof.csv"; JS = f"{POD}/out/p26_kimi_kcl_enc_nested5.json"
PV = f"{POD}/out/p26_kimi_kcl_podverify.json"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv"
STATES = f"{GPOD}/out/p26_kimi_kcl_states.npz"
SCHK = f"{R}/verify/p26_states_check_kimi_kcl.json"
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

SC = json.load(open(SCHK)); assert SC["states_ok"], "states check did not pass"
J = json.load(open(JS)); V = json.load(open(PV)) if os.path.exists(PV) else {}
assert J["states"]["sha256"] == SC["states_sha256"], "probe ran on a different states file"
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
os.makedirs(f"{R}/draws", exist_ok=True); os.makedirs(f"{R}/per_clip", exist_ok=True)
DR = f"{R}/draws/p26_kimi_kcl_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc))
PC = f"{R}/per_clip/p26_kimi_kcl_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
chk = SC["checks"]
rec = {"task": "PART26 probe grid: Kimi-Audio encoder nested probe (layer chosen inside the training folds), five repeats on the saved fold files, paired probe minus zero-shot answer gap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "model": "Kimi-Audio-7B-Instruct", "dataset": "MDVR-KCL", "condition": "PD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()), "n_layers": J["n_layers"],
       "method_note": "NEW in PART26: no Kimi encoder states existed before. Earlier Kimi runs hooked only the language model and the paper Method says the Kimi encoder side is skipped. These states come from new forward hooks on the Whisper-large-v3 encoder inside kimia_infer (the path that builds history.continuous_feature). Using this cell in the paper needs the Method text changed.",
       "states": {"script": "stage/gpu/kimi_enc_probe.py (podD2_final/kimi_probe.py plus Whisper encoder hooks)",
                  "setup": "stage/gpu/setup_kimi.sh (edaic_conflict_new/kimi/setup_kimi.sh, snapshot pinned to 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b)",
                  "window": "first 30 s (x[:16000*30]); all 37 KCL clips are 30.0 s",
                  "encoder": "32 Whisper-large-v3 layers, 1280 dims, bf16; each layer output cut to the 1500 frames Kimi keeps and mean-pooled (h.mean(dim=0).float()), as the Qwen2.5-Omni encoder states",
                  "file": STATES, "sha256": SC["states_sha256"], "check_file": SCHK,
                  "rerun_p_yes_vs_saved_zeroshot": chk["rerun_p_yes_reproduces_saved_zeroshot"],
                  "rerun_llm_vs_saved_llm_states": chk["rerun_llm_states_reproduce_saved"],
                  "final_layernorm_vs_continuous_feature": chk["final_layernorm_equals_kimi_continuous_feature"]},
       "probe": {"estimator": "outer folds = saved pod fold files seed0..4; inner GroupKFold(4) with the pod tie rule on the outer training speakers scores every encoder layer (32) by mean inner AUC; argmax; refit StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'); score held-out clips. Probe = mean over the five repeats of the out-of-fold AUC.",
                 "script": "part25/A/podfiles/p25a_nested5.py (unchanged, md5 615ced0380cca2636ed7cc30c71810e6)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc": J["single_split_auc"], "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files": {t: J["tags"][t]["fold_file"] for t in J["tags"]},
                 "fold_file_sha256": {t: J["tags"][t]["fold_file_sha256"] for t in J["tags"]},
                 "fold_files_status": "UNVERIFIED in release/folds (they reproduce the saved per-repeat AUCs within 0.008, no per-clip OOF ever existed to check them clip by clip); each equals the pod tie rule rebuild (asserted by the probe script)",
                 "table1_saved": None, "table1_note": "no earlier Kimi encoder probe exists, so there is no Table 1 value to compare with",
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
                 "states": STATES},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(JS), "zeroshot": sha(ZS), "states": sha(STATES),
                  "perclip_csv": sha(PC), "draws_npz": sha(DR)}}
json.dump(rec, open(f"{R}/per_clip/p26_kimi_kcl_perclip.sidecar.json", "w"), indent=1)
print(f"Kimi-Audio MDVR-KCL encoder probe mean5 {m5:.4f} per {[round(a, 4) for a in per]}  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}", flush=True)
