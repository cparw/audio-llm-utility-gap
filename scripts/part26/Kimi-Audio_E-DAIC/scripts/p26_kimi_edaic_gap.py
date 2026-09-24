"""PART 26 Kimi-Audio E-DAIC: encoder probe minus zero-shot answer, paired speaker bootstrap.
Inputs
  probe : the CPU pod out-of-fold file of p25a_nested5.py (per-repeat OOF scores p_seed0..p_seed4 for every clip; the
          five repeats use the saved release fold files edaic_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv)
  answer: the canonical Kimi-Audio E-DAIC zero-shot per-clip file (master_lookup data row 25, 30 s window)
Method (the PART 25 track C estimator, C_build.py):
  probe AUC = mean of the five per-repeat OOF AUCs (sklearn roc_auc_score); gap = probe minus answer AUC on the same clips;
  paired speaker bootstrap, 2000 draws, fresh numpy default_rng(0), idx = rng.choice(n_spk, size=n_spk, replace=True) over
  the sorted unique speaker ids of the probe file, all clips of each drawn speaker; in each draw the mean of the five
  per-repeat AUCs minus the zero-shot AUC; 2.5 / 97.5 percentiles; draws with one class only dropped and counted.
Writes (cell folder): p26_kimi_edaic_enc_perclip.csv, p26_kimi_edaic_enc_perclip.sidecar.json,
  draws/p26_kimi_edaic_enc_draws.npz
usage: /usr/local/bin/python3 p26_kimi_edaic_gap.py OOF_CSV NESTED_JSON STATES_NPZ PYES_CHECK_JSON"""
import os, sys, csv, json, hashlib, datetime, platform
import numpy as np, pandas as pd, sklearn, scipy
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_E-DAIC"
ML = "<local data dir>/release/master/master_lookup.csv"
ZS = "<local data dir>/release/overnight2/part3/kimi_edaic_zeroshot_scores.csv"
ZSJ = "<local data dir>/release/overnight2/part3/kimi_edaic_zeroshot.json"
OOF, NJ, NPZ, PYC = sys.argv[1:5]
NB = 2000

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

ml = list(csv.DictReader(open(ML)))
zr = ml[25 - 1]
assert zr["model"] == "Kimi-Audio" and zr["dataset"] == "edaic" and zr["stream"] == "zero shot answer" and zr["source"] == ZSJ, zr
P = pd.read_csv(OOF, dtype={"clip": str, "speaker": str}, float_precision="round_trip"); Z = pd.read_csv(ZS, dtype={"clip": str, "speaker": str}, float_precision="round_trip")
J = json.load(open(NJ))
cols = [f"p_seed{k}" for k in range(5)]
M = P.merge(Z[["clip", "speaker", "label", "p_yes"]].rename(columns={"speaker": "speaker_zs", "label": "label_zs"}), on="clip", how="inner")
assert len(M) == len(P) == len(Z) == 275, (len(M), len(P), len(Z))
assert (M["label"].astype(int) == M["label_zs"].astype(int)).all(), "labels differ"
assert (M["speaker"] == M["speaker_zs"]).all(), "speakers differ"
y = M["label"].astype(int).values; spk = M["speaker"].astype(str).values
R = np.stack([M[c].astype(float).values for c in cols], 1); zs = M["p_yes"].astype(float).values
assert np.isfinite(R).all() and np.isfinite(zs).all()
per = [float(roc_auc_score(y, R[:, k])) for k in range(5)]
probe = float(np.mean(per)); ans = float(roc_auc_score(y, zs)); gap = probe - ans
assert round(ans, 4) == round(float(zr["auc"]), 4), (ans, zr["auc"])
assert all(abs(a - b) < 1e-12 for a, b in zip(per, J["per_repeat_auc"])), (per, J["per_repeat_auc"])
single = float(roc_auc_score(y, M["p_single"].astype(float).values))
# try 2 addition (sensitivity only, the headline gap above uses the canonical zero-shot file): p_yes of the PART 26
# re-extraction forward pass, the same pass that produced the encoder states, joined by clip name
_zr = np.load(NPZ, allow_pickle=True); _rr = {str(n): float(v) for n, v in zip(_zr["name"], _zr["p_yes"])}
zs_re = np.array([_rr[c] for c in M["clip"].astype(str)]); assert np.isfinite(zs_re).all() and len(_rr) == 275
ans_re = float(roc_auc_score(y, zs_re)); gap_re = probe - ans_re
# bootstrap
u = np.unique(spk); by = {s: np.flatnonzero(spk == s) for s in u}
rng = np.random.default_rng(0); dd, pp, qq, qr = [], [], [], []
for _ in range(NB):
    ii = np.concatenate([by[u[i]] for i in rng.choice(len(u), size=len(u), replace=True)])
    yy = y[ii]
    if yy.min() == yy.max(): dd.append(np.nan); pp.append(np.nan); qq.append(np.nan); qr.append(np.nan); continue
    a = float(np.mean([roc_auc_score(yy, R[ii, k]) for k in range(5)])); b = float(roc_auc_score(yy, zs[ii]))
    pp.append(a); qq.append(b); dd.append(a - b); qr.append(float(roc_auc_score(yy, zs_re[ii])))
dd, pp, qq, qr = np.array(dd), np.array(pp), np.array(qq), np.array(qr); ok = ~np.isnan(dd)
dre = pp - qr; rlo, rhi = [float(v) for v in np.percentile(dre[ok], [2.5, 97.5])]
lo, hi = [float(v) for v in np.percentile(dd[ok], [2.5, 97.5])]
plo, phi = [float(v) for v in np.percentile(pp[ok], [2.5, 97.5])]
zlo, zhi = [float(v) for v in np.percentile(qq[ok], [2.5, 97.5])]
os.makedirs(f"{C}/draws", exist_ok=True)
DR = f"{C}/draws/p26_kimi_edaic_enc_draws.npz"
np.savez(DR, diff=dd, probe_mean5=pp, zeroshot=qq, point_diff=gap, point_probe=probe, point_zs=ans, per_repeat=np.array(per),
         speakers_sorted=u, n_boot=NB, seed=0,
         zeroshot_rerun_extraction=qr, diff_vs_rerun_extraction=dre, point_zs_rerun_extraction=ans_re, point_diff_vs_rerun_extraction=gap_re)
# per-clip file
PC = f"{C}/p26_kimi_edaic_enc_perclip.csv"
out = M[["clip", "speaker", "label"] + cols + ["p_single"]].copy(); out["p_yes_zeroshot"] = zs; out["p_yes_rerun_extraction"] = zs_re
for t in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]: out[f"fold_{t}"] = M[f"fold_{t}"].values
out.to_csv(PC, index=False, float_format="%.17g")
pyc = json.load(open(PYC))
zst = np.load(NPZ, allow_pickle=True)
side = {
 "cell": "Kimi-Audio E-DAIC (MDD), 30 s window", "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
 "script": os.path.abspath(__file__), "command": "/usr/local/bin/python3 " + " ".join(os.path.abspath(a) if os.path.exists(a) else a for a in sys.argv),
 "probe_stream": "encoder probe: Kimi-Audio continuous-feature Whisper encoder (whisper-large-v3 folder of the Kimi-Audio-7B-Instruct checkpoint), 32 layer outputs, mean over the 1500 frames Kimi keeps",
 "method_change_note": "NEW in PART 26. No Kimi encoder states existed before; every earlier Kimi run hooked only the language model and the release Method says the Kimi encoder side is skipped. These states come from new forward hooks added to release podD2_final/kimi_probe.py (podfiles/gpu/kimi_enc_probe.py). Whether the paper uses this cell is an author decision.",
 "probe_auc_mean5": probe, "per_repeat_auc": per, "per_repeat_auc_4dp": [round(v, 4) for v in per], "single_split_auc": single,
 "layers_chosen": {t: J["tags"][t]["layers"] for t in J["tags"]},
 "answer_auc": ans, "answer_master_lookup": {"data_row_1based": 25, "auc": float(zr["auc"]), "source": zr["source"], "estimator": zr["estimator"]},
 "gap": gap, "gap_ci95": [lo, hi], "interval_excludes_zero": bool(lo > 0 or hi < 0), "interval_above_zero": bool(lo > 0),
 "probe_ci95": [plo, phi], "answer_ci95": [zlo, zhi], "p_boot_one_sided_gap_le0": float(np.mean(dd[ok] <= 0)),
 "boot_usable": int(ok.sum()), "boot_total": NB,
 "sensitivity_vs_rerun_extraction_zeroshot": {"note": "not the headline. Same probe, answer side taken from the p_yes of the PART 26 GPU re-extraction forward pass (the pass that produced the encoder states) instead of the canonical overnight2/part3 zero-shot file; same draws (same idx per draw)",
     "answer_auc": ans_re, "gap": gap_re, "gap_ci95": [rlo, rhi], "interval_excludes_zero": bool(rlo > 0 or rhi < 0)},
 "bootstrap": "paired speaker bootstrap, 2000 draws, fresh numpy default_rng(0), idx = rng.choice(n_spk, size=n_spk, replace=True) over np.unique of the probe file speaker ids, all clips of each drawn speaker, both terms in the same draw, mean of the five per-repeat AUCs minus the zero-shot AUC, percentiles 2.5/97.5; one-class draws dropped",
 "n": int(len(y)), "n_pos": int(y.sum()), "n_speakers": int(len(u)),
 "files": {"perclip_csv": PC, "draws_npz": DR, "probe_oof_csv": os.path.abspath(OOF), "probe_oof_sha256": sha(OOF),
           "nested_json": os.path.abspath(NJ), "nested_json_sha256": sha(NJ), "states_npz": os.path.abspath(NPZ), "states_sha256": sha(NPZ),
           "zeroshot_csv": ZS, "zeroshot_sha256": sha(ZS), "master_lookup_sha256": sha(ML)},
 "states": {k: [list(zst[k].shape), str(zst[k].dtype)] for k in zst.files},
 "fold_files": {t: {"file": J["tags"][t]["fold_file"], "sha256": J["tags"][t]["fold_file_sha256"],
                    "equals_pod_tie_rule": J["tags"][t]["fold_file_equals_pod_tie_rule"]} for t in J["tags"]},
 "probe_run_versions": J["versions"], "probe_model": "StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), layer chosen by mean inner GroupKFold(4) AUC inside each outer training fold (p25a_nested5.py, unchanged)",
 "pyes_rerun_check": pyc,
 "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
}
json.dump(side, open(PC.replace(".csv", ".sidecar.json"), "w"), indent=1)
print(f"GAP kimi edaic enc probe {probe:.4f} per {[round(v, 4) for v in per]} answer {ans:.4f} gap {gap:+.4f} [{lo:+.4f}, {hi:+.4f}] usable {int(ok.sum())}")
