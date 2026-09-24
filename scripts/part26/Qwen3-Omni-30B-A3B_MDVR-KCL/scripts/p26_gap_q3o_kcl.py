"""PART26, cell Qwen3-Omni-30B-A3B x MDVR-KCL: paired probe-minus-answer gap on the paper's estimator (mean of the five
per-repeat nested encoder AUCs) from the per-repeat out-of-fold vectors saved by the PART26 CPU pod run of
p25a_nested5.py (the PART25 A script, unchanged). Same bootstrap code path as part25/A/scripts/A_gaps.py boot():
2000 draws, a fresh numpy default_rng(0); each draw resamples the unique speakers (np.unique order) with replacement,
idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every drawn speaker. In each draw: the five
per-repeat AUCs (sklearn roc_auc_score) are averaged, the zero-shot answer AUC is computed on the same rows, and the
difference is taken. 2.5 and 97.5 percentiles. Zero-shot per-clip file: overnight2/q3o_new/q3o_kcl_zeroshot_scores.csv
(the file C_build_cells.csv lists for this cell; read only).
Writes per_clip/p26_q3o_kcl_perclip.csv (+ .sidecar.json), draws/p26_q3o_kcl_draws.npz, p26_q3o_kcl_gap.tsv
(+ .sidecar.json).   usage: /usr/local/bin/python3 p26_gap_q3o_kcl.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen3-Omni-30B-A3B_MDVR-KCL"
POD = f"{R}/pull/p26-q3o-kcl"
OOF = f"{POD}/out/p26_q3o_kcl_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_q3o_kcl_enc_nested5.json"
PV = f"{POD}/out/p26_q3o_kcl_podverify.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_kcl_zeroshot_scores.csv"
ZSJ = "<local data dir>/release/overnight2/q3o_new/q3o_kcl_zeroshot.json"
T1 = "<local data dir>/release/overnight2/q3o_new/q3o_kcl_nested_repeats.json"
STATES = "<local data dir>/release/overnight2/q3o/q3o_kcl_states.npz"
FOLDS = "<local data dir>/release/folds"
NB = 2000
for d in ("per_clip", "draws"): os.makedirs(f"{R}/{d}", exist_ok=True)

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
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb), p_le0=float(np.mean(d[ok] <= 0)))

J = json.load(open(PJ)); V = json.load(open(PV))
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
T = json.load(open(T1))["enc"]
assert J["states"]["sha256"] == sha(STATES), "pod states sha differs from the Mac states file"
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{R}/draws/p26_q3o_kcl_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_mean5=np.array(m5), point_zeroshot=np.array(zauc))
PC = f"{R}/per_clip/p26_q3o_kcl_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
ex0 = bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0)
rec = {"model": "Qwen3-Omni-30B-A3B", "dataset": "MDVR-KCL", "condition": "PD", "n": n, "n_speakers": int(len(np.unique(spk))),
       "n_pos": int(y.sum()), "probe": "encoder, nested (layer chosen by inner GroupKFold(4) out-of-fold AUC inside each outer training fold), "
       "StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), 32 encoder layers, mean pooled states",
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_saved_json": T["per_repeat"], "probe_mean5_saved_json": T["mean"],
       "probe_mean5_minus_saved_json": m5 - T["mean"], "zeroshot_auc": zauc, "zeroshot_json_auc": json.load(open(ZSJ))["auc"],
       "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_interval_excludes_zero": ex0, "gap_interval_above_zero": bool(B["ci_diff"][0] > 0),
       "p_boot_diff_le0": B["p_le0"], "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "single_split_auc": float(roc_auc_score(y, Ps)), "layers": {t: J["tags"][t]["layers"] for t in J["tags"]},
       "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule",
                                                     "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs")} for t in J["tags"]},
       "inner_folds_equal_pod_sklearn": {t: J["tags"][t]["inner_sk_equal"] for t in J["tags"]},
       "pod_versions": J["versions"], "pod_seconds": J["seconds"], "pod_n_jobs": J["n_jobs"], "podverify_pass": V.get("PASS"),
       "podverify": {k: V[k] for k in ("mean5_rank_from_oof", "abs_mean5_diff", "max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches")},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score", "same_code_as": "part25/A/scripts/A_gaps.py boot()"},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV, "zeroshot_perclip": ZS,
                 "zeroshot_json": ZSJ, "saved_nested_repeats_json": T1, "states": STATES, "fold_dir": FOLDS,
                 "probe_script": f"{R}/stage/p25a_nested5.py", "pod_driver": f"{R}/stage/run26.sh"},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "zeroshot_perclip": sha(ZS), "states": sha(STATES), "perclip_csv": sha(PC),
                  "draws_npz": sha(DR), "probe_script": sha(f"{R}/stage/p25a_nested5.py")},
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(PC.replace(".csv", ".sidecar.json"), "w"), indent=1)
TS = f"{R}/p26_q3o_kcl_gap.tsv"
with open(TS, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["model", "dataset", "n", "n_speakers", "probe_mean5", "per_repeat", "zeroshot_auc", "gap_point", "gap_lo", "gap_hi",
                "interval_above_zero", "usable_draws", "podverify", "perclip_file", "draws_file"])
    w.writerow(["Qwen3-Omni-30B-A3B", "MDVR-KCL", n, rec["n_speakers"], f"{m5:.4f}", " ".join(f"{a:.4f}" for a in per), f"{zauc:.4f}",
                f"{m5 - zauc:+.4f}", f"{B['ci_diff'][0]:.4f}", f"{B['ci_diff'][1]:.4f}", "yes" if B["ci_diff"][0] > 0 else "no",
                B["usable"], "PASS" if V.get("PASS") else "CHECK", PC, DR])
json.dump(rec, open(TS.replace(".tsv", ".sidecar.json"), "w"), indent=1)
print(f"Qwen3-Omni-30B-A3B MDVR-KCL probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (saved json {T['mean']:.4f})  "
      f"zs {zauc:.4f}  gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']} podverify {V.get('PASS')}", flush=True)
