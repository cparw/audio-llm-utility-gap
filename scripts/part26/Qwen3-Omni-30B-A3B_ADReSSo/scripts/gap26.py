"""PART 26, one cell: Qwen3-Omni-30B-A3B on ADReSSo. Paired probe minus zero-shot answer gap on the paper's estimator
(mean of the five per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART 26 CPU pod
run of p25a_nested5.py (PART 25 A script, unchanged) on the saved release fold files.

Bootstrap (same code path as part25/A/scripts/A_gaps.py): 2000 draws, a fresh numpy default_rng(0); each draw resamples
the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and takes
every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged, the
zero-shot answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot per-clip file: overnight2/q3o_new/q3o_adresso_zeroshot_scores.csv (part25/C C_other_models.tsv and
C_build_cells.csv name it; read only).
Writes under part26/Qwen3-Omni-30B-A3B_ADReSSo/: Qwen3-Omni-30B-A3B_ADReSSo_perclip.csv, _perclip.sidecar.json,
draws/Qwen3-Omni-30B-A3B_ADReSSo_draws.npz.   usage: /usr/local/bin/python3 gap26.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn, scipy
from sklearn.metrics import roc_auc_score

MODEL, DSN, DS = "Qwen3-Omni-30B-A3B", "ADReSSo", "adresso"
R = f"<local data dir>/release_from_mac/scores/part26/{MODEL}_{DSN}"
POD = f"{R}/pull/p26-q3o-adresso"
OOF = f"{POD}/out/p26_q3o_adresso_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_q3o_adresso_enc_nested5.json"
PV = f"{POD}/out/p26_q3o_adresso_podverify.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adresso_zeroshot_scores.csv"
ZSJ = "<local data dir>/release/overnight2/q3o_new/q3o_adresso_zeroshot.json"
T1 = "<local data dir>/release/overnight2/q3o_new/q3o_adresso_nested_repeats.json"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adresso_states.npz"
C25 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q3o_adresso_enc_rep5_oof.csv"
FOLDS = "<local data dir>/release/folds"
NB = 2000
os.makedirs(f"{R}/draws", exist_ok=True)

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

J = json.load(open(PJ)); V = json.load(open(PV)) if os.path.exists(PV) else {}
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
T = json.load(open(T1))["enc"]
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{R}/draws/{MODEL}_{DSN}_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_diff_table1=np.array(T["mean"] - zauc))
# PART 25 C Linux refit (sklearn GroupKFold outer and inner folds) as a clip-by-clip cross-check, read only
c25 = {r["clip"]: r for r in csv.DictReader(open(C25))}
C25P = np.array([[float(c25[c][f"p_seed{s}"]) for c in names] for s in range(5)])
c25_fold_equal = all(all(int(c25[c][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"]) for c, r in zip(names, rows)) for s in range(5))
PC = f"{R}/{MODEL}_{DSN}_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
rec = {"task": "PART 26 probe grid, one cell: paired encoder probe minus zero-shot answer gap",
       "model": MODEL, "dataset": DSN, "condition": "AD", "stream": "enc (32 encoder layers, mean-pooled, 1280 dims)",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe_per_repeat": per, "probe_mean5": m5, "probe_per_repeat_4dp": [round(a, 4) for a in per], "probe_mean5_4dp": round(m5, 4),
       "probe_per_repeat_master": T["per_repeat"], "probe_mean5_master": T["mean"],
       "mean5_minus_master": m5 - T["mean"], "per_repeat_abs_diff_vs_master_4dp": [abs(round(a, 4) - b) for a, b in zip(per, T["per_repeat"])],
       "single_split_auc": float(roc_auc_score(y, Ps)),
       "zeroshot_auc": zauc, "zeroshot_auc_json": json.load(open(ZSJ))["auc"],
       "gap_point": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_point_master": T["mean"] - zauc,
       "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
       "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
       "p_boot_one_sided_le0": float(np.mean(B["diff"][~np.isnan(B["diff"])] <= 0)),
       "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)}, "layers_single": J["tags"]["single"]["layers"],
       "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule",
                                                     "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs", "inner_sk_equal")}
                      for t in J["tags"]},
       "cross_check_part25C_refit": {"file": C25, "sha256": sha(C25), "fold_columns_equal": c25_fold_equal,
                                     "max_abs_oof_diff": float(np.max(np.abs(C25P - P))),
                                     "per_repeat_auc_part25C": [float(roc_auc_score(y, C25P[s])) for s in range(5)]},
       "pod_versions": J["versions"], "pod_n_jobs": J["n_jobs"], "pod_seconds": J["seconds"], "podverify_pass": V.get("PASS"),
       "podverify": {k: V.get(k) for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff")},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "probe (mean of the five per-repeat AUCs) and zero-shot AUC computed on the same draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV,
                 "zeroshot_perclip": ZS, "zeroshot_json": ZSJ, "master_nested_repeats_json": T1, "states": STATES,
                 "script": os.path.abspath(__file__)},
       "sha256": {k: sha(v) for k, v in {"pod_oof_csv": OOF, "pod_json": PJ, "zeroshot_perclip": ZS, "master_nested_repeats_json": T1,
                                         "states": STATES, "perclip_csv": PC, "draws_npz": DR}.items()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{R}/{MODEL}_{DSN}_perclip.sidecar.json", "w"), indent=1)
print(f"{MODEL} {DSN}: probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (master {T['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}]  usable {B['usable']}  podverify {V.get('PASS')}  "
      f"vs part25C max|oof diff| {rec['cross_check_part25C_refit']['max_abs_oof_diff']:.3g}", flush=True)
