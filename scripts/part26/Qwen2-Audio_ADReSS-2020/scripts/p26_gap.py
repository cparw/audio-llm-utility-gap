"""PART26 cell Qwen2-Audio / ADReSS-2020: paired probe-minus-answer gap on the paper's estimator (mean of the five
per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod run of
part25/A/podfiles/p25a_nested5.py (unchanged) on the saved release fold files.

Bootstrap (same code path as part25/A/scripts/A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True),
and takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged,
the zero-shot answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot answer: paper1_local_runs/probe2/adress2020_zeroshot_scores.csv (p_yes; the part25/C C_build_cells zs_file, read only).
Writes, under part26/Qwen2-Audio_ADReSS-2020/: Qwen2-Audio_ADReSS-2020_perclip.csv, Qwen2-Audio_ADReSS-2020.sidecar.json,
Qwen2-Audio_ADReSS-2020_draws.npz.   usage: /usr/local/bin/python3 p26_gap.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

C = "scores/part26/Qwen2-Audio_ADReSS-2020"
POD = f"{C}/pull/p26-q2a-adress2020"
ZS = "<local data dir>/paper1_local_runs/probe2/adress2020_zeroshot_scores.csv"
ZS_SHA_EXPECTED = "6a211f89a4305144e9f17352891f7b2afb08301840a46176a85c896a6723133a"   # part25/C C_build_cells.csv zs_file_sha256
T1 = "<local data dir>/paper1_local_runs/omni_final/q2a_adress2020_nested_repeats.json"
C2 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-2/out/q2a_adress2020_enc_rep5_oof.csv"
SINGLE_SAVED = "<local data dir>/paper1_local_runs/probe2/adress2020_enc_nested_oof.csv"
STATES = "<local data dir>/paper1_local_runs/probe2/adress2020_states.npz"
STAGE = f"{C}/stage/q2a_adress2020_encstates.npz"
NB = 2000; TAG = "Qwen2-Audio_ADReSS-2020"

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

oofp = f"{POD}/out/p26_q2a_adress2020_enc_nested5_oof.csv"; jp = f"{POD}/out/p26_q2a_adress2020_enc_nested5.json"
vj = f"{POD}/out/p26_q2a_adress2020_podverify.json"
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
T = json.load(open(T1))["enc"]
B = boot(y, spk, P, zs)
dr = f"{C}/{TAG}_draws.npz"
np.savez_compressed(dr, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_probe_mean5=np.array(m5),
                    point_zeroshot=np.array(zauc), ci_diff=np.array(B["ci_diff"]))
# earlier independent Linux refit (part25/C p25C-2, sklearn GroupKFold inside the run, not the saved fold files)
c2 = {r["clip"]: r for r in csv.DictReader(open(C2))}
c2_fold_equal = [all(int(c2[c][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"]) for c, r in zip(names, rows)) for s in range(5)]
c2_maxdiff = [float(max(abs(float(c2[c][f"p_seed{s}"]) - float(r[f"p_seed{s}"])) for c, r in zip(names, rows))) for s in range(5)]
ss = {r["clip"]: r for r in csv.DictReader(open(SINGLE_SAVED))}
pcf = f"{C}/{TAG}_perclip.csv"
with open(pcf, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_podtie", "fold_single_podtie", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i]))])
rec = {"cell": "Qwen2-Audio / ADReSS-2020", "model": "Qwen2-Audio-7B-Instruct", "dataset": "ADReSS-2020", "condition": "AD",
       "part": "PART26", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "encoder nested probe, layer chosen inside each outer training fold by mean inner GroupKFold(4) AUC (pod tie rule), StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced') refit at that layer; outer folds = saved release fold files adress2020_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv; value = mean of the five per-repeat out-of-fold AUCs",
                 "script": "part25/A/podfiles/p25a_nested5.py (unchanged, sha256 710e3de8d94eaa9cbc0cb3a5c30868415574ad669aa4fa6e0c728edfdc0fb68e)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_podtie_auc": float(roc_auc_score(y, Ps)), "single_split_layers": J["tags"]["single"]["layers"],
                 "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule",
                                                                  "fold_file_equals_this_pod_sklearn", "n_clips_sklearn_differs")} for t in J["tags"]},
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "states_on_pod": J["states"]},
       "answer": {"file": ZS, "sha256": sha(ZS), "zeroshot_auc": zauc, "zeroshot_auc_4dp": round(zauc, 4), "score": "p_yes"},
       "gap": {"point": m5 - zauc, "point_4dp": round(m5 - zauc, 4), "ci95": list(B["ci_diff"]), "ci95_4dp": [round(v, 4) for v in B["ci_diff"]],
               "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
               "probe_mean5_ci95": list(B["ci_mean5"]), "zeroshot_ci95": list(B["ci_zs"]), "usable_draws": B["usable"],
               "p_boot_diff_le0": float(np.mean(B["diff"][~np.isnan(B["diff"])] <= 0))},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat probe AUCs and the zero-shot answer AUC on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score", "code_path": "part25/A/scripts/A_gaps.py boot(), copied unchanged"},
       "comparisons": {"master_lookup_table1": {"file": T1, "per_repeat": T["per_repeat"], "mean": T["mean"],
                                                "per_repeat_abs_diff_4dp": [abs(round(a, 4) - b) for a, b in zip(per, T["per_repeat"])],
                                                "mean5_matches_4dp": round(m5, 4) == T["mean"]},
                       "part25C_p25C2_linux_refit": {"file": C2, "folds_equal_per_seed": c2_fold_equal, "max_abs_pred_diff_per_seed": c2_maxdiff},
                       "probe2_single_split_saved_mac": {"file": SINGLE_SAVED, "note": "earlier Mac single split (Mac GroupKFold order), a different split from these five; recorded only",
                                                         "auc": float(roc_auc_score(y, [float(ss[c]["p_probe"]) for c in names]))}},
       "podverify": {"file": vj, "PASS": V.get("PASS"), "max_abs_pred_diff": V.get("max_abs_pred_diff"),
                     "max_abs_inner_score_diff": V.get("max_abs_inner_score_diff"), "layer_mismatches": V.get("layer_mismatches"),
                     "abs_mean5_diff": V.get("abs_mean5_diff")},
       "inputs": {"states_source": STATES, "states_stage_enc_only": STAGE, "stage_sha256": sha(STAGE)},
       "files": {"perclip_csv": pcf, "draws_npz": dr, "pod_oof_csv": oofp, "pod_json": jp, "pod_verify_json": vj},
       "sha256": {"pod_oof_csv": sha(oofp), "pod_json": sha(jp), "perclip_csv": sha(pcf), "draws_npz": sha(dr)},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{C}/{TAG}.sidecar.json", "w"), indent=1)
print(f"{TAG}: probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (master {T['mean']:.4f} {T['per_repeat']})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}] usable {B['usable']}  podverify {V.get('PASS')}  "
      f"p25C-2 folds equal {c2_fold_equal} maxdiff {max(c2_maxdiff):.3g}")
