"""PART 26, Qwen3-Omni-30B-A3B on PC-GITA: paired probe minus zero-shot answer gap on the paper's estimator.

Probe: the per-repeat out-of-fold scores of the nested encoder probe (layer chosen inside the training folds by mean
inner AUC), rerun on the PART 26 CPU pod with p25a_nested5.py (PART 25 A script, unchanged) on the saved release fold
files pcgita_groupkfold5_pod_seed0..4.csv, states = part16 POD3 re-extraction (master row 168/169 source).
Answer: zero-shot p_yes from overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv (master row 172, read only).

Bootstrap (same code path as part25/A/scripts/A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True),
takes every clip of every drawn speaker; in each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged and
the zero-shot AUC on the same rows is subtracted. 2.5 and 97.5 percentiles.
Writes q3o_pcgita_perclip.csv, q3o_pcgita_perclip.sidecar.json, q3o_pcgita_draws.npz in the cell folder.
usage: /usr/local/bin/python3 gap26.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np, sklearn
from sklearn.metrics import roc_auc_score

W = "scores/part26/Qwen3-Omni-30B-A3B_PC-GITA"
POD = f"{W}/pull/p26-q3o-pcgita"
OOF = f"{POD}/out/p26_q3o_pcgita_enc_nested5_oof.csv"; JS = f"{POD}/out/p26_q3o_pcgita_enc_nested5.json"
PV = f"{POD}/out/p26_q3o_pcgita_podverify.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"
SRC = "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz"
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

J = json.load(open(JS)); V = json.load(open(PV)) if os.path.exists(PV) else {}
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zr) == n and set(zr) == set(names), "zero-shot clip set differs"
assert all(int(zr[c]["label"]) == int(l) and zr[c]["speaker"] == s for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zr[c]["p_yes"]) for c in names])
# the states' speaker ids and labels in the same clip order (the bootstrap groups on these)
z = np.load(SRC, allow_pickle=True); sn = z["name"].astype(str)
assert list(sn) == names and np.array_equal(z["spk"].astype(str), spk) and np.array_equal(z["label"].astype(int), y)
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
DR = f"{W}/q3o_pcgita_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_probe_mean5=np.array(m5),
                    point_zeroshot=np.array(zauc))
PC = f"{W}/q3o_pcgita_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] +
                   [r[f"fold_seed{s}"] for s in range(5)] + [r["p_single"], r["fold_single"], zr[r["clip"]]["p_yes"]])
rec = {"task": "PART 26 probe grid: Qwen3-Omni-30B-A3B on PC-GITA, paired encoder probe (mean of five per-repeat nested "
               "AUCs) minus zero-shot answer AUC, 2000-draw speaker bootstrap",
       "date_utc": datetime.datetime.utcnow().isoformat() + "Z", "model": "Qwen3-Omni-30B-A3B", "dataset": "PC-GITA",
       "condition": "PD", "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "nested encoder probe: outer GroupKFold(5) folds read from the saved release fold files "
                              "(pcgita_groupkfold5_pod_seed0..4.csv), inner GroupKFold(4) with the pod tie rule on the "
                              "renumbered training speakers picks the encoder layer by mean inner AUC, then StandardScaler + "
                              "LogisticRegression(max_iter=2000, class_weight='balanced') refit on the outer training rows; "
                              "value = mean over the five repeats of the out-of-fold AUC",
                 "script": "p25a_nested5.py (PART 25 A, md5 615ced0380cca2636ed7cc30c71810e6), unchanged",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_auc_not_used": float(roc_auc_score(y, Ps)),
                 "fold_files": {f"seed{s}": [J["tags"][f"seed{s}"]["fold_file"], J["tags"][f"seed{s}"]["fold_file_sha256"]] for s in range(5)},
                 "fold_file_equals_pod_tie_rule": all(J["tags"][f"seed{s}"]["fold_file_equals_pod_tie_rule"] for s in range(5)),
                 "pod_versions": J["versions"], "pod_seconds": J["seconds"], "pod_command": J["command"], "n_jobs": J["n_jobs"],
                 "states_on_pod": J["states"], "podverify_pass": V.get("PASS"),
                 "podverify": {k: V.get(k) for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff")}},
       "zeroshot": {"file": ZS, "sha256": sha(ZS), "auc": zauc, "auc_4dp": round(zauc, 4), "master_lookup_row": 172, "master_value": 0.5967},
       "gap": {"point": m5 - zauc, "ci": list(B["ci_diff"]), "point_4dp": round(m5 - zauc, 4),
               "ci_4dp": [round(B["ci_diff"][0], 4), round(B["ci_diff"][1], 4)],
               "interval_above_zero": bool(B["ci_diff"][0] > 0), "usable_draws": B["usable"],
               "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]),
               "p_boot_diff_le_0": float(np.mean(B["diff"][~np.isnan(B["diff"])] <= 0))},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh",
                     "resample": "unique speakers (np.unique order of the states spk = fold speaker_id), idx = rng.choice(n_spk, "
                                 "size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs and the zero-shot AUC on the same draw; gap = difference",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "states": {"source": SRC, "note": "part16 POD3 re-extraction, the master row 168/169 source; the pod read an enc-only copy "
                  "(enc, label, spk, name, p_yes arrays byte-identical, checked in verify/preflight.json)"},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": JS, "pod_verify_json": PV},
       "versions_mac": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
rec["sha256"] = {"perclip_csv": sha(PC), "draws_npz": sha(DR), "pod_oof_csv": sha(OOF), "pod_json": sha(JS)}
json.dump(rec, open(f"{W}/q3o_pcgita_perclip.sidecar.json", "w"), indent=1)
print(f"Qwen3-Omni PC-GITA probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} zs {zauc:.4f} "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']}", flush=True)
