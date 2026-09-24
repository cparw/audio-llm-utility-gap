"""PART26 Qwen3-Omni-30B-A3B Pitt: paired probe-minus-answer gap on the paper's estimator (mean of the five per-repeat
nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod rerun (p26_nested5.py on the
part16 POD3 states and the saved Control15-id fold files).

Bootstrap: the PART25 A code path (A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw resamples the
unique speakers (np.unique order of the probe file speaker column) with replacement, idx = rng.choice(n_spk, size=n_spk,
replace=True), and takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score)
are averaged, the zero-shot answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Zero-shot scores: overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv (the master_lookup row 179 file, read only).
Also computed, for reference only: the same bootstrap with the zero-padded speaker ids of the zero-shot file (same
partition, different np.unique order), and the gap from the saved POD3 OOF file behind the Table 1 value 0.8122.
Writes under part26/Qwen3-Omni-30B-A3B_Pitt/: per_clip/*_perclip.csv and .sidecar.json, draws/*_draws.npz,
Qwen3-Omni-30B-A3B_Pitt_gap.json and .tsv.   usage: /usr/local/bin/python3 p26_gap.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen3-Omni-30B-A3B_Pitt"
CELL = "Qwen3-Omni-30B-A3B_Pitt"
POD = f"{R}/pull/p26-q3o-pitt"
OOF = f"{POD}/out/p26_q3o_pitt_enc_nested5_oof.csv"; PJ = f"{POD}/out/p26_q3o_pitt_enc_nested5.json"
PV = f"{POD}/out/p26_q3o_pitt_podverify.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"
ZSJ = "<local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot.json"
STATES = "<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz"
STAGED = f"{R}/stage/q3o_pitt_encstates.npz"
POD3 = "<local data dir>/release/edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv"
T1J = "<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_nested_repeats.json"
FOLDS = "<local data dir>/release/folds"
NB = 2000
os.makedirs(f"{R}/per_clip", exist_ok=True); os.makedirs(f"{R}/draws", exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot(y, spk, P, zs):
    """A_gaps.py boot(), unchanged. P: (5, n) per-repeat OOF scores; zs: (n,) zero-shot p_yes."""
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

J = json.load(open(PJ)); V = json.load(open(PV)) if os.path.exists(PV) else {}
rows = list(csv.DictReader(open(OOF))); n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
F = np.array([[int(r[f"fold_seed{s}"]) for r in rows] for s in range(5)])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zmap) == set(names) and len(zmap) == n, "zero-shot clip set differs"
assert all(int(zmap[c]["label"]) == int(l) for c, l in zip(names, y)), "labels differ"
spk_pad = np.array([zmap[c]["speaker"] for c in names])
pairs = set(zip(spk, spk_pad)); bij = len(pairs) == len(set(spk)) == len(set(spk_pad))
assert bij, "speaker ids are not one to one between the probe file and the zero-shot file"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
z = np.load(STATES, allow_pickle=True)
assert list(z["name"].astype(str)) == names and np.array_equal(z["label"].astype(int), y) and list(z["spk"].astype(str)) == list(spk)
p3 = {r["clip"]: r for r in csv.DictReader(open(POD3))}
P3 = np.array([[float(p3[c][f"p_probe_rep{s}"]) for c in names] for s in range(5)])
T1 = json.load(open(T1J))["enc"]

per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
per3 = [float(roc_auc_score(y, P3[s])) for s in range(5)]; m53 = float(np.mean(per3))
B = boot(y, spk, P, zs)
Bpad = boot(y, spk_pad, P, zs)
B3 = boot(y, spk, P3, zs)
DR = f"{R}/draws/{CELL}_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc),
                    sens_padded_ids_diff=Bpad["diff"], sens_padded_ids_speakers=Bpad["speakers"],
                    ref_pod3_oof_diff=B3["diff"], ref_pod3_oof_mean5=B3["mean5"])
PC = f"{R}/per_clip/{CELL}_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "speaker_padded", "label"] + [f"p_probe_seed{s}" for s in range(5)] +
               [f"fold_seed{s}" for s in range(5)] + ["p_yes_zeroshot"] + [f"p_probe_pod3_rep{s}" for s in range(5)])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], spk_pad[i], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] +
                   [r[f"fold_seed{s}"] for s in range(5)] + [zmap[r["clip"]]["p_yes"]] + [p3[r["clip"]][f"p_probe_rep{s}"] for s in range(5)])
ext = lambda b: bool(b["ci_diff"][0] > 0 or b["ci_diff"][1] < 0)
rec = {"task": "PART26 Qwen3-Omni-30B-A3B Pitt: nested five-repeat encoder probe (saved folds) and paired probe minus zero-shot gap",
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "model": "Qwen3-Omni-30B-A3B", "dataset": "Pitt", "condition": "AD", "n": n, "n_speakers": int(len(np.unique(spk))),
       "n_pos": int(y.sum()),
       "probe": {"estimator": "mean of five per-repeat AUCs; outer folds = saved release fold files pitt_groupkfold5_pod_Control15ids_seed0..4; "
                              "inner GroupKFold(4) with the pod tie rule picks the encoder layer by mean inner AUC; StandardScaler + "
                              "LogisticRegression(max_iter=2000, class_weight='balanced') refit on the outer training rows",
                 "per_repeat_auc": per, "mean5": m5, "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "table1_value": 0.8122, "table1_per_repeat": T1["per_repeat"], "table1_source_pod3_oof_per_repeat": per3,
                 "table1_source_pod3_oof_mean5": m53, "mean5_minus_table1_pod3": m5 - m53,
                 "max_abs_clip_diff_vs_pod3_oof": float(np.max(np.abs(P - P3))),
                 "median_abs_clip_diff_vs_pod3_oof": float(np.median(np.abs(P - P3)))},
       "answer": {"file": ZS, "auc": zauc, "json_auc": json.load(open(ZSJ))["auc"], "window_seconds": 30,
                  "note": "the POD3 states carry their own p_yes from a separate run (differs by up to 0.38); the answer side uses the q3o_new file, as master_lookup does"},
       "gap": {"point": m5 - zauc, "lo": B["ci_diff"][0], "hi": B["ci_diff"][1], "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "interval_excludes_zero": ext(B), "usable_draws": B["usable"], "p_boot_diff_le0": B["p_le0"],
               "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"])},
       "sensitivity_padded_speaker_ids": {"lo": Bpad["ci_diff"][0], "hi": Bpad["ci_diff"][1], "usable_draws": Bpad["usable"],
                                          "interval_above_zero": bool(Bpad["ci_diff"][0] > 0),
                                          "note": "same 228-speaker partition, np.unique order of the zero-padded ids of the zero-shot file"},
       "reference_pod3_oof_gap": {"point": m53 - zauc, "lo": B3["ci_diff"][0], "hi": B3["ci_diff"][1],
                                  "part25C_reported": "+0.0503 [-0.0087, +0.1095]",
                                  "reproduces_part25C_4dp": [round(m53 - zauc, 4), round(B3["ci_diff"][0], 4), round(B3["ci_diff"][1], 4)] == [0.0503, -0.0087, 0.1095]},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh per bootstrap", "resample":
                     "unique speakers (np.unique order of the probe file speaker column, Control15-style ids, 228 labels), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "probe (mean of the five per-repeat AUCs) and zero-shot AUC on the same draw", "percentiles": [2.5, 97.5],
                     "auc": "sklearn.metrics.roc_auc_score", "code": "PART25 A A_gaps.py boot(), unchanged"},
       "pod": {"versions": J["versions"], "n_jobs": J["n_jobs"], "seconds": J["seconds"], "command": J["command"],
               "podverify_pass": V.get("PASS"), "podverify": {k: V.get(k) for k in ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff", "mean5_rank_from_oof")},
               "fold_files": {t: {k: J["tags"][t][k] for k in ("fold_file", "fold_file_sha256", "fold_file_equals_pod_tie_rule", "fold_file_equals_this_pod_sklearn")} for t in J["tags"]}},
       "states": {"file": STATES, "sha256": sha(STATES), "staged_enc_only": STAGED, "staged_sha256": sha(STAGED), "pod_states_sha256": J["states"]["sha256"],
                  "note": "part16 POD3 re-extraction, 30 s window, enc 468 x 32 x 1280 mean-pooled; staged copy holds the enc, label, spk, p_yes and name arrays byte-identical"},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV, "zeroshot": ZS,
                 "pod3_oof": POD3, "table1_json": T1J},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "zeroshot": sha(ZS), "pod3_oof": sha(POD3), "perclip_csv": sha(PC), "draws_npz": sha(DR)},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{R}/per_clip/{CELL}_perclip.sidecar.json", "w"), indent=1)
json.dump(rec, open(f"{R}/{CELL}_gap.json", "w"), indent=1)
hdr = ["model", "dataset", "n", "n_speakers", "probe_mean5", "per_repeat", "table1_value", "answer_auc", "gap", "lo", "hi",
       "interval_above_zero", "usable_draws", "podverify", "per_clip_file", "draws_file"]
with open(f"{R}/{CELL}_gap.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(hdr)
    w.writerow(["Qwen3-Omni-30B-A3B", "Pitt", n, rec["n_speakers"], f"{m5:.4f}", " ".join(f"{a:.4f}" for a in per), "0.8122",
                f"{zauc:.4f}", f"{m5 - zauc:+.4f}", f"{B['ci_diff'][0]:.4f}", f"{B['ci_diff'][1]:.4f}",
                "yes" if B["ci_diff"][0] > 0 else "no", B["usable"], "PASS" if V.get("PASS") else "CHECK", PC, DR])
print(f"probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (Table 1 POD3 {m53:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']}  "
      f"padded-id order [{Bpad['ci_diff'][0]:.4f}, {Bpad['ci_diff'][1]:.4f}]  POD3-OOF gap {m53 - zauc:+.4f} [{B3['ci_diff'][0]:.4f}, {B3['ci_diff'][1]:.4f}]  "
      f"podverify {V.get('PASS')}", flush=True)
