"""PART26 cell Qwen2-Audio E-DAIC (30 s window): paired probe minus zero-shot gap on the mean-of-five estimator.

Probe: the per-repeat out-of-fold scores written by p25a_nested5.py (the PART25 A Qwen2.5-Omni nested encoder probe,
unchanged) on the pod p26-q2a-edaic, from the saved Qwen2-Audio E-DAIC states (enc, 33 stages x 1280) and the saved
release fold files edaic_groupkfold5_pod_seed{0..4}_UNVERIFIED.csv.
Answer: the saved zero-shot per-clip file paper1_local_runs/probe2/edaic_zeroshot_scores.csv (p_yes), read only.

Bootstrap (same code as part25/A/scripts/A_gaps.py boot()): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True),
and takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are
averaged, the zero-shot AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles.
Writes into the cell folder: Qwen2-Audio_E-DAIC_perclip.csv, Qwen2-Audio_E-DAIC_perclip.sidecar.json,
Qwen2-Audio_E-DAIC_draws.npz.   usage: /usr/local/bin/python3 gap26_q2a_edaic.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen2-Audio_E-DAIC"
POD = f"{R}/pull/p26-q2a-edaic"
OOF = f"{POD}/out/p26_q2a_edaic_enc_nested5_oof.csv"
PJ = f"{POD}/out/p26_q2a_edaic_enc_nested5.json"
PV = f"{POD}/out/p26_q2a_edaic_podverify.json"
ZS = "<local data dir>/paper1_local_runs/probe2/edaic_zeroshot_scores.csv"
STATES = "<local data dir>/paper1_local_runs/probe2/edaic_states.npz"
PROV = f"{R}/stage/in/q2a_edaic_encstates.provenance.json"
MASTER = "<local data dir>/paper1_local_runs/omni_final/q2a_edaic_nested_repeats.json"
FOLDDIR = "<local data dir>/release/folds"
NB = 2000
PCSV = f"{R}/Qwen2-Audio_E-DAIC_perclip.csv"; PSIDE = f"{R}/Qwen2-Audio_E-DAIC_perclip.sidecar.json"
DRAWS = f"{R}/Qwen2-Audio_E-DAIC_draws.npz"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def boot(y, spk, P, zs):
    """P: (5, n) per-repeat OOF scores; zs: (n,) zero-shot p_yes. Copied from part25/A/scripts/A_gaps.py."""
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
rows = list(csv.DictReader(open(OOF)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs from the probe clip set"
assert all(int(zmap[c]["label"]) == int(l) for c, l in zip(names, y)), "labels differ"
assert all(str(zmap[c]["speaker"]) == str(s) for c, s in zip(names, spk)), "speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
st = np.load(STATES, allow_pickle=True); sidx = {c: i for i, c in enumerate(st["name"].astype(str))}
p_states = np.array([float(st["p_yes"][sidx[c]]) for c in names])
# fold columns in the OOF csv against the saved release fold files, read again here
fold_check = {}
for t in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]:
    fn = "edaic_groupkfold5_pod.csv" if t == "single" else f"edaic_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fm = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDDIR}/{fn}"))}
    fold_check[t] = {"file": fn, "sha256": sha(f"{FOLDDIR}/{fn}"),
                     "csv_equals_file": all(int(r[f"fold_{t}"]) == fm[r["clip"]] for r in rows)}
M = json.load(open(MASTER))
Menc = M.get("enc", M)
per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
B = boot(y, spk, P, zs)
np.savez_compressed(DRAWS, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"],
                    speaker_idx=B["idx"], speakers=B["speakers"], point_diff=np.array(m5 - zauc),
                    ci_diff=np.array(B["ci_diff"]))
with open(PCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single", "fold_single", "p_yes_zeroshot", "p_yes_states"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(zs[i])), repr(float(p_states[i]))])
rec = {"task": "PART26 probe grid cell: Qwen2-Audio, E-DAIC (first 30 s of dcaps_proc/<pid>.wav), encoder nested probe minus zero-shot answer",
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "model": "Qwen2-Audio", "dataset": "E-DAIC", "condition": "MDD", "window": "30 s (x[:16000*30])",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "nested: outer = saved fold files (five renumbered repeats, pod tie order); inner GroupKFold(4) with the pod tie rule on the outer training speakers picks the encoder stage by mean inner AUC; StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced') refit on the outer training rows at that stage; value = mean of the five per-repeat out-of-fold AUCs",
                 "script": "part25/A/podfiles/p25a_nested5.py (unchanged, sha256 710e3de8d94eaa9cbc0cb3a5c30868415574ad669aa4fa6e0c728edfdc0fb68e)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "single_split_auc": float(roc_auc_score(y, Ps)),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "single_split_layers": J["tags"]["single"]["layers"],
                 "json_per_repeat_auc": J["per_repeat_auc"], "json_mean5": J["mean5"],
                 "master_json": MASTER, "master_per_repeat": Menc.get("per_repeat"), "master_mean": Menc.get("mean"),
                 "pod_versions": J["versions"], "n_jobs": J["n_jobs"], "pod_seconds": J["seconds"],
                 "states_used_on_pod": J["states"], "states_provenance": json.load(open(PROV)),
                 "fold_files": fold_check},
       "answer": {"file": ZS, "sha256": sha(ZS), "auc": zauc,
                  "p_yes_states_vs_file_max_abs_diff": float(np.max(np.abs(p_states - zs)))},
       "gap": {"point": m5 - zauc, "ci": list(B["ci_diff"]), "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]),
               "usable_draws": B["usable"], "interval_above_zero": bool(B["ci_diff"][0] > 0),
               "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0)},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "probe (mean of the five per-repeat AUCs) and zero-shot AUC on the same draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score"},
       "podverify": {"PASS": V.get("PASS"), "max_abs_pred_diff": V.get("max_abs_pred_diff"),
                     "max_abs_inner_score_diff": V.get("max_abs_inner_score_diff"), "layer_mismatches": V.get("layer_mismatches")},
       "pod": {"name": "p26-q2a-edaic", "create_response": f"{R}/launch/create_p26-q2a-edaic_resp.json",
               "id": json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["id"],
               "cpuFlavorId": json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["cpuFlavorId"],
               "vcpu": json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["vcpuCount"],
               "cost_per_hr": json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["costPerHr"],
               "created_utc": json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["createdAt"],
               "deleted": open(f"{R}/wd/p26-q2a-edaic.gone").read().strip() + " UTC (watchdog26.log, REST status 404 after)",
               "est_cost_usd": round(json.load(open(f"{R}/launch/create_p26-q2a-edaic_resp.json"))["costPerHr"] * (10.03 / 60), 2)},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
       "files": {"perclip_csv": PCSV, "draws_npz": DRAWS, "pod_oof_csv": OOF, "pod_json": PJ, "pod_verify_json": PV},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(PJ), "perclip_csv": sha(PCSV), "draws_npz": sha(DRAWS)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(PSIDE, "w"), indent=1)
print(f"Qwen2-Audio E-DAIC probe mean5 {m5:.4f} per {[round(a, 4) for a in per]}  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}] usable {B['usable']}  podverify {V.get('PASS')}", flush=True)
