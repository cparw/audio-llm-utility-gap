"""PART26, cell Qwen3-Omni-30B-A3B x ADReSS-2020: paired probe minus zero-shot answer gap on the paper's estimator
(mean of the five per-repeat nested encoder AUCs), from the per-repeat out-of-fold vectors saved by the PART26 CPU pod
run of p25a_nested5.py (the PART25 A Qwen2.5-Omni nested probe script, unchanged).

Bootstrap, same code path as part25/A/scripts/A_gaps.py boot(): 2000 draws, a fresh numpy default_rng(0); each draw
resamples the unique speakers (np.unique order) with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and
takes every clip of every drawn speaker. In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged, the
zero-shot answer AUC is computed on the same rows, and the difference is taken. 2.5 and 97.5 percentiles over usable draws.
Zero-shot per-clip file: overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv (the C_build_cells.csv zs_file, read only).
Writes under part26/Qwen3-Omni-30B-A3B_ADReSS-2020/: per_clip/p26_q3o_adress2020_perclip.csv (+ .sidecar.json),
draws/p26_q3o_adress2020_draws.npz, cell_result.tsv (p26_verify.py later adds a verification block to the sidecar).   usage: /usr/local/bin/python3 p26_gap.py"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn, scipy
from sklearn.metrics import roc_auc_score

R = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
POD = f"{R}/pull/p26-q3o-adress2020"
OOF = f"{POD}/out/p26_q3o_adress2020_enc_nested5_oof.csv"
JS = f"{POD}/out/p26_q3o_adress2020_enc_nested5.json"
VJ = f"{POD}/out/p26_q3o_adress2020_podverify.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv"
ZSJ = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_zeroshot.json"
MASTER = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_nested_repeats.json"
SINGLE_SAVED = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_enc_nested_oof.csv"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
NB = 2000
os.makedirs(f"{R}/per_clip", exist_ok=True); os.makedirs(f"{R}/draws", exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def boot(y, spk, P, zs):
    """P: (5, n) per-repeat OOF scores; zs: (n,) zero-shot p_yes. Identical to part25/A/scripts/A_gaps.py boot()."""
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


J = json.load(open(JS)); V = json.load(open(VJ)) if os.path.exists(VJ) else {}
rows = list(csv.DictReader(open(OOF)))
n = len(rows); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
Ps = np.array([float(r["p_single"]) for r in rows])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert len(zmap) == n and set(zmap) == set(names), "zero-shot clip set differs from the probe clip set"
assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), "labels/speakers differ"
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
smap = {r["clip"]: float(r["p_probe"]) for r in csv.DictReader(open(SINGLE_SAVED))}; ps_saved = np.array([smap[c] for c in names])
T = json.load(open(MASTER))["enc"]

per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
assert all(abs(a - b) < 1e-12 for a, b in zip(per, J["per_repeat_auc"])), "per-repeat AUC differs from the pod json"
B = boot(y, spk, P, zs)
DR = f"{R}/draws/p26_q3o_adress2020_draws.npz"
np.savez_compressed(DR, diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"], speaker_idx=B["idx"],
                    speakers=B["speakers"], point_diff=np.array(m5 - zauc), point_probe_mean5=np.array(m5),
                    point_zeroshot=np.array(zauc), ci_diff=np.array(B["ci_diff"]))
PC = f"{R}/per_clip/p26_q3o_adress2020_perclip.csv"
with open(PC, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
               ["p_probe_single_rerun", "fold_single", "p_probe_single_saved", "p_yes_zeroshot"])
    for i, r in enumerate(rows):
        w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                   [r["p_single"], r["fold_single"], repr(float(ps_saved[i])), zmap[r["clip"]]["p_yes"]])
rho = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
rec = {"task": "PART26 encoder probe grid, cell Qwen3-Omni-30B-A3B x ADReSS-2020",
       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "model": "Qwen3-Omni-30B-A3B", "checkpoint": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "dataset": "ADReSS-2020", "condition": "AD",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "probe": {"estimator": "nested encoder probe: outer = saved pod-order fold files seed0..4 (speakers renumbered with default_rng(seed)); "
                              "inner GroupKFold(4) with the stable tie rule picks the layer by mean inner AUC; StandardScaler + "
                              "LogisticRegression(max_iter=2000, class_weight='balanced') refit at that layer; value = mean of the five per-repeat AUCs",
                 "script": "p25a_nested5.py (part25/A/podfiles copy, md5 615ced0380cca2636ed7cc30c71810e6, unchanged)",
                 "per_repeat_auc": per, "mean5": m5, "per_repeat_4dp": [round(a, 4) for a in per], "mean5_4dp": round(m5, 4),
                 "layers": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
                 "fold_files": {f"seed{s}": J["tags"][f"seed{s}"]["fold_file"] for s in range(5)},
                 "fold_file_sha256": {f"seed{s}": J["tags"][f"seed{s}"]["fold_file_sha256"] for s in range(5)},
                 "master_lookup_per_repeat": T["per_repeat"], "master_lookup_mean": T["mean"],
                 "rerun_minus_master_mean": m5 - T["mean"],
                 "per_repeat_abs_diff_vs_master_4dp": [abs(round(a, 4) - b) for a, b in zip(per, T["per_repeat"])],
                 "single_split_auc_rerun": float(roc_auc_score(y, Ps)), "single_split_auc_saved": float(roc_auc_score(y, ps_saved)),
                 "single_split_layers_rerun": J["tags"]["single"]["layers"],
                 "single_split_rank_corr_rerun_vs_saved": rho(Ps, ps_saved)},
       "answer": {"file": ZS, "sha256": sha(ZS), "auc": zauc, "auc_4dp": round(zauc, 4),
                  "zeroshot_json_auc": json.load(open(ZSJ))["auc"], "prompt": rows and zmap[names[0]]["prompt"]},
       "gap": {"point": m5 - zauc, "ci_2_5": B["ci_diff"][0], "ci_97_5": B["ci_diff"][1], "usable_draws": B["usable"],
               "interval_above_zero": bool(B["ci_diff"][0] > 0), "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
               "p_boot_one_sided_le0": float(np.mean(B["diff"][~np.isnan(B["diff"])] <= 0)),
               "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"])},
       "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh for this cell",
                     "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                     "paired": "mean of the five per-repeat AUCs minus the zero-shot answer AUC, same rows in every draw",
                     "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score", "code": "part25/A/scripts/A_gaps.py boot(), copied"},
       "pod": {"name": "p26-q3o-adress2020", "id": "marzjz3lnmpdhs", "versions": J["versions"], "n_jobs": J["n_jobs"],
               "seconds": J["seconds"], "podverify_pass": V.get("PASS"), "podverify": {k: V.get(k) for k in
               ("max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "abs_mean5_diff", "mean5_rank_from_oof")}},
       "states": {"file": STATES, "sha256_mac": sha(STATES), "sha256_pod_json": J["states"]["sha256"], "keys": J["states"]["keys"]},
       "mac_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
       "files": {"perclip_csv": PC, "draws_npz": DR, "pod_oof_csv": OOF, "pod_json": JS, "pod_verify_json": VJ,
                 "zeroshot": ZS, "master_json": MASTER, "single_split_saved": SINGLE_SAVED},
       "sha256": {"pod_oof_csv": sha(OOF), "pod_json": sha(JS), "perclip_csv": sha(PC), "draws_npz": sha(DR), "zeroshot": sha(ZS)},
       "command": "/usr/local/bin/python3 " + " ".join(sys.argv)}
json.dump(rec, open(f"{PC[:-4]}.sidecar.json", "w"), indent=1)
with open(f"{R}/cell_result.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["model", "dataset", "n", "n_speakers", "probe_mean5", "per_repeat", "answer_auc", "gap", "gap_lo", "gap_hi",
                "interval_above_zero", "usable_draws", "podverify", "perclip_csv", "draws_npz"])
    w.writerow(["Qwen3-Omni-30B-A3B", "ADReSS-2020", n, rec["n_speakers"], f"{m5:.4f}", " ".join(f"{a:.4f}" for a in per), f"{zauc:.4f}",
                f"{m5 - zauc:+.4f}", f"{B['ci_diff'][0]:+.4f}", f"{B['ci_diff'][1]:+.4f}", "yes" if B["ci_diff"][0] > 0 else "no",
                B["usable"], "PASS" if V.get("PASS") else "CHECK", PC, DR])
print(f"Qwen3-Omni ADReSS-2020 probe mean5 {m5:.4f} per {[round(a, 4) for a in per]} (master {T['mean']:.4f})  answer {zauc:.4f}  "
      f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:+.4f}, {B['ci_diff'][1]:+.4f}]  usable {B['usable']}  podverify {V.get('PASS')}", flush=True)
