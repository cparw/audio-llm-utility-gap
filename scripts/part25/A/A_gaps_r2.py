"""PART25 A retry (transformers 5.5.4, the original omni-final environment): paired probe-minus-answer gap on the paper's
estimator (mean of five per-repeat nested encoder AUCs), Qwen2.5-Omni, PC-GITA, NeuroVoz, ADReSSo, ADReSS-2020, from the
per-repeat out-of-fold vectors saved on the retry pods (p25a_nested5.py), plus Pitt from part10 pitt_enc_nested5_oof.npz.

Bootstrap: 2000 draws, a fresh numpy default_rng(0) per cell; each draw resamples the unique speakers (np.unique order)
with replacement, idx = rng.choice(n_spk, size=n_spk, replace=True), and takes every clip of every drawn speaker.
In each draw the five per-repeat AUCs (sklearn roc_auc_score) are averaged (mean-of-five), the zero-shot answer AUC is
computed on the same rows, and the difference is taken (paired). 2.5 and 97.5 percentiles.
Zero-shot scores: the saved paper files omni_final/omni_<ds>_zeroshot_scores.csv (read only).
Writes under retry_tf554/: per_clip/A_<ds>_perclip.csv + .sidecar.json, draws/A_<ds>_draws.npz, A_gaps_r2.tsv,
A_gaps_r2.sidecar.json.   usage: A_gaps_r2.py [datasets...]"""
import os, sys, csv, json, hashlib, datetime, platform, numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

R = "<local data dir>/release_from_mac/scores/part25/A/retry_tf554"
OF = "<local data dir>/release/omni_final"
PITT = "<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz"
TEXT = {"pcgita": 0.33, "neurovoz": 0.23, "adresso": 0.26, "adress2020": 0.18}
NAMES = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "adresso": "ADReSSo", "adress2020": "ADReSS-2020", "pitt": "Pitt"}
POD = {d: f"{R}/pull/p25Ar-{d}" for d in TEXT}
if os.environ.get("A_POD_OVERRIDE"):  # e.g. adress2020=/path/to/pull/dir
    for kv in os.environ["A_POD_OVERRIDE"].split(","): k, v = kv.split("="); POD[k] = v
NB = 2000
os.makedirs(f"{R}/per_clip", exist_ok=True); os.makedirs(f"{R}/draws", exist_ok=True)

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
    ok = ~np.isnan(zb); m5 = rep.mean(axis=1); d = m5 - zb
    pc = lambda v: (float(np.percentile(v[ok], 2.5)), float(np.percentile(v[ok], 97.5)))
    return dict(rep=rep, zs=zb, mean5=m5, diff=d, idx=idxs, usable=int(ok.sum()), speakers=u,
                ci_diff=pc(d), ci_mean5=pc(m5), ci_zs=pc(zb))

def zs_map(ds):
    f = f"{OF}/omni_{ds}_zeroshot_scores.csv"
    return f, {r["clip"]: r for r in csv.DictReader(open(f))}

def grep1(path, key):
    try:
        for l in open(path):
            if key in l: return l.strip()
    except OSError: return None

out_rows = []
side = {"task": "PART25 A retry: paired probe minus zero-shot gaps on the paper's mean-of-five estimator, states re-extracted "
                "in the original omni-final environment (H200 image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, transformers 5.5.4)",
        "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "bootstrap": {"n_draws": NB, "rng": "numpy default_rng(0), fresh per cell",
                      "resample": "unique speakers (np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
                      "paired": "probe (mean of the five per-repeat AUCs) and zero-shot AUC computed on the same draw",
                      "percentiles": [2.5, 97.5], "auc": "sklearn.metrics.roc_auc_score",
                      "zeroshot_source": "saved paper files omni_final/omni_<ds>_zeroshot_scores.csv"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
        "command": "/usr/local/bin/python3 " + " ".join(sys.argv), "datasets": {}}
DS = sys.argv[1:] or ["pcgita", "neurovoz", "adresso", "adress2020"]
for ds in DS:
    pod = POD[ds]
    oofp = f"{pod}/out/r2_{ds}_enc_nested5_oof.csv"; jp = f"{pod}/out/r2_{ds}_enc_nested5.json"
    vj = f"{pod}/out/r2_{ds}_podverify.json"; zcj = f"{pod}/out/r2_{ds}_zs_compare.json"; oj = f"{pod}/out/r2orig_{ds}_nested_repeats.json"
    J = json.load(open(jp)); V = json.load(open(vj)) if os.path.exists(vj) else {}
    ZC = json.load(open(zcj)); O = json.load(open(oj))["enc"] if os.path.exists(oj) else None
    rows = list(csv.DictReader(open(oofp)))
    n = len(rows); names = [r["clip"] for r in rows]
    y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
    P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
    Ps = np.array([float(r["p_single"]) for r in rows])
    zf, zmap = zs_map(ds)
    assert set(zmap) == set(names) and len(zmap) == n, f"{ds}: zero-shot clip set differs"
    assert all(int(zmap[c]["label"]) == int(l) and str(zmap[c]["speaker"]) == str(s) for c, l, s in zip(names, y, spk)), f"{ds}: labels/speakers differ"
    zs = np.array([float(zmap[c]["p_yes"]) for c in names])
    sf = f"{OF}/omni_{ds}_enc_nested_oof.csv"; smap = {r["clip"]: float(r["p_probe"]) for r in csv.DictReader(open(sf))}
    ps_saved = np.array([smap[c] for c in names])
    ezs = f"{pod}/out/r2_omni_{ds}_zeroshot_scores.csv"
    rz = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(ezs))}; zs_rerun = np.array([rz[c] for c in names])
    T = json.load(open(f"{OF}/omni_{ds}_nested_repeats.json"))["enc"]
    per = [float(roc_auc_score(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); zauc = float(roc_auc_score(y, zs))
    B = boot(y, spk, P, zs)
    np.savez_compressed(f"{R}/draws/A_{ds}_draws.npz", diff=B["diff"], mean5=B["mean5"], zeroshot=B["zs"], per_repeat=B["rep"],
                        speaker_idx=B["idx"], speakers=B["speakers"], point_diff=np.array(m5 - zauc),
                        point_diff_table1=np.array(T["mean"] - zauc))
    pc = f"{R}/per_clip/A_{ds}_perclip.csv"
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "speaker", "label"] + [f"p_probe_seed{s}" for s in range(5)] + [f"fold_seed{s}" for s in range(5)] +
                   ["p_probe_single_rerun", "fold_single", "p_probe_single_saved", "p_yes_zeroshot_saved", "p_yes_zeroshot_rerun"])
        for i, r in enumerate(rows):
            w.writerow([r["clip"], r["speaker"], r["label"]] + [r[f"p_seed{s}"] for s in range(5)] + [r[f"fold_seed{s}"] for s in range(5)] +
                       [r["p_single"], r["fold_single"], repr(float(ps_saved[i])), repr(float(zs[i])), repr(float(zs_rerun[i]))])
    saved_layers = json.load(open(f"{OF}/omni_{ds}_nested.json"))["enc"]["chosen_layers"]
    rec = {"dataset": NAMES[ds], "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
           "probe_per_repeat_rerun": per, "probe_mean5_rerun": m5,
           "probe_per_repeat_table1": T["per_repeat"], "probe_mean5_table1": T["mean"],
           "mean5_rerun_minus_table1": m5 - T["mean"], "matches_table1_4dp": round(m5, 4) == T["mean"],
           "per_repeat_4dp_equal_table1": [round(a, 4) == b for a, b in zip(per, T["per_repeat"])],
           "original_script_nested_repeats_all_on_rerun_states": O,
           "original_script_matches_table1_json": (O == T) if O else None,
           "zeroshot_auc_saved": zauc, "zeroshot_auc_rerun_extraction": float(roc_auc_score(y, zs_rerun)),
           "zeroshot_rerun_vs_saved": ZC,
           "single_split_auc_rerun": float(roc_auc_score(y, Ps)), "single_split_auc_saved": float(roc_auc_score(y, ps_saved)),
           "single_split_max_abs_p_diff_vs_saved_oof": float(np.max(np.abs(Ps - ps_saved))),
           "single_split_layers_rerun": J["tags"]["single"]["layers"], "single_split_layers_saved": saved_layers,
           "gap_point_rerun": m5 - zauc, "gap_ci": list(B["ci_diff"]), "gap_point_table1": T["mean"] - zauc,
           "probe_mean5_ci": list(B["ci_mean5"]), "zeroshot_ci": list(B["ci_zs"]), "usable_draws": B["usable"],
           "text_single_split_gap": TEXT[ds], "interval_excludes_zero": bool(B["ci_diff"][0] > 0 or B["ci_diff"][1] < 0),
           "layers_rerun": {f"seed{s}": J["tags"][f"seed{s}"]["layers"] for s in range(5)},
           "pod_versions": J["versions"], "pod_env": grep1(f"{pod}/logs/pip.log", "VERSIONS"), "podverify_pass": V.get("PASS"),
           "files": {"perclip_csv": pc, "draws_npz": f"{R}/draws/A_{ds}_draws.npz", "pod_oof_csv": oofp, "pod_json": jp,
                     "pod_verify_json": vj, "zeroshot_saved": zf, "single_split_saved": sf, "zs_compare": zcj, "original_script_json": oj,
                     "table1_json": f"{OF}/omni_{ds}_nested_repeats.json", "rerun_zeroshot": ezs,
                     "rerun_encstates": f"{pod}/out/r2_omni_{ds}_encstates.npz"},
           "sha256": {k: sha(v) for k, v in {"pod_oof_csv": oofp, "zeroshot_saved": zf, "table1_json": f"{OF}/omni_{ds}_nested_repeats.json",
                                             "perclip_csv": pc, "rerun_encstates": f"{pod}/out/r2_omni_{ds}_encstates.npz"}.items()}}
    json.dump(rec, open(f"{R}/per_clip/A_{ds}_perclip.sidecar.json", "w"), indent=1)
    side["datasets"][ds] = rec
    out_rows.append([NAMES[ds], n, rec["n_speakers"], f"{T['mean']:.4f}", f"{m5:.4f}", f"{m5 - T['mean']:+.4f}",
                     " ".join(f"{a:.4f}" for a in per), " ".join(f"{a:.4f}" for a in T["per_repeat"]), f"{zauc:.4f}",
                     f"{m5 - zauc:+.4f}", f"{B['ci_diff'][0]:.4f}", f"{B['ci_diff'][1]:.4f}", f"{T['mean'] - zauc:+.4f}",
                     f"{TEXT[ds]:.2f}", "yes" if rec["interval_excludes_zero"] else "no", B["usable"],
                     f"{ZC['p_yes_exact_equal']}/{ZC['n']}", "PASS" if V.get("PASS") else "CHECK", pc])
    print(f"{NAMES[ds]:12s} rerun mean5 {m5:.4f} (Table 1 {T['mean']:.4f}, {m5 - T['mean']:+.4f}) per {[round(a,4) for a in per]} zs-exact {ZC['p_yes_exact_equal']}/{ZC['n']}  "
          f"gap {m5 - zauc:+.4f} [{B['ci_diff'][0]:.4f}, {B['ci_diff'][1]:.4f}]  Table1 point {T['mean'] - zauc:+.4f}  text {TEXT[ds]:.2f}", flush=True)

z = np.load(PITT, allow_pickle=True); yP = z["label"].astype(int); sP = z["spk"].astype(str); nP = z["name"].astype(str)
zf, zmap = zs_map("pitt"); zsP = np.array([float(zmap[c]["p_yes"]) for c in nP])
perP = [float(roc_auc_score(yP, z["oof"][r])) for r in range(5)]; m5P = float(np.mean(perP)); zP = float(roc_auc_score(yP, zsP))
BP = boot(yP, sP, z["oof"], zsP)
np.savez_compressed(f"{R}/draws/A_pitt_draws.npz", diff=BP["diff"], mean5=BP["mean5"], zeroshot=BP["zs"], per_repeat=BP["rep"],
                    speaker_idx=BP["idx"], speakers=BP["speakers"], point_diff=np.array(m5P - zP))
side["pitt_reference"] = {"file": PITT, "sha256": sha(PITT), "per_repeat": perP, "mean5": m5P, "zeroshot": zP, "gap": m5P - zP,
                          "gap_ci": list(BP["ci_diff"]), "usable_draws": BP["usable"], "expected": "+0.1128 [0.0455, 0.1768]",
                          "reproduced_4dp": [round(m5P - zP, 4), round(BP["ci_diff"][0], 4), round(BP["ci_diff"][1], 4)] == [0.1128, 0.0455, 0.1768]}
out_rows.append(["Pitt (reference)", len(yP), int(len(np.unique(sP))), "0.7706", f"{m5P:.4f}", f"{m5P - 0.7706:+.4f}",
                 " ".join(f"{a:.4f}" for a in perP), "", f"{zP:.4f}", f"{m5P - zP:+.4f}", f"{BP['ci_diff'][0]:.4f}",
                 f"{BP['ci_diff'][1]:.4f}", f"{0.7706 - zP:+.4f}", "", "yes" if BP["ci_diff"][0] > 0 else "no", BP["usable"], "saved part10 OOF",
                 "reproduces +0.1128 [0.0455, 0.1768]" if side["pitt_reference"]["reproduced_4dp"] else "DOES NOT reproduce", PITT])
print(f"Pitt (ref)   mean5 {m5P:.4f} zs {zP:.4f} gap {m5P - zP:+.4f} [{BP['ci_diff'][0]:.4f}, {BP['ci_diff'][1]:.4f}]", flush=True)
hdr = ["dataset", "n", "n_speakers", "probe_mean5_table1", "probe_mean5_rerun", "rerun_minus_table1", "per_repeat_rerun",
       "per_repeat_table1", "zeroshot_auc", "gap_point_rerun", "gap_lo", "gap_hi", "gap_point_table1", "gap_in_text_single_split",
       "interval_excludes_zero", "usable_draws", "zeroshot_p_yes_exact_equal_to_saved", "pod_check", "per_clip_file"]
tsv = f"{R}/A_gaps_r2.tsv"
old = {}
if os.path.exists(tsv):
    for r in csv.reader(open(tsv), delimiter="\t"):
        if r and r[0] != "dataset": old[r[0]] = r
for r in out_rows: old[r[0]] = r
order = ["PC-GITA", "NeuroVoz", "ADReSSo", "ADReSS-2020", "Pitt (reference)"]
with open(tsv, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(hdr)
    for k in order + [k for k in old if k not in order]:
        if k in old: w.writerow(old[k])
sp = f"{R}/A_gaps_r2.sidecar.json"
prev = json.load(open(sp)) if os.path.exists(sp) else {"datasets": {}}
prev["datasets"].update(side["datasets"]); side["datasets"] = prev["datasets"]
json.dump(side, open(sp, "w"), indent=1)
print("wrote", tsv)
