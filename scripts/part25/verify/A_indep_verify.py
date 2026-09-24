#!/usr/local/bin/python3
"""Independent verifier for PART25 track A.

Written from scratch; imports nothing from track A's scripts. Reads only:
  part25/A/per_clip/A_<ds>_perclip.csv          (canonical per-clip, five OOF repeats + single split)
  part25/A/retry_tf554/pull/<pod>/out/*         (pod OOF, pod rerun zero-shot, original-script JSON)
  part25/A/draws/A_<ds>_draws.npz               (track A's raw draws, compared to mine)
  release/omni_final/*                          (Table 1 JSON, saved zero-shot, saved single split; read only)
  release/folds/*                               (fold files; read only)
  release/overnight2/part10/pitt_enc_nested5_oof.npz
Writes only under part25/verify/.
"""
import csv, json, hashlib, os, sys, datetime, platform
import numpy as np
from sklearn.metrics import roc_auc_score
import sklearn, scipy
from scipy.stats import rankdata

ROOT = "<local data dir>/release_from_mac/scores/part25"
A = f"{ROOT}/A"
R = f"{A}/retry_tf554"
OUT = f"{ROOT}/verify"
OMNI = "<local data dir>/release/omni_final"
FOLDS = "<local data dir>/release/folds"
PITT10 = "<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex"
NB = 2000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def mw_auc(y, s):
    """Mann-Whitney rank AUC, ties averaged. Independent of sklearn."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    r = rankdata(s)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot(spk, y, probe5, zs):
    """probe5: (5, n). Fresh default_rng(0). Unique speakers as str, np.unique order."""
    u = np.unique(spk)
    rows = {k: np.where(spk == k)[0] for k in u}
    rows_list = [rows[k] for k in u]
    rng = np.random.default_rng(0)
    d, m5, z, idxs = [], [], [], []
    for _ in range(NB):
        idx = rng.choice(len(u), size=len(u), replace=True)
        idxs.append(idx)
        r = np.concatenate([rows_list[i] for i in idx])
        yy = y[r]
        if yy.min() == yy.max():
            d.append(np.nan); m5.append(np.nan); z.append(np.nan); continue
        pr = np.mean([roc_auc_score(yy, probe5[k][r]) for k in range(5)])
        zz = roc_auc_score(yy, zs[r])
        m5.append(pr); z.append(zz); d.append(pr - zz)
    d = np.array(d); ok = ~np.isnan(d)
    lo, hi = np.percentile(d[ok], [2.5, 97.5])
    return dict(diff=d, mean5=np.array(m5), zs=np.array(z), idx=np.array(idxs), usable=int(ok.sum()),
                lo=float(lo), hi=float(hi), speakers=u)


def f4(x):
    return f"{x:.4f}"


DS = [("pcgita", "PC-GITA", "p25Ar-pcgita"), ("neurovoz", "NeuroVoz", "p25Ar-neurovoz"),
      ("adresso", "ADReSSo", "p25Ar-adresso"), ("adress2020", "ADReSS-2020", "p25Ar-adress2020-h200")]
CLAIM = {  # from the track A return, the values under test
    "pcgita": dict(gap=0.3306, lo=0.2498, hi=0.4088, n=1100, nspk=100, mean5=0.8941, zs=0.5635, single_gap=0.3454,
                   reps=[0.8882, 0.8900, 0.8926, 0.8954, 0.9044]),
    "neurovoz": dict(gap=0.2262, lo=0.1735, hi=0.2851, n=1270, nspk=107, mean5=0.9213, zs=0.6951, single_gap=0.2209),
    "adresso": dict(gap=0.2602, lo=0.1755, hi=0.3396, n=237, nspk=237, mean5=0.8752, zs=0.6150, single_gap=0.2748),
    "adress2020": dict(gap=0.1802, lo=0.0766, hi=0.2805, n=156, nspk=156, mean5=0.8043, zs=0.6241, single_gap=0.1637),
}

tex = open(TEX).read()
side = dict(task="PART25 independent verifier, track A", date_utc=datetime.datetime.utcnow().isoformat() + "Z",
            versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                          scipy=scipy.__version__, platform=platform.platform()),
            method=dict(auc="sklearn.metrics.roc_auc_score, cross-checked with a scipy rankdata Mann-Whitney AUC",
                        bootstrap="2000 draws, fresh numpy default_rng(0) per cell, idx = rng.choice(n_spk, size=n_spk, replace=True) over np.unique(speaker as str), all clips of each drawn speaker, paired on the same draw, mean of the five per-repeat AUCs per draw, percentiles 2.5/97.5 (numpy linear)"),
            inputs_sha256={}, datasets={})
rows = []
all_t1 = []

for ds, name, pod in DS:
    c = CLAIM[ds]; info = {}; fails = []
    pc_path = f"{A}/per_clip/A_{ds}_perclip.csv"
    pc = read_csv(pc_path); side["inputs_sha256"][pc_path] = sha(pc_path)
    clips = [r["clip"] for r in pc]
    spk = np.array([str(r["speaker"]) for r in pc])
    y = np.array([int(r["label"]) for r in pc])
    P = np.array([[float(r[f"p_probe_seed{k}"]) for r in pc] for k in range(5)])
    # pod OOF: must equal the per-clip columns
    oof_path = f"{R}/pull/{pod}/out/r2_{ds}_enc_nested5_oof.csv"
    oof = read_csv(oof_path); side["inputs_sha256"][oof_path] = sha(oof_path)
    oofd = {r["clip"]: r for r in oof}
    info["pod_oof_clip_set_equal"] = set(oofd) == set(clips) and len(oof) == len(pc)
    P_pod = np.array([[float(oofd[cl][f"p_seed{k}"]) for cl in clips] for k in range(5)])
    info["pod_oof_max_abs_diff_vs_perclip"] = float(np.abs(P_pod - P).max())
    info["pod_oof_labels_equal"] = all(int(oofd[cl]["label"]) == yy for cl, yy in zip(clips, y))
    # Table 1 JSON
    t1_path = f"{OMNI}/omni_{ds}_nested_repeats.json"
    t1 = json.load(open(t1_path))["enc"]; side["inputs_sha256"][t1_path] = sha(t1_path)
    reps = [roc_auc_score(y, P[k]) for k in range(5)]
    reps_mw = [mw_auc(y, P[k]) for k in range(5)]
    mean5 = float(np.mean(reps))
    info["per_repeat_auc"] = reps
    info["per_repeat_mw_max_diff"] = float(np.max(np.abs(np.array(reps) - np.array(reps_mw))))
    info["table1_per_repeat"] = t1["per_repeat"]; info["table1_mean"] = t1["mean"]
    rep_ok = all(round(a, 4) == round(b, 4) for a, b in zip(reps, t1["per_repeat"]))
    m5_ok = round(mean5, 4) == round(t1["mean"], 4) == round(c["mean5"], 4)
    info["mean5"] = mean5; info["per_repeat_4dp_match_table1"] = rep_ok; info["mean5_4dp_match_table1"] = m5_ok
    all_t1.append(rep_ok and m5_ok)
    # original script JSON printed on the pod
    orig_path = f"{R}/pull/{pod}/out/r2orig_{ds}_nested_repeats.json"
    orig = json.load(open(orig_path))["enc"]; side["inputs_sha256"][orig_path] = sha(orig_path)
    info["orig_script_enc_equals_table1_json"] = orig == t1
    # zero-shot: saved paper file
    zs_path = f"{OMNI}/omni_{ds}_zeroshot_scores.csv"
    zsr = read_csv(zs_path); side["inputs_sha256"][zs_path] = sha(zs_path)
    zsd = {r["clip"]: r for r in zsr}
    info["zs_saved_clip_set_equal"] = set(zsd) == set(clips) and len(zsr) == len(pc)
    zs = np.array([float(zsd[cl]["p_yes"]) for cl in clips])
    info["zs_saved_labels_equal"] = all(int(zsd[cl]["label"]) == yy for cl, yy in zip(clips, y))
    info["zs_saved_speakers_equal"] = all(str(zsd[cl]["speaker"]) == s for cl, s in zip(clips, spk))
    info["perclip_zs_saved_col_equal_file"] = bool(np.array_equal(zs, np.array([float(r["p_yes_zeroshot_saved"]) for r in pc])))
    zauc = roc_auc_score(y, zs); info["zeroshot_auc"] = zauc
    info["zeroshot_auc_mw_diff"] = abs(zauc - mw_auc(y, zs))
    zj = json.load(open(f"{OMNI}/omni_{ds}_zeroshot.json"))
    info["zeroshot_json_auc"] = zj["auc"]
    # zero-shot rerun (pod output file) vs saved: exact equality count
    zr_path = f"{R}/pull/{pod}/out/r2_omni_{ds}_zeroshot_scores.csv"
    zrr = read_csv(zr_path); side["inputs_sha256"][zr_path] = sha(zr_path)
    zrd = {r["clip"]: r for r in zrr}
    zrun = np.array([float(zrd[cl]["p_yes"]) for cl in clips])
    info["zs_rerun_exact_equal"] = int((zrun == zs).sum())
    info["zs_rerun_max_abs_diff"] = float(np.abs(zrun - zs).max())
    # single split: saved paper OOF vs rerun
    ss_path = f"{OMNI}/omni_{ds}_enc_nested_oof.csv"
    ssr = read_csv(ss_path); side["inputs_sha256"][ss_path] = sha(ss_path)
    ssd = {r["clip"]: r for r in ssr}
    p_ss_saved = np.array([float(ssd[cl]["p_probe"]) for cl in clips])
    p_ss_rerun = np.array([float(oofd[cl]["p_single"]) for cl in clips])
    info["single_split_rerun_vs_saved_max_abs"] = float(np.abs(p_ss_rerun - p_ss_saved).max())
    ss_auc = roc_auc_score(y, p_ss_saved)
    info["single_split_auc_saved"] = ss_auc
    info["single_split_auc_json"] = json.load(open(f"{OMNI}/omni_{ds}_nested.json"))["enc"]["auc_nested_oof"]
    info["single_split_gap"] = ss_auc - zauc
    nj = json.load(open(f"{R}/pull/{pod}/out/r2_{ds}_enc_nested5.json"))
    info["single_split_layers_saved"] = json.load(open(f"{OMNI}/omni_{ds}_nested.json"))["enc"]["chosen_layers"]
    info["single_split_layers_rerun_keys"] = [k for k in nj.keys() if "single" in k.lower() or "layer" in k.lower()]
    # folds vs release/folds
    fold_ok = []
    for k in range(5):
        cand = [f"{FOLDS}/{ds}_groupkfold5_pod_seed{k}.csv", f"{FOLDS}/{ds}_groupkfold5_pod_seed{k}_UNVERIFIED.csv"]
        fp = [p for p in cand if os.path.exists(p)][0]
        fr = {r["clip_id"]: int(r["fold"]) for r in read_csv(fp)}
        fold_ok.append(all(fr[cl] == int(r[f"fold_seed{k}"]) for cl, r in zip(clips, pc)) and len(fr) == len(pc))
    info["folds_equal_release"] = fold_ok
    # counts
    info["n"] = len(pc); info["n_speakers"] = int(len(np.unique(spk))); info["n_pos"] = int(y.sum())
    # bootstrap, my own
    b = boot(spk, y, P, zs)
    gap = mean5 - zauc
    info.update(gap=gap, lo=b["lo"], hi=b["hi"], usable=b["usable"])
    tz = np.load(f"{A}/draws/A_{ds}_draws.npz")
    info["track_draws_max_abs_diff_vs_mine"] = float(np.nanmax(np.abs(tz["diff"] - b["diff"])))
    info["track_speaker_idx_equal_mine"] = bool(np.array_equal(tz["speaker_idx"], b["idx"]))
    info["track_speakers_equal_mine"] = bool(np.array_equal(tz["speakers"], b["speakers"]))
    info["track_point_diff"] = float(tz["point_diff"])
    np.savez_compressed(f"{OUT}/draws/Vfy_A_{ds}_draws.npz", diff=b["diff"], mean5=b["mean5"], zeroshot=b["zs"],
                        speaker_idx=b["idx"], speakers=b["speakers"], point_diff=gap)
    side["datasets"][ds] = info
    # verdict
    checks = dict(
        gap4=f4(gap) == f4(c["gap"]), lo4=f4(b["lo"]) == f4(c["lo"]), hi4=f4(b["hi"]) == f4(c["hi"]),
        n=info["n"] == c["n"], nspk=info["n_speakers"] == c["nspk"], usable=b["usable"] == 2000,
        mean5=m5_ok, reps=rep_ok, zs4=f4(zauc) == f4(c["zs"]), single4=f4(ss_auc - zauc) == f4(c["single_gap"]),
        pod_oof=info["pod_oof_max_abs_diff_vs_perclip"] == 0.0 and info["pod_oof_clip_set_equal"],
        zs_file=info["zs_saved_clip_set_equal"] and info["zs_saved_labels_equal"] and info["zs_saved_speakers_equal"] and info["perclip_zs_saved_col_equal_file"],
        folds=all(fold_ok), excl0=b["lo"] > 0,
    )
    info["checks"] = checks
    ver = all(checks.values())
    textnote = ""
    rows.append(dict(id=f"A_{ {'pcgita':'PCGITA','neurovoz':'NeuroVoz','adresso':'ADReSSo','adress2020':'ADReSS2020'}[ds]}_gap",
                     claimed=f"{c['gap']:+.4f} [{c['lo']:.4f}, {c['hi']:.4f}]",
                     recomputed=f"{gap:+.4f} [{b['lo']:.4f}, {b['hi']:.4f}]; mean5 {mean5:.4f} (T1 {t1['mean']:.4f}), zs {zauc:.4f}, single-split gap {ss_auc - zauc:+.4f}, n {info['n']}/{info['n_speakers']} spk, usable {b['usable']}",
                     verified="yes" if ver else "no", fails=[k for k, v in checks.items() if not v]))
    print(ds, json.dumps({k: v for k, v in info.items() if k not in ("per_repeat_auc",)}, default=str)[:3000])
    print(ds, "CHECKS", checks)

# ---- Pitt reference from part10 OOF
z = np.load(PITT10, allow_pickle=True); side["inputs_sha256"][PITT10] = sha(PITT10)
names = [str(x) for x in z["name"]]; spk = np.array([str(x) for x in z["spk"]]); y = z["label"].astype(int); P = z["oof"]
zsr = read_csv(f"{OMNI}/omni_pitt_zeroshot_scores.csv"); zsd = {r["clip"]: r for r in zsr}
side["inputs_sha256"][f"{OMNI}/omni_pitt_zeroshot_scores.csv"] = sha(f"{OMNI}/omni_pitt_zeroshot_scores.csv")
pitt = {}
pitt["clip_set_equal"] = set(zsd) == set(names) and len(zsr) == len(names)
zs = np.array([float(zsd[n]["p_yes"]) for n in names])
pitt["labels_equal"] = all(int(zsd[n]["label"]) == yy for n, yy in zip(names, y))
pitt["speakers_equal"] = all(str(zsd[n]["speaker"]) == s for n, s in zip(names, spk))
reps = [roc_auc_score(y, P[k]) for k in range(5)]; mean5 = float(np.mean(reps)); zauc = roc_auc_score(y, zs)
b = boot(spk, y, P, zs); gap = mean5 - zauc
tz = np.load(f"{A}/draws/A_pitt_draws.npz")
pitt.update(n=len(names), n_speakers=int(len(np.unique(spk))), per_repeat=reps, mean5=mean5, zeroshot=zauc, gap=gap,
            lo=b["lo"], hi=b["hi"], usable=b["usable"],
            track_draws_max_abs_diff=float(np.nanmax(np.abs(tz["diff"] - b["diff"]))),
            track_idx_equal=bool(np.array_equal(tz["speaker_idx"], b["idx"])),
            omni_final_pitt_enc_mean=json.load(open(f"{OMNI}/omni_pitt_nested_repeats.json"))["enc"]["mean"],
            tex_pitt_interval_present="0.11 [0.05, 0.18] on Pitt" in tex)
np.savez_compressed(f"{OUT}/draws/Vfy_A_pitt_draws.npz", diff=b["diff"], mean5=b["mean5"], zeroshot=b["zs"],
                    speaker_idx=b["idx"], speakers=b["speakers"], point_diff=gap)
side["datasets"]["pitt"] = pitt
print("pitt", json.dumps(pitt, default=str))
pchk = dict(gap=f4(gap) == "0.1128", lo=f4(b["lo"]) == "0.0455", hi=f4(b["hi"]) == "0.1768", n=pitt["n"] == 468,
            nspk=pitt["n_speakers"] == 228, mean5=f4(mean5) == "0.7706", zs=f4(zauc) == "0.6578",
            files=pitt["clip_set_equal"] and pitt["labels_equal"] and pitt["speakers_equal"])
rows.append(dict(id="A_Pitt_reference", claimed="+0.1128 [0.0455, 0.1768]",
                 recomputed=f"{gap:+.4f} [{b['lo']:.4f}, {b['hi']:.4f}]; mean5 {mean5:.4f}, zs {zauc:.4f}, n {pitt['n']}/{pitt['n_speakers']} speaker labels, usable {b['usable']}",
                 verified="yes" if all(pchk.values()) else "no", fails=[k for k, v in pchk.items() if not v]))
pitt["checks"] = pchk

json.dump(side, open(f"{OUT}/A_VERIFIED_partial.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
json.dump(rows, open(f"{OUT}/A_VERIFIED_rows_partial.json", "w"), indent=1)
for r in rows:
    print(r)
