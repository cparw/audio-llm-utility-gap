"""PART 26 Qwen3-Omni E-DAIC: independent check of p26_q3o_edaic_gap.py, written separately (no shared code).
- rank (Mann-Whitney) AUCs with scipy.stats.rankdata instead of sklearn;
- per-clip csv columns against the pod OOF csv, the q3o_new zero-shot file, the release fold files and the states;
- the bootstrap rebuilt with its own loop from the speaker ids (default_rng(0), same draw call), compared draw by draw
  with the saved npz and at 4 dp on point and interval;
- the re-extracted states file: shape, finite values, clip order, labels and speakers against the zero-shot file.
usage: /usr/local/bin/python3 p26_q3o_edaic_verify.py"""
import csv, json, sys, numpy as np, hashlib
from scipy.stats import rankdata
R = "scores/part26/Qwen3-Omni-30B-A3B_E-DAIC"
S = json.load(open(f"{R}/Q3O_EDAIC_perclip.sidecar.json"))
F = S["files"]
def auc(y, s):
    r = rankdata(s); p = y == 1; n1 = p.sum(); n0 = len(y) - n1
    return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows = list(csv.DictReader(open(F["perclip_csv"])))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows]); clips = [r["clip"] for r in rows]
P = np.array([[float(r[f"p_probe_seed{k}"]) for r in rows] for k in range(5)]); zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
chk = {}
oof = {r["clip"]: r for r in csv.DictReader(open(F["pod_oof_csv"]))}
chk["perclip_probe_equals_pod_oof"] = all(float(oof[c][f"p_seed{k}"]) == P[k][i] for i, c in enumerate(clips) for k in range(5))
zz = {r["clip"]: r for r in csv.DictReader(open(F["zeroshot"]))}
chk["perclip_zs_equals_q3o_new"] = all(float(zz[c]["p_yes"]) == zs[i] and int(zz[c]["label"]) == y[i] and zz[c]["speaker"] == spk[i] for i, c in enumerate(clips)) and len(zz) == len(clips)
fo = {}
for k in range(5):
    fo[k] = {r["clip_id"]: (r["fold"], r["speaker_id"]) for r in csv.DictReader(open(f"<local data dir>/release/folds/edaic_groupkfold5_pod_seed{k}_UNVERIFIED.csv"))}
chk["folds_equal_release_files"] = all(fo[k][c][0] == rows[i][f"fold_seed{k}"] and fo[k][c][1] == spk[i] for i, c in enumerate(clips) for k in range(5))
z = np.load(F["rerun_encstates"], allow_pickle=True)
chk["states_shape"] = list(z["enc"].shape); chk["states_finite"] = bool(np.isfinite(z["enc"]).all())
chk["states_order_labels_speakers_equal_zs"] = ([str(v) for v in z["name"]] == [r["clip"] for r in csv.DictReader(open(F["zeroshot"]))]
    and [int(v) for v in z["label"]] == [int(r["label"]) for r in csv.DictReader(open(F["zeroshot"]))]
    and [str(v) for v in z["spk"]] == [r["speaker"] for r in csv.DictReader(open(F["zeroshot"]))])
per = [auc(y, P[k]) for k in range(5)]; m5 = float(np.mean(per)); za = float(auc(y, zs))
chk["per_repeat_rank"] = per; chk["mean5_rank"] = m5; chk["zs_rank"] = za
chk["point_matches_4dp"] = round(m5, 4) == S["probe_mean5_4dp"] and round(za, 4) == S["zeroshot_auc_4dp"] and round(m5 - za, 4) == S["gap_point_4dp"]
chk["per_repeat_max_abs_diff_vs_sidecar"] = float(np.max(np.abs(np.array(per) - np.array(S["probe_per_repeat"]))))
# bootstrap rebuilt
sp_sorted = sorted(set(spk)); members = {s: [i for i in range(len(spk)) if spk[i] == s] for s in sp_sorted}
g = np.random.default_rng(0); D = []
for b in range(2000):
    pick = g.choice(len(sp_sorted), size=len(sp_sorted), replace=True)
    ii = [j for p in pick for j in members[sp_sorted[p]]]; yy = y[ii]
    if yy.min() == yy.max(): D.append(np.nan); continue
    D.append(np.mean([auc(yy, P[k][ii]) for k in range(5)]) - auc(yy, zs[ii]))
D = np.array(D); ok = ~np.isnan(D)
lo, hi = np.percentile(D[ok], [2.5, 97.5])
sv = np.load(F["draws_npz"])
chk["draws_max_abs_diff_vs_saved"] = float(np.nanmax(np.abs(D - sv["diff"])))
chk["draws_nan_pattern_equal"] = bool(np.array_equal(np.isnan(D), np.isnan(sv["diff"])))
chk["interval_rebuilt"] = [float(lo), float(hi)]
chk["interval_matches_4dp"] = [round(float(lo), 4), round(float(hi), 4)] == S["gap_ci_4dp"]
chk["n_speakers"] = len(sp_sorted); chk["usable"] = int(ok.sum())
chk["PASS"] = bool(chk["perclip_probe_equals_pod_oof"] and chk["perclip_zs_equals_q3o_new"] and chk["folds_equal_release_files"]
                   and chk["states_finite"] and chk["states_order_labels_speakers_equal_zs"] and chk["point_matches_4dp"]
                   and chk["interval_matches_4dp"] and chk["draws_max_abs_diff_vs_saved"] < 1e-9 and chk["draws_nan_pattern_equal"]
                   and chk["states_shape"] == [275, 32, 1280])
json.dump(chk, open(f"{R}/verify/Q3O_EDAIC_verify.json", "w"), indent=1)
print("VERIFY PASS" if chk["PASS"] else "VERIFY FAIL", json.dumps({k: v for k, v in chk.items() if k not in ("per_repeat_rank",)}))
