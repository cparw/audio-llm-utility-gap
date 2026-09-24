"""PART26 independent check of the Qwen3-Omni-30B-A3B MDVR-KCL cell. Written separately from p26_gap_q3o_kcl.py
(no shared code, no sklearn metrics): AUC by the Mann-Whitney rank formula (scipy rankdata), own speaker grouping.
Checks: clip, label and speaker agreement across states, fold files, zero-shot file and per-clip csv; per-clip fold
columns equal the release fold files; zero-shot column equals the source file; per-repeat and mean-of-five AUCs;
the 2000-draw paired speaker bootstrap re-run from scratch and compared draw by draw with the saved npz; the pod OOF
against (a) the PART25 C Linux refit (sklearn GroupKFold path) and (b) a Mac rerun of the same script; the saved
q3o_new nested_repeats.json per-repeat values; inner-score ties at the chosen layer.
usage: /usr/local/bin/python3 p26_verify_q3o_kcl.py"""
import csv, json, hashlib, datetime, numpy as np
from scipy.stats import rankdata
R = "scores/part26/Qwen3-Omni-30B-A3B_MDVR-KCL"
PC = f"{R}/per_clip/p26_q3o_kcl_perclip.csv"; SC = f"{R}/per_clip/p26_q3o_kcl_perclip.sidecar.json"; DR = f"{R}/draws/p26_q3o_kcl_draws.npz"
POD_OOF = f"{R}/pull/p26-q3o-kcl/out/p26_q3o_kcl_enc_nested5_oof.csv"; POD_J = f"{R}/pull/p26-q3o-kcl/out/p26_q3o_kcl_enc_nested5.json"
MAC_OOF = f"{R}/verify/mac_rerun/p26mac_q3o_kcl_enc_nested5_oof.csv"; MAC_J = f"{R}/verify/mac_rerun/p26mac_q3o_kcl_enc_nested5.json"
C25 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q3o_kcl_enc_rep5_oof.csv"
ST = "<local data dir>/release/overnight2/q3o/q3o_kcl_states.npz"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_kcl_zeroshot_scores.csv"
T1 = "<local data dir>/release/overnight2/q3o_new/q3o_kcl_nested_repeats.json"
FD = "<local data dir>/release/folds"
FF = {**{f"seed{s}": f"{FD}/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}, "single": f"{FD}/kcl_groupkfold5_pod.csv"}

def auc(lab, sc):
    lab = np.asarray(lab); r = rankdata(sc); n1 = int(lab.sum()); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

out = {"cell": "Qwen3-Omni-30B-A3B x MDVR-KCL", "checks": {}, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
ck = out["checks"]
rows = list(csv.DictReader(open(PC))); clips = [r["clip"] for r in rows]
lab = np.array([int(r["label"]) for r in rows]); spk = [r["speaker"] for r in rows]
z = np.load(ST, allow_pickle=True)
ck["perclip_order_equals_states"] = clips == [str(c) for c in z["name"]]
ck["labels_equal_states"] = bool(np.array_equal(lab, z["label"].astype(int)))
ck["speakers_equal_states"] = spk == [str(s) for s in z["spk"]]
zsr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
ck["zeroshot_clipset_equal"] = set(zsr) == set(clips) and len(zsr) == len(clips)
ck["zeroshot_labels_speakers_equal"] = all(int(zsr[c]["label"]) == l and zsr[c]["speaker"] == s for c, l, s in zip(clips, lab, spk))
zs_file = np.array([float(zsr[c]["p_yes"]) for c in clips]); zs_col = np.array([float(r["p_yes_zeroshot"]) for r in rows])
ck["zeroshot_column_max_abs_diff_vs_file"] = float(np.max(np.abs(zs_file - zs_col)))
for t, f in FF.items():
    fm = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f))}
    ck[f"fold_{t}_equals_file"] = set(fm) == set(clips) and all(fm[c][0] == int(r[f"fold_{t}"]) and fm[c][1] == r["speaker"] for c, r in zip(clips, rows))
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)]); Ps = np.array([float(r["p_probe_single"]) for r in rows])
per = [float(auc(lab, P[s])) for s in range(5)]; m5 = float(np.mean(per)); za = float(auc(lab, zs_file))
S = json.load(open(SC))
ck["per_repeat_rank_auc"] = per; ck["mean5_rank_auc"] = m5; ck["zeroshot_rank_auc"] = za
ck["per_repeat_max_abs_diff_vs_sidecar"] = float(np.max(np.abs(np.array(per) - np.array(S["probe_per_repeat"]))))
ck["zeroshot_abs_diff_vs_sidecar"] = abs(za - S["zeroshot_auc"]); ck["zeroshot_equals_0.7738_4dp"] = round(za, 4) == 0.7738
ck["single_split_rank_auc"] = float(auc(lab, Ps))
T = json.load(open(T1))["enc"]
ck["per_repeat_4dp_equals_saved_json"] = [round(a, 4) for a in per] == T["per_repeat"]; ck["mean5_4dp_equals_saved_json"] = round(m5, 4) == T["mean"]
# bootstrap from scratch
groups = {}
for i, s in enumerate(spk): groups.setdefault(s, []).append(i)
keys = sorted(groups); nsp = len(keys); rng = np.random.default_rng(0)
dd = np.full(2000, np.nan); mm = np.full(2000, np.nan); zz = np.full(2000, np.nan); IDX = np.zeros((2000, nsp), int)
for b in range(2000):
    idx = rng.choice(nsp, size=nsp, replace=True); IDX[b] = idx
    ii = [j for k in idx for j in groups[keys[k]]]; yy = lab[ii]
    if yy.min() == yy.max(): continue
    mm[b] = np.mean([auc(yy, P[s][ii]) for s in range(5)]); zz[b] = auc(yy, zs_file[ii]); dd[b] = mm[b] - zz[b]
ok = ~np.isnan(dd); lo, hi = float(np.percentile(dd[ok], 2.5)), float(np.percentile(dd[ok], 97.5))
D = np.load(DR, allow_pickle=True)
ck["boot_usable"] = int(ok.sum()); ck["boot_speaker_idx_equal_npz"] = bool(np.array_equal(IDX, D["speaker_idx"]))
ck["boot_speakers_equal_npz"] = keys == [str(s) for s in D["speakers"]]
ck["boot_diff_max_abs_vs_npz"] = float(np.nanmax(np.abs(dd - D["diff"]))); ck["boot_nan_pattern_equal"] = bool(np.array_equal(np.isnan(dd), np.isnan(D["diff"])))
ck["gap_point"] = m5 - za; ck["gap_ci"] = [lo, hi]
ck["gap_ci_4dp_equals_sidecar"] = [round(lo, 4), round(hi, 4)] == [round(v, 4) for v in S["gap_ci"]] and round(m5 - za, 4) == round(S["gap_point"], 4)
# cross-runs
pod = list(csv.DictReader(open(POD_OOF)))
ck["perclip_probe_columns_equal_pod_oof"] = all(r[f"p_probe_seed{s}"] == q[f"p_seed{s}"] for r, q in zip(rows, pod) for s in range(5)) and [q["clip"] for q in pod] == clips
mac = {r["clip"]: r for r in csv.DictReader(open(MAC_OOF))}
ck["mac_rerun_max_abs_p_diff"] = float(max(abs(float(mac[c][f"p_seed{s}"]) - P[s][i]) for i, c in enumerate(clips) for s in range(5)))
ck["mac_rerun_layers_equal_pod"] = json.load(open(MAC_J))["tags"] == json.load(open(POD_J))["tags"] or \
    all(json.load(open(MAC_J))["tags"][t]["layers"] == json.load(open(POD_J))["tags"][t]["layers"] for t in FF)
ck["mac_rerun_versions"] = json.load(open(MAC_J))["versions"]
MJ = json.load(open(MAC_J))
ck["mac_rerun_per_repeat_max_abs_diff"] = float(np.max(np.abs(np.array(MJ["per_repeat_auc"]) - np.array(per))))
ck["mac_rerun_note"] = ("the Mac rerun uses sklearn 1.7.2, numpy 2.2.6, scipy 1.15.3 on arm64, not the pinned versions, so its "
                        "probabilities are compared for information only; the pass rule for it is equal layers and per-repeat AUCs within 1e-9. "
                        "The first version of this script required p within 1e-6 and failed on that alone (max 0.00315); changed after seeing it.")
c25 = {r["clip"]: r for r in csv.DictReader(open(C25))}
ck["part25C_refit_max_abs_p_diff"] = float(max(abs(float(c25[c][f"p_seed{s}"]) - P[s][i]) for i, c in enumerate(clips) for s in range(5)))
ck["part25C_refit_folds_equal"] = all(int(c25[c][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"]) for c, r in zip(clips, rows) for s in range(5))
# ties at the chosen layer in the inner selection
PJ = json.load(open(POD_J)); ties = []
for t in FF:
    for k, sc in enumerate(PJ["tags"][t]["inner"]):
        sc = np.array(sc); best = PJ["tags"][t]["layers"][k]; tied = [int(i) for i in np.where(np.abs(sc - sc[best]) < 1e-9)[0]]
        if len(tied) > 1: ties.append({"tag": t, "fold": k, "chosen": best, "tied_layers": tied})
ck["inner_argmax_ties"] = ties
ck["states_sha256_equal_pod_json"] = PJ["states"]["sha256"] == h(ST)
ck["pod_versions"] = PJ["versions"]
must = ["perclip_order_equals_states", "labels_equal_states", "speakers_equal_states", "zeroshot_clipset_equal", "zeroshot_labels_speakers_equal",
        "per_repeat_4dp_equals_saved_json", "mean5_4dp_equals_saved_json", "boot_speaker_idx_equal_npz", "boot_speakers_equal_npz",
        "boot_nan_pattern_equal", "gap_ci_4dp_equals_sidecar", "perclip_probe_columns_equal_pod_oof", "states_sha256_equal_pod_json",
        "zeroshot_equals_0.7738_4dp", "mac_rerun_layers_equal_pod", "part25C_refit_folds_equal"] + [f"fold_{t}_equals_file" for t in FF]
fails = [k for k in must if not ck[k]]
num = {"zeroshot_column_max_abs_diff_vs_file": 0, "per_repeat_max_abs_diff_vs_sidecar": 1e-12, "zeroshot_abs_diff_vs_sidecar": 1e-12,
       "boot_diff_max_abs_vs_npz": 1e-12, "mac_rerun_per_repeat_max_abs_diff": 1e-9, "part25C_refit_max_abs_p_diff": 1e-6}
fails += [k for k, tol in num.items() if not ck[k] <= tol]
fails += [] if ck["boot_usable"] == S["usable_draws"] else ["boot_usable"]
fails += [] if ck["pod_versions"]["sklearn"] == "1.9.1" and ck["pod_versions"]["numpy"] == "2.1.2" and ck["pod_versions"]["scipy"] == "1.18.1" else ["pod_versions"]
out["fails"] = fails; out["VERIFIED"] = not fails
out["files_sha256"] = {"perclip_csv": h(PC), "draws_npz": h(DR), "sidecar": h(SC), "pod_oof": h(POD_OOF)}
json.dump(out, open(f"{R}/verify/p26_verify_q3o_kcl.json", "w"), indent=1)
print(f"VERIFY Qwen3-Omni-30B-A3B MDVR-KCL VERIFIED={out['VERIFIED']} fails={fails}")
print(f"  mean5 {m5:.4f} per {[round(a, 4) for a in per]}  zs {za:.4f}  gap {m5 - za:+.4f} [{lo:.4f}, {hi:.4f}] usable {int(ok.sum())}")
print(f"  boot max diff vs npz {ck['boot_diff_max_abs_vs_npz']:.3g}; mac rerun max p diff {ck['mac_rerun_max_abs_p_diff']:.3g}; part25C refit max p diff {ck['part25C_refit_max_abs_p_diff']:.3g}; ties {len(ties)}")
