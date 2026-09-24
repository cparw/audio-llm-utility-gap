"""PART 26 Qwen3-Omni PC-GITA: independent check of the per-clip csv, the draws and the sidecar written by gap26.py.
Written separately from gap26.py; no shared code. Checks:
 1. per-clip csv vs pod OOF csv: identical scores and folds for every clip; fold columns equal the release fold files
    (read again here by clip) and every speaker sits in one fold.
 2. every AUC recomputed with the Mann-Whitney rank formula (scipy rankdata), point gap recomputed.
 3. the zero-shot column equals the q3o_new file, joined again by clip.
 4. the bootstrap is re-implemented (speaker row lists built with a dict, rank-formula AUC) and every draw compared with
    the saved npz; the percentiles recomputed.
 5. refit every outer fold of every repeat on the Mac at the pod's chosen layer (StandardScaler + LogisticRegression,
    max_iter 2000, balanced), compare predictions with the pod OOF (pod vs Mac numeric tolerance, not exact) and AUCs.
 6. compare with the earlier part16 POD3 per-repeat OOF (same states, same folds; master row 169 values).
 7. pod-side podverify PASS.
usage: /usr/local/bin/python3 verify26.py"""
import os, csv, json, hashlib, datetime, platform, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

W = "scores/part26/Qwen3-Omni-30B-A3B_PC-GITA"
POD = f"{W}/pull/p26-q3o-pcgita/out"
FD = "<local data dir>/release/folds"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"
SRC = "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz"
POD3 = "<local data dir>/release/edaic_rerun/part16/POD3/q3o_pcgita_encoder_nested_oof.csv"
POD3S = "<local data dir>/release/edaic_rerun/part16/POD3/q3o_pcgita_encoder_perlayer_summary.json"

def rauc(yy, s):
    r = rankdata(s); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

S = json.load(open(f"{W}/q3o_pcgita_perclip.sidecar.json"))
D = np.load(f"{W}/q3o_pcgita_draws.npz", allow_pickle=True)
pc = list(csv.DictReader(open(f"{W}/q3o_pcgita_perclip.csv")))
po = {r["clip"]: r for r in csv.DictReader(open(f"{POD}/p26_q3o_pcgita_enc_nested5_oof.csv"))}
J = json.load(open(f"{POD}/p26_q3o_pcgita_enc_nested5.json"))
out = {"date_utc": datetime.datetime.utcnow().isoformat() + "Z", "checks": {}}
C = out["checks"]
names = [r["clip"] for r in pc]; n = len(names)
y = np.array([int(r["label"]) for r in pc]); spk = np.array([r["speaker"] for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in pc] for s in range(5)])
# 1
same = all(r[f"p_probe_seed{s}"] == po[r["clip"]][f"p_seed{s}"] and r[f"fold_seed{s}"] == po[r["clip"]][f"fold_seed{s}"]
           for r in pc for s in range(5)) and len(po) == n
fold_ok = {}
for s in range(5):
    fm = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/pcgita_groupkfold5_pod_seed{s}.csv"))}
    f = np.array([int(r[f"fold_seed{s}"]) for r in pc])
    fold_ok[f"seed{s}"] = bool(set(fm) == set(names) and all(fm[c][0] == int(fv) and fm[c][1] == sp for c, fv, sp in zip(names, f, spk))
                               and all(len(set(f[spk == u])) == 1 for u in np.unique(spk)))
C["1_perclip_equals_pod_oof"] = bool(same); C["1_folds_equal_release_files_and_speaker_disjoint"] = fold_ok
# 2
per = [rauc(y, P[s]) for s in range(5)]; m5 = float(np.mean(per))
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
zs = np.array([float(zr[c]["p_yes"]) for c in names]); za = rauc(y, zs)
C["2_per_repeat_rank"] = per; C["2_mean5_rank"] = m5; C["2_zeroshot_rank"] = za; C["2_gap_rank"] = m5 - za
C["2_matches_sidecar"] = bool(abs(m5 - S["probe"]["mean5"]) < 1e-12 and abs(za - S["zeroshot"]["auc"]) < 1e-12
                              and abs((m5 - za) - S["gap"]["point"]) < 1e-12)
C["2_pod_json_per_repeat_diff"] = float(np.max(np.abs(np.array(per) - np.array(J["per_repeat_auc"]))))
# 3
C["3_zeroshot_column_equal"] = bool(all(float(r["p_yes_zeroshot"]) == float(zr[r["clip"]]["p_yes"]) and int(zr[r["clip"]]["label"]) == int(r["label"]) for r in pc))
# 4
z = np.load(SRC, allow_pickle=True)
C["4_speakers_equal_states_spk"] = bool(list(z["name"].astype(str)) == names and np.array_equal(z["spk"].astype(str), spk))
uniq = sorted(set(spk.tolist())); where = {}
for i, s in enumerate(spk): where.setdefault(s, []).append(i)
rng = np.random.default_rng(0); dd = []; mm = []; zz = []
for b in range(2000):
    pick = rng.choice(len(uniq), size=len(uniq), replace=True)
    ii = np.array([j for k in pick for j in where[uniq[k]]]); yy = y[ii]
    if yy.min() == yy.max(): dd.append(np.nan); mm.append(np.nan); zz.append(np.nan); continue
    a = np.mean([rauc(yy, P[s][ii]) for s in range(5)]); q = rauc(yy, zs[ii]); dd.append(a - q); mm.append(a); zz.append(q)
dd = np.array(dd); ok = ~np.isnan(dd)
C["4_draw_max_abs_diff_vs_saved"] = float(np.nanmax(np.abs(dd - D["diff"])))
C["4_mean5_draw_max_abs_diff"] = float(np.nanmax(np.abs(np.array(mm) - D["mean5"])))
C["4_zs_draw_max_abs_diff"] = float(np.nanmax(np.abs(np.array(zz) - D["zeroshot"])))
ci = [float(np.percentile(dd[ok], 2.5)), float(np.percentile(dd[ok], 97.5))]
C["4_ci_recomputed"] = ci; C["4_usable"] = int(ok.sum())
C["4_ci_matches_sidecar"] = bool(max(abs(ci[0] - S["gap"]["ci"][0]), abs(ci[1] - S["gap"]["ci"][1])) < 1e-9)
# 5
X = z["enc"].astype(np.float32); maxd = 0.0; mac_auc = []
for s in range(5):
    f = np.array([int(r[f"fold_seed{s}"]) for r in pc]); pm = np.zeros(n)
    for k in range(5):
        L = J["tags"][f"seed{s}"]["layers"][k]; tr = np.where(f != k)[0]; te = np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr, L])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, L]), y[tr])
        pm[te] = m.predict_proba(sc.transform(X[te, L]))[:, 1]
    maxd = max(maxd, float(np.max(np.abs(pm - P[s])))); mac_auc.append(rauc(y, pm))
C["5_mac_refit_max_abs_pred_diff"] = maxd; C["5_mac_refit_auc"] = mac_auc
C["5_mac_refit_mean5"] = float(np.mean(mac_auc)); C["5_mac_vs_pod_mean5_diff"] = float(np.mean(mac_auc) - m5)
# 6
p3 = {r["clip"]: r for r in csv.DictReader(open(POD3))}
P3 = np.array([[float(p3[c][f"p_probe_rep{s}"]) for c in names] for s in range(5)])
a3 = [rauc(y, P3[s]) for s in range(5)]
L3 = json.load(open(POD3S))["streams"]["enc"]["nested_chosen_layers"]
C["6_pod3_per_repeat"] = a3; C["6_pod3_mean5"] = float(np.mean(a3))
C["6_rerun_minus_pod3_mean5"] = float(m5 - np.mean(a3))
C["6_per_repeat_abs_diff"] = [abs(a - b) for a, b in zip(per, a3)]
C["6_max_abs_pred_diff_vs_pod3"] = float(np.max(np.abs(P - P3)))
C["6_layers_rerun"] = [J["tags"][f"seed{s}"]["layers"] for s in range(5)]; C["6_layers_pod3"] = L3
C["6_layer_choices_equal"] = [a == b for a, b in zip(C["6_layers_rerun"], L3)]
# 7
V = json.load(open(f"{POD}/p26_q3o_pcgita_podverify.json"))
C["7_podverify_pass"] = V.get("PASS"); C["7_podverify_layer_mismatches"] = V.get("layer_mismatches")
C["7_pod_versions"] = J["versions"]
passed = (C["1_perclip_equals_pod_oof"] and all(fold_ok.values()) and C["2_matches_sidecar"]
          and C["2_pod_json_per_repeat_diff"] < 1e-9 and C["3_zeroshot_column_equal"] and C["4_speakers_equal_states_spk"]
          and C["4_draw_max_abs_diff_vs_saved"] < 1e-9 and C["4_ci_matches_sidecar"] and C["4_usable"] == 2000
          and C["7_podverify_pass"] is True and abs(C["5_mac_vs_pod_mean5_diff"]) < 0.005
          and J["versions"]["sklearn"] == "1.9.1" and J["versions"]["numpy"] == "2.1.2" and J["versions"]["scipy"] == "1.18.1")
out["VERIFIED"] = bool(passed)
out["versions_mac"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
out["sha256"] = {"perclip_csv": sha(f"{W}/q3o_pcgita_perclip.csv"), "draws_npz": sha(f"{W}/q3o_pcgita_draws.npz"),
                 "sidecar": sha(f"{W}/q3o_pcgita_perclip.sidecar.json")}
json.dump(out, open(f"{W}/verify/verify26.json", "w"), indent=1)
print("VERIFIED" if passed else "NOT VERIFIED", json.dumps(C, default=str)[:3000])
