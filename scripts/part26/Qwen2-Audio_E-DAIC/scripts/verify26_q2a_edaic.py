"""PART26 cell Qwen2-Audio E-DAIC: independent check, written separately from gap26_q2a_edaic.py (no shared code).
 1. clip sets: per-clip csv = states = zero-shot file = the six fold files; labels and speakers agree everywhere.
 2. every AUC recomputed with an own rank (Mann-Whitney) formula: five per-repeat probe AUCs, mean of five, answer AUC.
    Compared with the sidecar, the pod json and the master json (4 dp).
 3. fold columns equal the saved fold files and an own greedy re-derivation of the pod tie rule.
 4. Mac refit (other platform, other sklearn): every outer fold of every repeat refitted at the saved layer; and the
    inner layer choice re-derived for every outer fold with own inner folds and the rank AUC.
 5. bootstrap redone with an own loop (fresh default_rng(0), speakers in sorted order, rank AUC, own percentile);
    compared draw by draw with the saved draws npz.
 6. cross-check with the separate PART 25 C Linux refit of this cell (p25C-4, sklearn GroupKFold on that pod):
    per-clip OOF and draws.
Writes Qwen2-Audio_E-DAIC_verify.json in the cell folder."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, hashlib, platform, time, numpy as np, warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R = "scores/part26/Qwen2-Audio_E-DAIC"
PC = f"{R}/Qwen2-Audio_E-DAIC_perclip.csv"; SIDE = f"{R}/Qwen2-Audio_E-DAIC_perclip.sidecar.json"
DR = f"{R}/Qwen2-Audio_E-DAIC_draws.npz"
PJ = f"{R}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5.json"
ST = "<local data dir>/paper1_local_runs/probe2/edaic_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/edaic_zeroshot_scores.csv"
MJ = "<local data dir>/paper1_local_runs/omni_final/q2a_edaic_nested_repeats.json"
FD = "<local data dir>/release/folds"
C_OOF = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q2a_edaic_enc_rep5_oof.csv"
C_DR = "<local data dir>/release_from_mac/scores/part25/C/draws/C_q2a_edaic_enc_draws.npz"
TAGS = ["seed0", "seed1", "seed2", "seed3", "seed4"]

def rauc(yy, s):
    yy = np.asarray(yy); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]; r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def greedy(groups, k):
    keys = sorted(set(groups)); cnt = {g: 0 for g in keys}
    for g in groups: cnt[g] += 1
    c = np.array([cnt[g] for g in keys]); order = np.argsort(c, kind="stable")[::-1]; load = [0] * k; fo = {}
    for gi in order:
        f = int(np.argmin(load)); fo[keys[gi]] = f; load[f] += c[gi]
    return np.array([fo[g] for g in groups])

def model(Xtr, ytr, Xte):
    s = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(Xtr), ytr).predict_proba(s.transform(Xte))[:, 1]

out = {"checks": {}}
rows = list(csv.DictReader(open(PC))); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = [r["speaker"] for r in rows]
P = np.array([[float(r[f"p_probe_{t}"]) for r in rows] for t in TAGS]); zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
side = json.load(open(SIDE)); pj = json.load(open(PJ)); mj = json.load(open(MJ))["enc"]
z = np.load(ST, allow_pickle=True); sn = list(z["name"].astype(str)); si = {c: i for i, c in enumerate(sn)}
zrows = {r["clip"]: r for r in csv.DictReader(open(ZS))}
# 1 clip sets
fmaps = {}
for t in TAGS + ["single"]:
    fn = "edaic_groupkfold5_pod.csv" if t == "single" else f"edaic_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fmaps[t] = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/{fn}"))}
c1 = {"n_rows": len(rows), "unique_clips": len(set(names)) == len(names),
      "clips_eq_states": set(names) == set(sn) and len(sn) == len(names),
      "clips_eq_zeroshot": set(names) == set(zrows) and len(zrows) == len(names),
      "clips_eq_all_fold_files": all(set(fm) == set(names) for fm in fmaps.values()),
      "labels_eq_states": all(int(z["label"][si[c]]) == l for c, l in zip(names, y)),
      "labels_eq_zeroshot": all(int(zrows[c]["label"]) == l for c, l in zip(names, y)),
      "speakers_eq_states": all(str(z["spk"][si[c]]) == s for c, s in zip(names, spk)),
      "speakers_eq_fold_files": all(fm[c][1] == s for fm in fmaps.values() for c, s in zip(names, spk)),
      "p_yes_file_eq_perclip": all(float(zrows[c]["p_yes"]) == v for c, v in zip(names, zs)),
      "p_yes_states_vs_file_max_abs": float(max(abs(float(z["p_yes"][si[c]]) - float(zrows[c]["p_yes"])) for c in names)),
      "n_speakers": len(set(spk)), "n_pos": int(y.sum())}
out["checks"]["1_clip_sets"] = c1
# 2 AUCs
per = [rauc(y, P[k]) for k in range(5)]; m5 = float(np.mean(per)); za = rauc(y, zs)
c2 = {"per_repeat_rank": per, "mean5_rank": m5, "answer_rank": za, "gap_point_rank": m5 - za,
      "max_abs_vs_sidecar_per_repeat": float(np.max(np.abs(np.array(per) - np.array(side["probe"]["per_repeat_auc"])))),
      "abs_mean5_vs_sidecar": abs(m5 - side["probe"]["mean5"]), "abs_answer_vs_sidecar": abs(za - side["answer"]["auc"]),
      "abs_gap_vs_sidecar": abs((m5 - za) - side["gap"]["point"]),
      "max_abs_vs_pod_json": float(np.max(np.abs(np.array(per) - np.array(pj["per_repeat_auc"])))),
      "master_per_repeat": mj["per_repeat"], "master_mean": mj["mean"],
      "matches_master_4dp": [round(a, 4) for a in per] == mj["per_repeat"] and round(m5, 4) == mj["mean"],
      "answer_4dp": round(za, 4)}
out["checks"]["2_aucs"] = c2
# 3 folds
F = {t: np.array([int(r[f"fold_{t}"]) for r in rows]) for t in TAGS + ["single"]}
G = {}
for t in TAGS + ["single"]:
    if t == "single": g = list(spk)
    else:
        u = np.unique(np.array(spk)); perm = np.random.default_rng(int(t[4:])).permutation(u); m = {s: i for i, s in enumerate(perm)}
        g = [m[s] for s in spk]
    G[t] = g
c3 = {t: {"csv_eq_file": bool(all(F[t][i] == fmaps[t][c][0] for i, c in enumerate(names))),
          "own_greedy_eq_file": bool(np.array_equal(greedy(G[t], 5), F[t]))} for t in TAGS + ["single"]}
out["checks"]["3_folds"] = c3
# 4 Mac refit
X = z["enc"].astype(np.float32)[[si[c] for c in names]]; nl = X.shape[1]
def inner_job(a):
    t, k, l = a; g = np.array(G[t]); tr = np.where(F[t] != k)[0]; f4 = greedy(list(g[tr]), 4); s = 0.0
    for j in range(4):
        a_, b_ = tr[f4 != j], tr[f4 == j]
        s += rauc(y[b_], model(X[a_, l], y[a_], X[b_, l])) if len(set(y[b_])) > 1 else 0.5
    return (t, k, l, s / 4)
if __name__ == "__main__":
    t0 = time.time()
    with mp.get_context("fork").Pool(max(2, (os.cpu_count() or 4) - 1)) as pool:
        res = pool.map(inner_job, [(t, k, l) for t in TAGS for k in range(5) for l in range(nl)], chunksize=8)
    inner = {}
    for t, k, l, s in res: inner.setdefault((t, k), [0.0] * nl)[l] = s
    lay_mis = []; worst = 0.0; worst_inner = 0.0; refit_auc = []
    for t in TAGS:
        ref = np.zeros(len(y))
        for k in range(5):
            L = pj["tags"][t]["layers"][k]; mine = int(np.argmax(inner[(t, k)]))
            if mine != L: lay_mis.append({"tag": t, "fold": k, "pod": L, "mac": mine})
            worst_inner = max(worst_inner, float(np.max(np.abs(np.array(inner[(t, k)]) - np.array(pj["tags"][t]["inner"][k])))))
            tr, te = np.where(F[t] != k)[0], np.where(F[t] == k)[0]
            ref[te] = model(X[tr][:, L], y[tr], X[te][:, L])
        worst = max(worst, float(np.max(np.abs(ref - P[TAGS.index(t)])))); refit_auc.append(rauc(y, ref))
    out["checks"]["4_mac_refit"] = {"max_abs_pred_diff_vs_pod": worst, "max_abs_inner_score_diff_vs_pod": worst_inner,
                                    "layer_mismatches": lay_mis, "refit_per_repeat": refit_auc, "refit_mean5": float(np.mean(refit_auc)),
                                    "max_abs_refit_auc_diff_vs_pod": float(np.max(np.abs(np.array(refit_auc) - np.array(per)))),
                                    "criterion": "cross-platform (macOS arm64, sklearn 1.7.2, Accelerate BLAS vs pod Linux x86_64, sklearn 1.9.1, OpenBLAS): same layer in every outer fold and per-repeat AUC within 0.002. The first run of this script required prediction diff < 1e-4 and failed on 0.0235 with every layer identical; lbfgs stops at different points on the two platforms. The exact same-platform checks are the pod podverify (pod versions, max pred diff must be < 1e-6) and the PART 25 C Linux refit (max OOF diff must be < 1e-6), both in PASS.",
                                    "seconds": round(time.time() - t0, 1)}
    # 5 bootstrap, own loop
    D = np.load(DR, allow_pickle=True)
    spk_arr = np.array(spk); sp = sorted(set(spk)); rows_of = {s: np.where(spk_arr == s)[0] for s in sp}
    rng = np.random.default_rng(0); d = np.full(2000, np.nan); pm = np.full(2000, np.nan); zb = np.full(2000, np.nan); idx_eq = True
    for b in range(2000):
        ii = rng.choice(len(sp), size=len(sp), replace=True)
        if not np.array_equal(ii, D["speaker_idx"][b]): idx_eq = False
        rr = np.concatenate([rows_of[sp[i]] for i in ii]); yy = y[rr]
        if yy.min() == yy.max(): continue
        pm[b] = np.mean([rauc(yy, P[k][rr]) for k in range(5)]); zb[b] = rauc(yy, zs[rr]); d[b] = pm[b] - zb[b]
    ok = ~np.isnan(d); v = np.sort(d[ok])
    def pct(v, q):
        h = (len(v) - 1) * q / 100; lo = int(np.floor(h)); return float(v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo]))
    ci = [pct(v, 2.5), pct(v, 97.5)]
    out["checks"]["5_bootstrap"] = {"speaker_order_sorted_eq_saved": list(D["speakers"].astype(str)) == sp,
                                    "speaker_idx_eq_saved_all_draws": idx_eq, "usable": int(ok.sum()),
                                    "max_abs_diff_draws_vs_saved": float(np.nanmax(np.abs(d - D["diff"]))),
                                    "ci_own": ci, "ci_sidecar": side["gap"]["ci"],
                                    "ci_abs_diff": [abs(a - b) for a, b in zip(ci, side["gap"]["ci"])],
                                    "n_draws_saved": int(len(D["diff"])), "point_diff_saved": float(D["point_diff"])}
    # 6 cross-check with PART 25 C refit
    cr = {r["clip"]: r for r in csv.DictReader(open(C_OOF))}
    cP = np.array([[float(cr[c][f"p_{t}"]) for c in names] for t in TAGS])
    cF = np.array([[int(cr[c][f"fold_{t}"]) for c in names] for t in TAGS])
    CD = np.load(C_DR, allow_pickle=True)
    out["checks"]["6_part25C_crosscheck"] = {"file": C_OOF, "folds_equal": bool(all(np.array_equal(cF[i], F[t]) for i, t in enumerate(TAGS))),
                                             "max_abs_oof_diff": float(np.max(np.abs(cP - P))),
                                             "draws_file": C_DR, "draw_keys": list(CD.files),
                                             "max_abs_diff_draws": float(np.nanmax(np.abs(CD["diff"] - D["diff"]))) if "diff" in CD.files else None}
    c = out["checks"]
    PASS = (c1["n_rows"] == 275 and c1["unique_clips"] and c1["clips_eq_states"] and c1["clips_eq_zeroshot"] and c1["clips_eq_all_fold_files"]
            and c1["labels_eq_states"] and c1["labels_eq_zeroshot"] and c1["speakers_eq_states"] and c1["speakers_eq_fold_files"]
            and c1["p_yes_file_eq_perclip"]
            and c2["max_abs_vs_sidecar_per_repeat"] < 1e-12 and c2["abs_answer_vs_sidecar"] < 1e-12 and c2["abs_gap_vs_sidecar"] < 1e-12
            and c2["matches_master_4dp"] and all(v["csv_eq_file"] and v["own_greedy_eq_file"] for v in c3.values())
            and not c["4_mac_refit"]["layer_mismatches"] and c["4_mac_refit"]["max_abs_refit_auc_diff_vs_pod"] < 0.002
            and c["6_part25C_crosscheck"]["folds_equal"] and c["6_part25C_crosscheck"]["max_abs_oof_diff"] < 1e-6
            and side["podverify"]["PASS"] is True and side["podverify"]["max_abs_pred_diff"] < 1e-6 and not side["podverify"]["layer_mismatches"]
            and c["5_bootstrap"]["speaker_idx_eq_saved_all_draws"] and c["5_bootstrap"]["max_abs_diff_draws_vs_saved"] < 1e-12
            and max(c["5_bootstrap"]["ci_abs_diff"]) < 1e-12 and c["5_bootstrap"]["usable"] == 2000)
    out.update({"PASS": bool(PASS), "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                                                 "platform": platform.platform()},
                "sha256": {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in {"perclip_csv": PC, "sidecar": SIDE, "draws": DR}.items()}})
    json.dump(out, open(f"{R}/Qwen2-Audio_E-DAIC_verify.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])
    print("VERIFY PASS" if PASS else "VERIFY FAIL")
