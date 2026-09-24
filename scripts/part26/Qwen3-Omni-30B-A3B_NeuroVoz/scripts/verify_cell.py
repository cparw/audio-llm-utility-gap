"""PART26 cell Qwen3-Omni-30B-A3B x NeuroVoz: independent Mac check of the pod probe and of gap_cell.py. Written
separately from gap_cell.py and shares no code with it. Checks:
 1. per-clip csv rows: clips, labels and speakers equal the source states file and the zero-shot file (read again here);
    fold columns equal the release fold files (read again here); no speaker in two folds; every score finite in [0, 1].
 2. per-repeat AUCs and the zero-shot AUC recomputed with the rank (Mann-Whitney) formula, compared with the pod json
    and the sidecar.
 3. Mac refit of every outer fold at the layer the pod chose (StandardScaler + LogisticRegression(max_iter=2000,
    class_weight='balanced'), the model spec). The Mac has other library versions, so predictions are compared with a
    tolerance and the AUCs are reported, not required to match to 1e-6 (folds README: pod vs Mac up to 0.083).
 4. bootstrap re-implemented: speakers in sorted order, fresh numpy default_rng(0), rng.choice(n, n, replace=True) per
    draw, rank-formula AUCs; every draw's gap compared with the saved draws, and the 2.5 / 97.5 percentiles recomputed.
 5. the pulled files match the pod's own sha256 list; podverify PASS.
usage: /usr/local/bin/python3 verify_cell.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import sys, csv, json, hashlib, datetime, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata

CELL = "Qwen3-Omni-30B-A3B_NeuroVoz"
C = f"<local data dir>/release_from_mac/scores/part26/{CELL}"
REL = "<local data dir>/release"
SRC = f"{REL}/overnight2/q3o/q3o_neurovoz_states.npz"
ZS = f"{REL}/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv"
FD = f"{REL}/folds"
POD = f"{C}/pull/p26-q3o-neurovoz"
PC = f"{C}/per_clip/{CELL}_perclip.csv"; SC = f"{C}/per_clip/{CELL}_perclip.sidecar.json"; DR = f"{C}/draws/{CELL}_draws.npz"
PJ = f"{POD}/out/p26_q3o_neurovoz_enc_nested5.json"; PV = f"{POD}/out/p26_q3o_neurovoz_podverify.json"

def h256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def mw_auc(yy, s):
    r = rankdata(s); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

out = {"cell": CELL, "date_utc": datetime.datetime.utcnow().isoformat() + "Z", "checks": {}}
ck = out["checks"]
rows = list(csv.DictReader(open(PC))); clips = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
S = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)])
Z = np.array([float(r["p_yes_zeroshot"]) for r in rows])
src = np.load(SRC, allow_pickle=True)
sm = {n: i for i, n in enumerate(src["name"].astype(str))}
ck["clips_equal_source_states"] = sorted(clips) == sorted(sm) and len(clips) == len(sm) == 1270
ck["labels_equal_source_states"] = all(int(src["label"][sm[c]]) == int(l) for c, l in zip(clips, y))
ck["speakers_equal_source_states"] = all(str(src["spk"][sm[c]]) == s for c, s in zip(clips, spk))
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
ck["zeroshot_column_equals_file"] = all(float(zr[c]["p_yes"]) == z for c, z in zip(clips, Z))
ck["zeroshot_labels_equal"] = all(int(zr[c]["label"]) == int(l) for c, l in zip(clips, y))
ck["scores_finite_in_0_1"] = bool(np.isfinite(S).all() and (S >= 0).all() and (S <= 1).all())
FOLDF = {f"seed{s}": f"neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}; FOLDF["single"] = "neurovoz_groupkfold5_pod.csv"
folds = {}
for t, f in FOLDF.items():
    fm = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/{f}"))}
    col = "fold_single" if t == "single" else f"fold_{t}"
    fo = np.array([fm[c][0] for c in clips]); folds[t] = fo
    ck[f"folds_{t}_equal_release_file"] = bool(np.array_equal(fo, np.array([int(r[col]) for r in rows]))) and all(fm[c][1] == s for c, s in zip(clips, spk))
    ck[f"folds_{t}_no_speaker_in_two_folds"] = all(len(set(fo[spk == s])) == 1 for s in set(spk))
# 2. AUCs
J = json.load(open(PJ)); SCJ = json.load(open(SC))
per_mw = [mw_auc(y, S[s]) for s in range(5)]; z_mw = mw_auc(y, Z)
out["per_repeat_auc_rank"] = per_mw; out["mean5_rank"] = float(np.mean(per_mw)); out["zeroshot_auc_rank"] = z_mw
out["gap_point_rank"] = float(np.mean(per_mw)) - z_mw
ck["per_repeat_auc_equal_pod_json_1e-12"] = max(abs(a - b) for a, b in zip(per_mw, J["per_repeat_auc"])) < 1e-12
ck["per_repeat_auc_equal_sidecar_1e-12"] = max(abs(a - b) for a, b in zip(per_mw, SCJ["probe_per_repeat"])) < 1e-12
ck["zeroshot_auc_equal_sidecar_1e-12"] = abs(z_mw - SCJ["zeroshot_auc"]) < 1e-12
ck["zeroshot_auc_4dp_is_0.6393"] = round(z_mw, 4) == 0.6393
ck["gap_point_equal_sidecar_1e-12"] = abs(out["gap_point_rank"] - SCJ["gap_point"]) < 1e-12
# 3. Mac refit at the pod's layers
X = src["enc"]; Xi = np.array([sm[c] for c in clips])

def refit(a):
    t, k, L = a
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    fo = folds[t]; tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
    Xtr = X[Xi[tr], L].astype(np.float32); Xte = X[Xi[te], L].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    p = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y[tr]).predict_proba(sc.transform(Xte))[:, 1]
    return t, k, te, p

jobs = [(f"seed{s}", k, J["tags"][f"seed{s}"]["layers"][k]) for s in range(5) for k in range(5)]
res = [refit(j) for j in jobs]   # serial: no fork of the Mac BLAS
R = np.zeros((5, len(y)))
for t, k, te, p in res: R[int(t[4:]), te] = p
out["mac_refit"] = {"per_repeat_auc": [mw_auc(y, R[s]) for s in range(5)], "max_abs_pred_diff_vs_pod": float(np.max(np.abs(R - S))),
                    "median_abs_pred_diff_vs_pod": float(np.median(np.abs(R - S)))}
out["mac_refit"]["max_abs_auc_diff_vs_pod"] = max(abs(a - b) for a, b in zip(out["mac_refit"]["per_repeat_auc"], per_mw))
ck["mac_refit_auc_within_0.005"] = out["mac_refit"]["max_abs_auc_diff_vs_pod"] < 0.005
ck["mac_refit_pred_within_0.1"] = out["mac_refit"]["max_abs_pred_diff_vs_pod"] < 0.1
# 4. bootstrap re-implementation
D = np.load(DR, allow_pickle=True)
us = sorted(set(spk)); members = {s: np.where(spk == s)[0] for s in us}
rng = np.random.default_rng(0); g = np.empty(2000); m5 = np.empty(2000); zb = np.empty(2000); idx_equal = True
for b in range(2000):
    pick = rng.choice(len(us), len(us), replace=True)
    idx_equal &= bool(np.array_equal(pick, D["speaker_idx"][b]))
    ii = np.concatenate([members[us[j]] for j in pick]); yy = y[ii]
    if yy.min() == yy.max(): g[b] = m5[b] = zb[b] = np.nan; continue
    m5[b] = np.mean([mw_auc(yy, S[s][ii]) for s in range(5)]); zb[b] = mw_auc(yy, Z[ii]); g[b] = m5[b] - zb[b]
ok = ~np.isnan(g)
ci = [float(np.percentile(g[ok], 2.5)), float(np.percentile(g[ok], 97.5))]
out["bootstrap_reimpl"] = {"usable": int(ok.sum()), "gap_ci": ci, "max_abs_draw_diff": float(np.nanmax(np.abs(g - D["diff"]))),
                           "max_abs_mean5_draw_diff": float(np.nanmax(np.abs(m5 - D["mean5"]))),
                           "max_abs_zeroshot_draw_diff": float(np.nanmax(np.abs(zb - D["zeroshot"]))),
                           "speaker_order_equal_saved": [str(s) for s in us] == [str(s) for s in D["speakers"]]}
ck["bootstrap_speaker_idx_equal_saved"] = idx_equal
ck["bootstrap_speaker_order_equal_saved"] = out["bootstrap_reimpl"]["speaker_order_equal_saved"]
ck["bootstrap_draws_equal_1e-12"] = out["bootstrap_reimpl"]["max_abs_draw_diff"] < 1e-12
ck["bootstrap_ci_equal_sidecar_1e-12"] = max(abs(a - b) for a, b in zip(ci, SCJ["gap_ci"])) < 1e-12
ck["bootstrap_2000_usable"] = int(ok.sum()) == 2000 == SCJ["usable_draws"]
# 5. pull integrity and pod check
shal = {}
for line in open(f"{POD}/logs/out_sha256.txt"):
    hh, f = line.split(); shal[f] = hh
ck["pulled_out_files_match_pod_sha256"] = all(h256(f"{POD}/out/{f}") == hh for f, hh in shal.items()) and len(shal) >= 4
V = json.load(open(PV)); ck["podverify_PASS"] = bool(V.get("PASS"))
ck["podverify_no_layer_mismatch"] = V.get("layer_mismatches") == []
ck["pod_versions_pinned"] = (J["versions"]["sklearn"], J["versions"]["numpy"], J["versions"]["scipy"]) == ("1.9.1", "2.1.2", "1.18.1")
E = json.load(open(f"{POD}/out/p26_env.json")); ck["pod_one_blas_thread"] = all(p["num_threads"] == 1 for p in E["threadpools"])
ck["pod_linux"] = J["versions"]["platform"].startswith("Linux")
ck["fold_files_equal_pod_tie_rule"] = all(J["tags"][t]["fold_file_equals_pod_tie_rule"] for t in J["tags"])
ck["sidecar_sha256_perclip"] = SCJ["sha256"]["perclip_csv"] == h256(PC)
ck["sidecar_sha256_draws"] = SCJ["sha256"]["draws_npz"] == h256(DR)
out["ALL_PASS"] = all(bool(v) for v in ck.values())
out["failed"] = [k for k, v in ck.items() if not v]
json.dump(out, open(f"{C}/verify/verify_cell.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
print(f"VERIFY {CELL} ALL_PASS={out['ALL_PASS']} failed={out['failed']} mean5 {out['mean5_rank']:.6f} zs {z_mw:.6f} "
      f"gap {out['gap_point_rank']:+.6f} ci [{ci[0]:.6f}, {ci[1]:.6f}] mac refit max AUC diff {out['mac_refit']['max_abs_auc_diff_vs_pod']:.4g} "
      f"max pred diff {out['mac_refit']['max_abs_pred_diff_vs_pod']:.4g}", flush=True)
