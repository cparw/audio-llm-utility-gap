"""PART26 independent check, cell Qwen3-Omni-30B-A3B x ADReSS-2020. Written separately from p26_gap.py and from the
pod scripts; shares no code with them. Reads only files on disk.
 1. per-clip csv: fold columns equal the release fold files read again from release/folds by clip; probe columns equal the
    pulled pod OOF csv byte for byte as strings; zero-shot column equals the zero-shot file.
 2. outer folds re-derived from scratch: speakers renumbered with default_rng(seed).permutation(sorted unique ids), then a
    greedy largest-first fill with a stable tie order; compared with the fold columns.
 3. every AUC recomputed by counting positive/negative pairs (ties count one half), no sklearn.
 4. the 2000 bootstrap draws rebuilt with an own loop (same generator calls), AUCs by pair counting; compared with the
    saved draws npz, and the 2.5/97.5 percentiles recomputed with numpy.quantile.
 5. comparison with the PART25 C Linux refit of the same cell (part25/C/pull/p25C-4, a different script that used this
    pod's sklearn GroupKFold): fold agreement, per-clip score agreement, per-repeat AUCs.
 6. Mac refit of every outer fold at the pod's saved layer, and the inner layer choice re-derived on the Mac (different
    platform and library versions, so reported, not required to match to 1e-6).
 7. master_lookup per-repeat and zero-shot values.
writes verify/p26_verify.json.   usage: /usr/local/bin/python3 p26_verify.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
import csv, json, hashlib, numpy as np
from multiprocessing import get_context

R = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
PC = f"{R}/per_clip/p26_q3o_adress2020_perclip.csv"
SIDE = f"{R}/per_clip/p26_q3o_adress2020_perclip.sidecar.json"
DR = f"{R}/draws/p26_q3o_adress2020_draws.npz"
POD_OOF = f"{R}/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5_oof.csv"
POD_JSON = f"{R}/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5.json"
POD_V = f"{R}/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_podverify.json"
FOLDS = "<local data dir>/release/folds"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv"
MASTER = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_nested_repeats.json"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
C_OOF = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q3o_adress2020_enc_rep5_oof.csv"
C_JSON = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q3o_adress2020_enc_rep5.json"


def pair_auc(lab, s):
    lab = np.asarray(lab); s = np.asarray(s, float)
    pos = s[lab == 1]; neg = s[lab == 0]
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


out = {"checks": {}}
ck = out["checks"]
rows = list(csv.DictReader(open(PC)))
side = json.load(open(SIDE))
clips = [r["clip"] for r in rows]; lab = np.array([int(r["label"]) for r in rows]); spk = [r["speaker"] for r in rows]
n = len(rows)

# 1. fold columns vs release fold files; probe columns vs pod csv; zero-shot column vs file
fold_ok = True; used = {}
for s in range(5):
    f = f"{FOLDS}/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv"; used[s] = sha(f)
    m = {r["clip_id"]: (r["fold"], r["speaker_id"]) for r in csv.DictReader(open(f))}
    fold_ok &= all(m[c][0] == r[f"fold_seed{s}"] and m[c][1] == r["speaker"] for c, r in zip(clips, rows)) and len(m) == n
podj = json.load(open(POD_JSON))
ck["fold_columns_equal_release_fold_files"] = bool(fold_ok)
ck["fold_file_sha256_equal_pod_used"] = all(used[s] == podj["tags"][f"seed{s}"]["fold_file_sha256"] for s in range(5))
prow = {r["clip"]: r for r in csv.DictReader(open(POD_OOF))}
ck["probe_columns_equal_pod_oof_strings"] = all(r[f"p_probe_seed{s}"] == prow[r["clip"]][f"p_seed{s}"] for r in rows for s in range(5)) and len(prow) == n
zrow = {r["clip"]: r for r in csv.DictReader(open(ZS))}
ck["zeroshot_column_equals_file"] = all(r["p_yes_zeroshot"] == zrow[r["clip"]]["p_yes"] and int(zrow[r["clip"]]["label"]) == int(r["label"]) for r in rows) and len(zrow) == n

# 2. outer folds from scratch
def greedy_fill(groups, k):
    keys = sorted(set(groups.tolist())); cnt = {g: 0 for g in keys}
    for g in groups.tolist(): cnt[g] += 1
    c = np.array([cnt[g] for g in keys]); order = np.argsort(c, kind="stable")[::-1]
    load = [0] * k; fo = {}
    for gi in order:
        f = min(range(k), key=lambda j: (load[j], j)); fo[keys[gi]] = f; load[f] += c[gi]
    return np.array([fo[g] for g in groups.tolist()])
rebuilt_ok = True
for s in range(5):
    u = np.array(sorted(set(spk))); perm = np.random.default_rng(s).permutation(u); ren = {sp: i for i, sp in enumerate(perm)}
    g = np.array([ren[sp] for sp in spk])
    rebuilt_ok &= bool(np.array_equal(greedy_fill(g, 5), np.array([int(r[f"fold_seed{s}"]) for r in rows])))
ck["outer_folds_rebuilt_from_scratch_equal"] = bool(rebuilt_ok)

# 3. AUCs by pair counting
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)])
zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
per = [pair_auc(lab, P[s]) for s in range(5)]; m5 = float(np.mean(per)); za = pair_auc(lab, zs)
out["pair_count"] = {"per_repeat": per, "mean5": m5, "zeroshot": za, "gap": m5 - za}
ck["per_repeat_auc_equal_sidecar_1e-12"] = max(abs(a - b) for a, b in zip(per, side["probe"]["per_repeat_auc"])) < 1e-12
ck["zeroshot_auc_equal_sidecar_1e-12"] = abs(za - side["answer"]["auc"]) < 1e-12
ck["gap_point_equal_sidecar_1e-12"] = abs((m5 - za) - side["gap"]["point"]) < 1e-12
ck["per_repeat_auc_equal_pod_json_1e-12"] = max(abs(a - b) for a, b in zip(per, podj["per_repeat_auc"])) < 1e-12

# 4. bootstrap rebuilt
u = sorted(set(spk)); where = {sp: [i for i, x in enumerate(spk) if x == sp] for sp in u}
g = np.random.default_rng(0); D = np.full(2000, np.nan); M5 = np.full(2000, np.nan); ZB = np.full(2000, np.nan)
for b in range(2000):
    pick = g.choice(len(u), size=len(u), replace=True)
    ii = [i for k in pick for i in where[u[k]]]; yy = lab[ii]
    if yy.min() == yy.max(): continue
    M5[b] = np.mean([pair_auc(yy, P[s][ii]) for s in range(5)]); ZB[b] = pair_auc(yy, zs[ii]); D[b] = M5[b] - ZB[b]
dz = np.load(DR, allow_pickle=True); ok = ~np.isnan(D)
out["bootstrap_rebuilt"] = {"usable": int(ok.sum()), "ci": [float(np.quantile(D[ok], 0.025)), float(np.quantile(D[ok], 0.975))],
                            "max_abs_diff_vs_saved_draws": float(np.nanmax(np.abs(D - dz["diff"]))),
                            "nan_pattern_equal": bool(np.array_equal(np.isnan(D), np.isnan(dz["diff"])))}
ck["bootstrap_draws_reproduced_1e-12"] = out["bootstrap_rebuilt"]["max_abs_diff_vs_saved_draws"] < 1e-12 and out["bootstrap_rebuilt"]["nan_pattern_equal"]
ck["bootstrap_ci_reproduced_1e-12"] = (abs(out["bootstrap_rebuilt"]["ci"][0] - side["gap"]["ci_2_5"]) < 1e-12 and
                                       abs(out["bootstrap_rebuilt"]["ci"][1] - side["gap"]["ci_97_5"]) < 1e-12)
ck["bootstrap_2000_usable"] = int(ok.sum()) == 2000

# 5. PART25 C Linux refit of the same cell
crow = {r["clip"]: r for r in csv.DictReader(open(C_OOF))}; cj = json.load(open(C_JSON))
cf_eq = all(int(crow[c][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"]) for c, r in zip(clips, rows) for s in range(5))
CP = np.array([[float(crow[c][f"p_seed{s}"]) for c in clips] for s in range(5)])
cper = [pair_auc(lab, CP[s]) for s in range(5)]
out["part25C_refit"] = {"folds_equal": bool(cf_eq), "max_abs_pred_diff": float(np.max(np.abs(CP - P))),
                        "per_repeat_C": cper, "per_repeat_abs_diff": [abs(a - b) for a, b in zip(cper, per)],
                        "layers_C": cj["chosen_layers"], "layers_pod": side["probe"]["layers"],
                        "layers_equal": cj["chosen_layers"] == side["probe"]["layers"], "C_versions": cj["versions"], "C_cpu": cj.get("cpu_model")}
if cf_eq:  # a required check only when the two runs used the same outer folds; otherwise reported
    ck["part25C_independent_refit_agrees_1e-6"] = bool(out["part25C_refit"]["max_abs_pred_diff"] < 1e-6 and out["part25C_refit"]["layers_equal"])

# 7. master values
T = json.load(open(MASTER))["enc"]
out["master"] = {"per_repeat": T["per_repeat"], "mean": T["mean"], "per_repeat_equal_4dp": [round(a, 4) for a in per] == T["per_repeat"],
                 "max_abs_diff_per_repeat": max(abs(a - b) for a, b in zip(per, T["per_repeat"])), "mean5_diff": m5 - T["mean"]}  # reported
ck["zeroshot_matches_master_4dp"] = round(za, 4) == 0.7097
V = json.load(open(POD_V)); ck["podverify_PASS"] = bool(V.get("PASS"))
ck["states_sha256_mac_equals_pod"] = sha(STATES) == podj["states"]["sha256"]
ck["pod_versions_pinned"] = (podj["versions"]["sklearn"], podj["versions"]["numpy"], podj["versions"]["scipy"]) == ("1.9.1", "2.1.2", "1.18.1")

# 6. Mac refit and inner re-derivation (informative)
z = np.load(STATES, allow_pickle=True); X = z["enc"].astype(np.float32)
assert list(z["name"].astype(str)) == clips


def fitp(a, b, c):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(a)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(a), b).predict_proba(sc.transform(c))[:, 1]


def inner_job(a):
    s, k, l = a
    u2 = np.array(sorted(set(spk))); ren = {sp: i for i, sp in enumerate(np.random.default_rng(s).permutation(u2))}
    gg = np.array([ren[sp] for sp in spk]); fo = np.array([int(r[f"fold_seed{s}"]) for r in rows])
    tr = np.where(fo != k)[0]; f4 = greedy_fill(gg[tr], 4); t = 0.0
    for j in range(4):
        a1, b1 = tr[f4 != j], tr[f4 == j]
        t += pair_auc(lab[b1], fitp(X[a1, l], lab[a1], X[b1, l])) if len(set(lab[b1])) > 1 else 0.5
    return (s, k, l, t / 4)


if __name__ == "__main__":
    import sklearn
    # the pool is forked before the parent touches BLAS (fork safety on macOS); the parent refit runs after it
    with get_context("fork").Pool(max(2, (os.cpu_count() or 4) - 1)) as pool:
        res = pool.map(inner_job, [(s, k, l) for s in range(5) for k in range(5) for l in range(X.shape[1])], chunksize=8)
    worst = 0.0; refit_auc = []
    for s in range(5):
        fo = np.array([int(r[f"fold_seed{s}"]) for r in rows]); pr = np.zeros(n)
        for k in range(5):
            L = side["probe"]["layers"][f"seed{s}"][k]; tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            pr[te] = fitp(X[tr, L], lab[tr], X[te, L])
        worst = max(worst, float(np.max(np.abs(pr - P[s])))); refit_auc.append(pair_auc(lab, pr))
    sc = {}
    for s, k, l, v in res: sc.setdefault((s, k), [0.0] * X.shape[1])[l] = v
    mism = [{"seed": s, "fold": k, "pod": side["probe"]["layers"][f"seed{s}"][k], "mac": int(np.argmax(sc[(s, k)]))}
            for s in range(5) for k in range(5) if int(np.argmax(sc[(s, k)])) != side["probe"]["layers"][f"seed{s}"][k]]
    innerj = podj["tags"]
    out["mac_refit"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "max_abs_pred_diff_vs_pod": worst,
                        "per_repeat_auc_mac_refit": refit_auc, "mean5_mac_refit": float(np.mean(refit_auc)),
                        "inner_layer_mismatches": mism,
                        "max_abs_inner_score_diff_vs_pod": float(max(np.max(np.abs(np.array(sc[(s, k)]) - np.array(innerj[f"seed{s}"]["inner"][k])))
                                                                     for s in range(5) for k in range(5)))}
    out["PASS"] = all(bool(v) for v in ck.values())
    json.dump(out, open(f"{R}/verify/p26_verify.json", "w"), indent=1)
    for k2, v in ck.items(): print(f"{'OK  ' if v else 'FAIL'} {k2}")
    print("pair-count", {k2: (round(v, 6) if isinstance(v, float) else [round(a, 6) for a in v]) for k2, v in out["pair_count"].items()})
    print("bootstrap rebuilt", out["bootstrap_rebuilt"])
    print("part25C", {k2: out["part25C_refit"][k2] for k2 in ("folds_equal", "max_abs_pred_diff", "per_repeat_abs_diff", "layers_equal")})
    print("mac refit", {k2: out["mac_refit"][k2] for k2 in ("max_abs_pred_diff_vs_pod", "mean5_mac_refit", "max_abs_inner_score_diff_vs_pod")},
          "layer mismatches", len(mism))
    print("VERIFY PASS" if out["PASS"] else "VERIFY FAIL")
