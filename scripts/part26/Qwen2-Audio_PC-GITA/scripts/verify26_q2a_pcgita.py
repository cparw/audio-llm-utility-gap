"""PART26 Qwen2-Audio PC-GITA independent check, written separately from gap26_q2a_pcgita.py (no shared code), on the Mac.
Reads the ORIGINAL inputs (probe2 states file, release fold files, probe2 zero-shot file), not the staged copies.
 1. per-clip csv: clip order, labels, speakers, p_yes and the five fold columns equal the original inputs;
 2. every AUC recomputed with the rank (Mann-Whitney) formula, compared with the sidecar;
 3. the 2000-draw speaker bootstrap rebuilt with its own speaker grouping (fold file speaker_id, python sorted order)
    and rank AUCs, compared draw by draw with the saved npz and on the 2.5/97.5 percentiles;
 4. Mac refit on this Mac's sklearn: for every repeat and outer fold, all 33 layers rescored by inner GroupKFold(4)
    (own greedy fill, largest group first, stable ties, least-loaded fold), argmax compared with the pod layer, and
    the outer refit at the pod layer compared with the pod out-of-fold scores (a different sklearn and BLAS, so a
    small tolerance is expected, not bit equality).
usage: verify26_q2a_pcgita.py [NPROC]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import sys, csv, json, time, platform, numpy as np, multiprocessing as mp, sklearn, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
C = "scores/part26/Qwen2-Audio_PC-GITA"
ST = "<local data dir>/paper1_local_runs/probe2/pcgita_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/pcgita_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
S = json.load(open(f"{C}/Qwen2-Audio_PC-GITA_perclip.sidecar.json"))
D = np.load(f"{C}/draws/Qwen2-Audio_PC-GITA_draws.npz", allow_pickle=True)
rows = list(csv.DictReader(open(f"{C}/Qwen2-Audio_PC-GITA_perclip.csv")))
z = np.load(ST, allow_pickle=True); X = None
names = [str(s) for s in z["name"]]; lab = [int(v) for v in z["label"]]; sspk = [str(s) for s in z["spk"]]; spy = [float(v) for v in z["p_yes"]]
out = {"checks": {}}; ck = out["checks"]
def rank_auc(yy, s):
    yy = np.asarray(yy); r = rankdata(s); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
ck["clip_order_equals_states"] = [r["clip"] for r in rows] == names
ck["labels_equal_states"] = [int(r["label"]) for r in rows] == lab
ck["speakers_equal_states"] = [r["speaker"] for r in rows] == sspk
zm = {r["clip"]: r for r in csv.DictReader(open(ZS))}
ck["p_yes_equal_zeroshot_file"] = all(float(r["p_yes_zeroshot"]) == float(zm[r["clip"]]["p_yes"]) for r in rows)
ck["p_yes_equal_states"] = all(float(r["p_yes_zeroshot"]) == v for r, v in zip(rows, spy))
ck["zeroshot_labels_equal"] = all(int(zm[r["clip"]]["label"]) == int(r["label"]) for r in rows)
fspk = {}
for s in range(5):
    fm = {r["clip_id"]: r for r in csv.DictReader(open(f"{FD}/pcgita_groupkfold5_pod_seed{s}.csv"))}
    ck[f"fold_seed{s}_equals_file"] = all(int(r[f"fold_seed{s}"]) == int(fm[r["clip"]]["fold"]) for r in rows)
    ck[f"fold_seed{s}_speaker_equals_file"] = all(r["speaker"] == fm[r["clip"]]["speaker_id"] for r in rows)
    if s == 0: fspk = {c: fm[c]["speaker_id"] for c in fm}
y = np.array([int(r["label"]) for r in rows]); P = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)])
zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
per = [rank_auc(y, P[s]) for s in range(5)]; m5 = float(np.mean(per)); za = rank_auc(y, zs)
out["rank_auc"] = {"per_repeat": per, "mean5": m5, "zeroshot": za, "gap": m5 - za}
ck["per_repeat_auc_match_sidecar_1e-12"] = bool(np.max(np.abs(np.array(per) - np.array(S["probe_per_repeat"]))) < 1e-12)
ck["zeroshot_auc_match_sidecar_1e-12"] = abs(za - S["zeroshot_auc"]) < 1e-12
ck["zeroshot_auc_4dp_is_0.5369"] = round(za, 4) == 0.5369
# 3. bootstrap, own grouping from the fold file speaker ids
spk_f = [fspk[r["clip"]] for r in rows]; uniq = sorted(set(spk_f)); pos = {u: i for i, u in enumerate(uniq)}
members = [[] for _ in uniq]
for i, s in enumerate(spk_f): members[pos[s]].append(i)
rng = np.random.default_rng(0); nsp = len(uniq); dd = np.empty(2000); mm = np.empty(2000); zz = np.empty(2000); bad = 0
for b in range(2000):
    pick = rng.choice(nsp, size=nsp, replace=True)
    ii = np.array([j for k in pick for j in members[k]]); yy = y[ii]
    if yy.min() == yy.max(): dd[b] = mm[b] = zz[b] = np.nan; bad += 1; continue
    mm[b] = np.mean([rank_auc(yy, P[s][ii]) for s in range(5)]); zz[b] = rank_auc(yy, zs[ii]); dd[b] = mm[b] - zz[b]
ok = ~np.isnan(dd)
lo, hi = np.percentile(dd[ok], 2.5), np.percentile(dd[ok], 97.5)
out["bootstrap_rebuilt"] = {"n_speakers": nsp, "usable": int(ok.sum()), "gap_ci": [float(lo), float(hi)],
                            "max_abs_diff_vs_saved_draws": float(np.nanmax(np.abs(dd - D["diff"]))),
                            "max_abs_mean5_vs_saved": float(np.nanmax(np.abs(mm - D["mean5"]))),
                            "max_abs_zeroshot_vs_saved": float(np.nanmax(np.abs(zz - D["zeroshot"])))}
ck["bootstrap_speakers_100"] = nsp == 100
ck["bootstrap_draws_match_1e-9"] = out["bootstrap_rebuilt"]["max_abs_diff_vs_saved_draws"] < 1e-9
ck["bootstrap_ci_match_4dp"] = [round(lo, 4), round(hi, 4)] == [round(S["gap_ci"][0], 4), round(S["gap_ci"][1], 4)]
ck["speaker_idx_match_saved"] = bool(np.array_equal(np.array(uniq), D["speakers"].astype(str)))
# 4. Mac refit
def fp(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
def fill(g, k):
    keys = sorted(set(g)); cnt = {q: 0 for q in keys}
    for q in g: cnt[q] += 1
    order = sorted(range(len(keys)), key=lambda i: cnt[keys[i]])  # stable ascending, then reversed = largest first, later keys first on ties
    load = [0] * k; asg = {}
    for i in reversed(order):
        f = int(np.argmin(load)); asg[keys[i]] = f; load[f] += cnt[keys[i]]
    return np.array([asg[q] for q in g])
def job(a):
    s, k, l = a
    fo = FO[s]; tr = np.where(fo != k)[0]; g = G[s][tr]; f4 = fill(list(g), 4); t = 0.0
    for j in range(4):
        itr, ite = tr[f4 != j], tr[f4 == j]
        t += rank_auc(y[ite], fp(X[itr, l], y[itr], X[ite, l])) if len(set(y[ite])) > 1 else 0.5
    return s, k, l, t / 4
X = z["enc"].astype(np.float32)
FO = {s: np.array([int(r[f"fold_seed{s}"]) for r in rows]) for s in range(5)}
G = {}
for s in range(5):
    r_ = np.random.default_rng(s); u = np.unique(np.array(sspk)); m = {q: i for i, q in enumerate(r_.permutation(u))}
    G[s] = np.array([m[q] for q in sspk])
if __name__ == "__main__":
    t0 = time.time(); NP = int(sys.argv[1]) if len(sys.argv) > 1 else max(2, (os.cpu_count() or 4) - 1)
    jobs = [(s, k, l) for s in range(5) for k in range(5) for l in range(X.shape[1])]
    with mp.get_context("fork").Pool(NP) as pool: res = pool.map(job, jobs, chunksize=3)
    inner = {}
    for s, k, l, v in res: inner.setdefault((s, k), np.zeros(X.shape[1]))[l] = v
    PJ = json.load(open(S["files"]["pod_json"]))
    mism = []; maxd = 0.0; maxinner = 0.0; refit_auc = []
    for s in range(5):
        pl = PJ["tags"][f"seed{s}"]["layers"]; rp = np.zeros(len(y))
        for k in range(5):
            mine = int(np.argmax(inner[(s, k)]))
            if mine != pl[k]: mism.append({"seed": s, "fold": k, "pod": pl[k], "mac": mine,
                                           "pod_layer_mac_score": float(inner[(s, k)][pl[k]]), "mac_best_score": float(inner[(s, k)][mine])})
            maxinner = max(maxinner, float(np.max(np.abs(inner[(s, k)] - np.array(PJ["tags"][f"seed{s}"]["inner"][k])))))
            tr, te = np.where(FO[s] != k)[0], np.where(FO[s] == k)[0]
            rp[te] = fp(X[tr][:, pl[k]], y[tr], X[te][:, pl[k]])
        maxd = max(maxd, float(np.max(np.abs(rp - P[s])))); refit_auc.append(rank_auc(y, rp))
    out["mac_refit"] = {"layer_mismatches": mism, "max_abs_inner_score_diff_vs_pod": maxinner, "max_abs_pred_diff_vs_pod": maxd,
                        "per_repeat_auc_mac_refit": refit_auc, "mean5_mac_refit": float(np.mean(refit_auc)),
                        "abs_mean5_diff_vs_pod": abs(float(np.mean(refit_auc)) - m5), "seconds": round(time.time() - t0, 1), "nproc": NP}
    ck["mac_refit_layers_match"] = not mism
    ck["mac_refit_pred_within_1e-4"] = maxd < 1e-4
    ck["mac_refit_mean5_within_5e-4"] = out["mac_refit"]["abs_mean5_diff_vs_pod"] < 5e-4
    out["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}
    out["PASS"] = all(bool(v) for v in ck.values())
    json.dump(out, open(f"{C}/verify/verify26_q2a_pcgita.json", "w"), indent=1)
    print(json.dumps(ck, indent=1)); print("rank", out["rank_auc"]); print("boot", out["bootstrap_rebuilt"]); print("mac", {k: v for k, v in out["mac_refit"].items()})
    print("VERIFY PASS" if out["PASS"] else "VERIFY FAIL")
