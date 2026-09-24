"""PART 26 independent Mac-side check of one cell (Qwen3-Omni-30B-A3B, ADReSSo). Written separately from gap26.py and
shares no code with it. Checks:
 1 pulled pod out/ files match the pod's own sha256 list;
 2 per-clip csv equals the pod OOF csv value for value (string exact), clip set / labels / speakers equal the states
   file, the zero-shot file and every fold file; fold columns equal the release fold files read here by clip;
   p_yes_zeroshot equals the zero-shot file;
 3 every AUC recomputed with scipy mannwhitneyu (not sklearn); mean of five and zero-shot AUC against the sidecar;
 4 bootstrap re-implemented: speakers sorted with Python sorted(), fresh default_rng(0), 2000 draws of
   rng.choice(n_spk, n_spk, replace=True), rank AUCs; draws and 2.5/97.5 percentiles compared with the saved npz;
 5 pinned pod versions (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1) and one BLAS thread;
 6 Mac refit on this Mac's sklearn: outer-fold predictions at the saved layers, and the inner layer choice re-derived
   for all 25 outer folds with an own greedy group fill (information only when versions differ: reported, not a gate
   unless the chosen layers differ AND the AUC moves by 0.001 or more).
usage: /usr/local/bin/python3 verify26.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import csv, json, hashlib, sys, datetime, numpy as np, multiprocessing as mp, warnings; warnings.filterwarnings("ignore")
from scipy.stats import mannwhitneyu, rankdata

R = "scores/part26/Qwen3-Omni-30B-A3B_ADReSSo"
POD = R + "/pull/p26-q3o-adresso"
SC = json.load(open(R + "/Qwen3-Omni-30B-A3B_ADReSSo_perclip.sidecar.json"))
STATES = "<local data dir>/release/overnight2/q3o/q3o_adresso_states.npz"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adresso_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
FF = {f"seed{s}": f"{FD}/adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}
FF["single"] = f"{FD}/adresso_groupkfold5_pod.csv"
out = {"date_utc": datetime.datetime.utcnow().isoformat() + "Z", "checks": {}}
def chk(name, ok, **info):
    out["checks"][name] = dict(ok=bool(ok), **info); print(f"{'PASS' if ok else 'FAIL'}  {name}  {info if info else ''}", flush=True)

# 1 pod file hashes
lines = [l.split() for l in open(POD + "/logs/out_sha256.txt") if l.strip()]
bad = [p for h, p in lines if hashlib.sha256(open(f"{POD}/out/{p}", "rb").read()).hexdigest() != h]
chk("pulled_out_files_match_pod_sha256", not bad and len(lines) >= 4, n_files=len(lines), bad=bad)

# 2 data identity
pc = list(csv.DictReader(open(R + "/Qwen3-Omni-30B-A3B_ADReSSo_perclip.csv")))
pod = list(csv.DictReader(open(POD + "/out/p26_q3o_adresso_enc_nested5_oof.csv")))
same = len(pc) == len(pod) and all(a["clip"] == b["clip"] and a["speaker"] == b["speaker"] and a["label"] == b["label"] and
        all(a[f"p_probe_seed{s}"] == b[f"p_seed{s}"] and a[f"fold_seed{s}"] == b[f"fold_seed{s}"] for s in range(5)) and
        a["p_probe_single"] == b["p_single"] for a, b in zip(pc, pod))
chk("perclip_equals_pod_oof_exact", same, n=len(pc))
z = np.load(STATES, allow_pickle=True)
sn = [str(v) for v in z["name"]]; ss = [str(v) for v in z["spk"]]; sl = [int(v) for v in z["label"]]
chk("perclip_rows_equal_states_order_labels_speakers", [r["clip"] for r in pc] == sn and [r["speaker"] for r in pc] == ss and
    [int(r["label"]) for r in pc] == sl, n=len(sn))
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
chk("zeroshot_file_same_clips_labels_speakers_pyes", set(zr) == set(sn) and len(zr) == len(sn) and
    all(zr[r["clip"]]["label"] == r["label"] and zr[r["clip"]]["speaker"] == r["speaker"] and float(zr[r["clip"]]["p_yes"]) == float(r["p_yes_zeroshot"]) for r in pc))
for t, f in FF.items():
    fr = {r["clip_id"]: r for r in csv.DictReader(open(f))}
    col = "fold_" + t if t == "single" else f"fold_{t}"
    ok = set(fr) == set(sn) and all(fr[r["clip"]]["speaker_id"] == r["speaker"] and int(fr[r["clip"]]["fold"]) == int(r[col]) for r in pc)
    chk(f"fold_column_{t}_equals_{os.path.basename(f)}", ok, sha256=hashlib.sha256(open(f, "rb").read()).hexdigest())

# 3 AUCs by Mann-Whitney
y = np.array([int(r["label"]) for r in pc]); spk = np.array([r["speaker"] for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in pc] for s in range(5)]); zsc = np.array([float(r["p_yes_zeroshot"]) for r in pc])
def mw(yy, s):
    return mannwhitneyu(s[yy == 1], s[yy == 0], alternative="two-sided").statistic / ((yy == 1).sum() * (yy == 0).sum())
per = [float(mw(y, P[s])) for s in range(5)]; m5 = float(np.mean(per)); za = float(mw(y, zsc))
chk("per_repeat_auc_mannwhitney_equals_sidecar", max(abs(a - b) for a, b in zip(per, SC["probe_per_repeat"])) < 1e-12, per_repeat=[round(a, 6) for a in per])
chk("mean5_and_zeroshot_auc_equal_sidecar", abs(m5 - SC["probe_mean5"]) < 1e-12 and abs(za - SC["zeroshot_auc"]) < 1e-12 and round(za, 4) == SC["zeroshot_auc_json"],
    mean5=round(m5, 6), zeroshot=round(za, 6), gap=round(m5 - za, 6))
pj = json.load(open(POD + "/out/p26_q3o_adresso_enc_nested5.json"))
chk("pod_json_per_repeat_equals_recomputed", max(abs(a - b) for a, b in zip(per, pj["per_repeat_auc"])) < 1e-12)

# 4 bootstrap re-implemented
spks = sorted(set(spk.tolist())); where = {s: np.flatnonzero(spk == s) for s in spks}; ns = len(spks)
def rauc(yy, s):
    r = rankdata(s); n1 = int(yy.sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
g = np.random.default_rng(0); D = np.full(2000, np.nan); M = np.full(2000, np.nan); Zb = np.full(2000, np.nan); IDX = np.zeros((2000, ns), int)
for b in range(2000):
    k = g.choice(ns, size=ns, replace=True); IDX[b] = k
    rows = np.concatenate([where[spks[i]] for i in k]); yy = y[rows]
    if yy.min() == yy.max(): continue
    M[b] = np.mean([rauc(yy, P[s][rows]) for s in range(5)]); Zb[b] = rauc(yy, zsc[rows]); D[b] = M[b] - Zb[b]
ok = ~np.isnan(D); lo, hi = np.percentile(D[ok], 2.5), np.percentile(D[ok], 97.5)
dz = np.load(R + "/draws/Qwen3-Omni-30B-A3B_ADReSSo_draws.npz", allow_pickle=True)
same_idx = np.array_equal(IDX, dz["speaker_idx"]) and [str(s) for s in dz["speakers"]] == spks
dd = float(np.nanmax(np.abs(D - dz["diff"]))); nanpat = np.array_equal(np.isnan(D), np.isnan(dz["diff"]))
chk("bootstrap_draws_reproduced", same_idx and nanpat and dd < 1e-12 and int(ok.sum()) == SC["usable_draws"] == 2000, max_abs_draw_diff=dd, usable=int(ok.sum()))
chk("bootstrap_interval_reproduced_4dp", [round(lo, 4), round(hi, 4)] == [round(v, 4) for v in SC["gap_ci"]] and abs(lo - SC["gap_ci"][0]) < 1e-12 and abs(hi - SC["gap_ci"][1]) < 1e-12,
    lo=round(float(lo), 6), hi=round(float(hi), 6))
chk("point_gap_in_draws_file", abs(float(dz["point_diff"]) - (m5 - za)) < 1e-12)

# 5 versions
v = pj["versions"]
chk("pod_versions_pinned", (v["sklearn"], v["numpy"], v["scipy"]) == ("1.9.1", "2.1.2", "1.18.1") and "Linux" in v["platform"], versions=v)
drv = open(POD + "/logs/DRIVER.log").read()
chk("pod_driver_one_blas_thread_and_upload_check", "UPLOAD CHECK OK 7 files" in drv and "OPENBLAS_NUM_THREADS=1" in open(POD + "/run26.sh").read())
pv = json.load(open(POD + "/out/p26_q3o_adresso_podverify.json"))
chk("podverify_PASS", pv.get("PASS") is True, max_pred=pv.get("max_abs_pred_diff"), max_inner=pv.get("max_abs_inner_score_diff"), mism=pv.get("layer_mismatches"))

# 6 Mac refit on this Mac's sklearn (different version from the pod)
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = z["enc"].astype(np.float32)
def fit(a, b, l):
    s = StandardScaler().fit(X[a, l]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(X[a, l]), y[a])
    return m.predict_proba(s.transform(X[b, l]))[:, 1]
def fill(groups, k):
    keys = sorted(set(groups)); c = {q: 0 for q in keys}
    for q in groups: c[q] += 1
    cnt = np.array([c[q] for q in keys]); order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); asg = {}
    for i in order:
        f = int(np.argmin(load)); asg[keys[i]] = f; load[f] += cnt[i]
    return np.array([asg[q] for q in groups])
def gids(seed):
    u = np.unique(spk); m = {s: i for i, s in enumerate(np.random.default_rng(seed).permutation(u))}; return np.array([m[s] for s in spk])
FO = {s: np.array([int(r[f"fold_seed{s}"]) for r in pc]) for s in range(5)}; G = {s: gids(s) for s in range(5)}
def job(a):
    s, k, l = a; tr = np.flatnonzero(FO[s] != k); f4 = fill(list(G[s][tr]), 4); t = 0.0
    for j in range(4):
        i1, i2 = tr[f4 != j], tr[f4 == j]; t += rauc(y[i2], fit(i1, i2, l)) if len(set(y[i2])) > 1 else 0.5
    return s, k, l, t / 4
if __name__ == "__main__":
    with mp.get_context("fork").Pool(int(os.environ.get("NP", "6"))) as pool:
        res = pool.map(job, [(s, k, l) for s in range(5) for k in range(5) for l in range(X.shape[1])], chunksize=8)
    inner = {}
    for s, k, l, t in res: inner.setdefault((s, k), np.zeros(X.shape[1]))[l] = t
    lay_mac = {s: [int(np.argmax(inner[(s, k)])) for k in range(5)] for s in range(5)}
    lay_pod = {s: pj["tags"][f"seed{s}"]["layers"] for s in range(5)}
    refit_saved = np.zeros((5, len(y))); refit_mac = np.zeros((5, len(y)))
    for s in range(5):
        for k in range(5):
            tr, te = np.flatnonzero(FO[s] != k), np.flatnonzero(FO[s] == k)
            refit_saved[s, te] = fit(tr, te, lay_pod[s][k]); refit_mac[s, te] = fit(tr, te, lay_mac[s][k])
    d_saved = float(np.max(np.abs(refit_saved - P)))
    per_mac = [float(rauc(y, refit_mac[s])) for s in range(5)]
    inner_diff = max(float(np.max(np.abs(inner[(s, k)] - np.array(pj["tags"][f"seed{s}"]["inner"][k])))) for s in range(5) for k in range(5))
    mism = [(s, k, lay_pod[s][k], lay_mac[s][k]) for s in range(5) for k in range(5) if lay_pod[s][k] != lay_mac[s][k]]
    # section 6 runs on this Mac's sklearn and BLAS, not the pinned pod versions: per the docstring it is information
    # only, and the only gate is "chosen layers differ AND the mean of five moves by 0.001 or more".
    per_saved = [float(rauc(y, refit_saved[s])) for s in range(5)]
    chk("INFO_mac_refit_at_saved_layers_matches_oof_1e-3", d_saved < 1e-3, info_only=True, max_abs_pred_diff=d_saved,
        mac_sklearn=sklearn.__version__, per_repeat_mac_refit_saved_layers=[round(a, 6) for a in per_saved],
        mean5_mac_refit_saved_layers=round(float(np.mean(per_saved)), 6))
    chk("mac_inner_layer_rederivation", (not mism) or abs(np.mean(per_mac) - m5) < 1e-3, layer_mismatches=mism,
        max_abs_inner_diff=inner_diff, mean5_mac=round(float(np.mean(per_mac)), 6), per_repeat_mac=[round(a, 6) for a in per_mac])
    # different script, different pod (PART 25 C p25C-4, sklearn GroupKFold folds, same pinned versions), read here by clip
    c25 = {r["clip"]: r for r in csv.DictReader(open("<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q3o_adresso_enc_rep5_oof.csv"))}
    c25p = np.array([[float(c25[r["clip"]][f"p_seed{s}"]) for r in pc] for s in range(5)])
    c25f = all(int(c25[r["clip"]][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"]) for r in pc for s in range(5))
    chk("part25C_independent_refit_matches_oof", c25f and float(np.max(np.abs(c25p - P))) < 1e-6 and set(c25) == set(sn),
        max_abs_oof_diff=float(np.max(np.abs(c25p - P))), fold_columns_equal=c25f)
    out["ALL_PASS_GATES"] = all(c["ok"] for c in out["checks"].values() if not c.get("info_only"))
    out["ALL_PASS"] = out["ALL_PASS_GATES"]
    out["info_failures"] = [k for k, c in out["checks"].items() if c.get("info_only") and not c["ok"]]
    json.dump(out, open(R + "/verify/verify26.json", "w"), indent=1, default=str)
    print("VERIFY26 ALL_PASS_GATES", out["ALL_PASS_GATES"], "info failures", out["info_failures"], flush=True)
