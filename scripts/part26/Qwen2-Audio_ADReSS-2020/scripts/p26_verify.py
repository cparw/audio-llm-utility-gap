"""PART26 cell Qwen2-Audio / ADReSS-2020: independent check on the Mac, written separately from p26_gap.py and from the
pod scripts (no shared code). Checks:
 A. the cell per-clip csv carries the pod out-of-fold vectors and fold ids unchanged, and the zero-shot p_yes of the saved file;
 B. fold ids equal the release fold files read again by clip, and equal an own greedy GroupKFold fill (stable tie order)
    on speakers renumbered with default_rng(seed).permutation(unique speakers);
 C. per-repeat AUCs, their mean and the zero-shot AUC recomputed with the Mann-Whitney rank formula;
 D. the 2000-draw paired speaker bootstrap rebuilt with an own loop (default_rng(0), rng.choice over the sorted unique
    speakers) and the rank AUC; draws and percentiles compared with the saved npz;
 E. the pod input equals the saved Qwen2-Audio states (enc, label, spk, name) and its sha256 equals the one the pod read;
 F. a Mac refit of every outer fold at the saved layer and a Mac re-run of the inner layer choice (information only:
    Mac sklearn/BLAS differ from the pod, so tiny prediction differences are expected).
Writes verify/p26_verify.json.   usage: /usr/local/bin/python3 p26_verify.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
import csv, json, hashlib, time, numpy as np, multiprocessing as mp, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
C = "scores/part26/Qwen2-Audio_ADReSS-2020"
TAG = "Qwen2-Audio_ADReSS-2020"
POD = f"{C}/pull/p26-q2a-adress2020/out"
FD = "<local data dir>/release/folds"
ZS = "<local data dir>/paper1_local_runs/probe2/adress2020_zeroshot_scores.csv"
SRC = "<local data dir>/paper1_local_runs/probe2/adress2020_states.npz"
STAGE = f"{C}/stage/q2a_adress2020_encstates.npz"

def h(p):
    x = hashlib.sha256(); f = open(p, "rb")
    while True:
        b = f.read(1 << 20)
        if not b: break
        x.update(b)
    return x.hexdigest()

def mw_auc(lab, s):
    lab = np.asarray(lab); r = rankdata(s); n1 = int(lab.sum()); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

res = {"checks": {}}
pod = list(csv.DictReader(open(f"{POD}/p26_q2a_adress2020_enc_nested5_oof.csv")))
cell = list(csv.DictReader(open(f"{C}/{TAG}_perclip.csv")))
zsr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
J = json.load(open(f"{POD}/p26_q2a_adress2020_enc_nested5.json"))
S = json.load(open(f"{C}/{TAG}.sidecar.json"))
D = np.load(f"{C}/{TAG}_draws.npz", allow_pickle=True)
clips = [r["clip"] for r in pod]
# A
a_ok = [r["clip"] for r in cell] == clips and all(
    c["speaker"] == p["speaker"] and c["label"] == p["label"] and
    all(c[f"p_probe_seed{s}"] == p[f"p_seed{s}"] and c[f"fold_seed{s}"] == p[f"fold_seed{s}"] for s in range(5)) and
    c["p_probe_single_podtie"] == p["p_single"] and float(c["p_yes_zeroshot"]) == float(zsr[c["clip"]]["p_yes"]) and
    int(zsr[c["clip"]]["label"]) == int(c["label"]) for c, p in zip(cell, pod))
res["checks"]["A_perclip_equals_pod_and_zeroshot"] = bool(a_ok)
y = np.array([int(r["label"]) for r in pod]); spk = np.array([r["speaker"] for r in pod])
P = np.array([[float(r[f"p_seed{s}"]) for r in pod] for s in range(5)]); zs = np.array([float(zsr[c]["p_yes"]) for c in clips])
# B
def fill(groups, k):
    keys = sorted(set(groups)); cnt = {g: 0 for g in keys}
    for g in groups: cnt[g] += 1
    arr = np.array([cnt[g] for g in keys]); order = np.argsort(arr, kind="stable")[::-1]
    load = [0] * k; asg = {}
    for i in order:
        f = min(range(k), key=lambda j: (load[j], j)); asg[keys[i]] = f; load[f] += arr[i]
    return np.array([asg[g] for g in groups])
b = {}
for s in range(5):
    fmap = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv"))}
    ff = np.array([fmap[c][0] for c in clips]); pc = np.array([int(r[f"fold_seed{s}"]) for r in pod])
    u = np.unique(spk); perm = np.random.default_rng(s).permutation(u); ren = {sp: i for i, sp in enumerate(perm)}
    own = fill([ren[x] for x in spk], 5)
    b[f"seed{s}"] = {"pod_equals_file": bool(np.array_equal(ff, pc)), "own_fill_equals_file": bool(np.array_equal(own, ff)),
                     "speakers_equal_file": all(fmap[c][1] == x for c, x in zip(clips, spk)), "fold_file_sha256": h(f"{FD}/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")}
res["checks"]["B_folds"] = b
# C
per = [mw_auc(y, P[s]) for s in range(5)]; m5 = float(np.mean(per)); za = mw_auc(y, zs)
res["checks"]["C_auc"] = {"per_repeat_rank": per, "mean5_rank": m5, "zeroshot_rank": za,
                          "max_abs_vs_sidecar_per_repeat": float(max(abs(a - b) for a, b in zip(per, S["probe"]["per_repeat_auc"]))),
                          "abs_mean5_vs_sidecar": abs(m5 - S["probe"]["mean5"]), "abs_mean5_vs_pod_json": abs(m5 - J["mean5"]),
                          "abs_zeroshot_vs_sidecar": abs(za - S["answer"]["zeroshot_auc"]), "gap_point_rank": m5 - za,
                          "abs_gap_vs_sidecar": abs((m5 - za) - S["gap"]["point"])}
# D
us = sorted(set(spk.tolist())); where = {sp: np.where(spk == sp)[0] for sp in us}
g = np.random.default_rng(0); dd = []; mm = []; zz = []; same_idx = True
for k in range(2000):
    pick = g.choice(len(us), size=len(us), replace=True)
    if not np.array_equal(pick, D["speaker_idx"][k]): same_idx = False
    ii = np.concatenate([where[us[j]] for j in pick]); yy = y[ii]
    if yy.min() == yy.max(): dd.append(np.nan); mm.append(np.nan); zz.append(np.nan); continue
    m = np.mean([mw_auc(yy, P[s][ii]) for s in range(5)]); z = mw_auc(yy, zs[ii]); mm.append(m); zz.append(z); dd.append(m - z)
dd = np.array(dd); ok = ~np.isnan(dd)
lo, hi = np.percentile(dd[ok], 2.5), np.percentile(dd[ok], 97.5)
res["checks"]["D_bootstrap"] = {"speaker_idx_equal": bool(same_idx), "usable": int(ok.sum()),
                                "max_abs_diff_draws": float(np.nanmax(np.abs(dd - D["diff"]))), "max_abs_mean5_draws": float(np.nanmax(np.abs(np.array(mm) - D["mean5"]))),
                                "max_abs_zeroshot_draws": float(np.nanmax(np.abs(np.array(zz) - D["zeroshot"]))),
                                "ci_rebuilt": [float(lo), float(hi)], "ci_sidecar": S["gap"]["ci95"],
                                "abs_ci_diff": [abs(float(lo) - S["gap"]["ci95"][0]), abs(float(hi) - S["gap"]["ci95"][1])],
                                "speakers_in_npz_equal": list(D["speakers"].astype(str)) == us}
# E
zsrc = np.load(SRC, allow_pickle=True); zst = np.load(STAGE, allow_pickle=True)
res["checks"]["E_inputs"] = {"stage_arrays_equal_source": all(np.array_equal(zsrc[k], zst[k]) for k in ("enc", "label", "spk", "name", "p_yes")),
                             "stage_sha256": h(STAGE), "pod_read_sha256": J["states"]["sha256"], "sha_equal": h(STAGE) == J["states"]["sha256"],
                             "stage_names_equal_pod_order": list(zst["name"].astype(str)) == clips}
# F (information only)
X = zst["enc"].astype(np.float32)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn
def fp(tr, te, l):
    sc = StandardScaler().fit(X[tr, l]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
    return m.predict_proba(sc.transform(X[te, l]))[:, 1]
def inner_score(args):
    s, k, l = args
    fo = np.array([int(r[f"fold_seed{s}"]) for r in pod]); tr = np.where(fo != k)[0]
    u = np.unique(spk); perm = np.random.default_rng(s).permutation(u); ren = {sp: i for i, sp in enumerate(perm)}
    f4 = fill([ren[x] for x in spk[tr]], 4); t = 0.0
    for j in range(4):
        a_, b_ = tr[f4 != j], tr[f4 == j]
        t += mw_auc(y[b_], fp(a_, b_, l)) if len(set(y[b_])) > 1 else 0.5
    return (s, k, l, t / 4)
if __name__ == "__main__":
    t0 = time.time()
    with mp.get_context("fork").Pool(8) as pool:
        out = pool.map(inner_score, [(s, k, l) for s in range(5) for k in range(5) for l in range(X.shape[1])], chunksize=4)
    inner = {}
    for s, k, l, v in out: inner.setdefault((s, k), [0.0] * X.shape[1])[l] = v
    f = {"mac_sklearn": sklearn.__version__, "seconds": 0, "per_seed": {}}
    for s in range(5):
        fo = np.array([int(r[f"fold_seed{s}"]) for r in pod]); refit = np.zeros(len(y)); mism = []; innd = 0.0
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]; L = J["tags"][f"seed{s}"]["layers"][k]
            refit[te] = fp(tr, te, L); mine = int(np.argmax(inner[(s, k)]))
            innd = max(innd, float(np.max(np.abs(np.array(inner[(s, k)]) - np.array(J["tags"][f"seed{s}"]["inner"][k])))))
            if mine != L: mism.append({"fold": k, "pod": L, "mac": mine})
        f["per_seed"][f"seed{s}"] = {"max_abs_pred_diff_mac_refit": float(np.max(np.abs(refit - P[s]))), "auc_mac_refit": mw_auc(y, refit),
                                     "auc_pod": per[s], "inner_layer_mismatches_mac": mism, "max_abs_inner_diff_mac": innd}
    f["seconds"] = round(time.time() - t0, 1)
    res["checks"]["F_mac_refit_information_only"] = f
    cA = res["checks"]["A_perclip_equals_pod_and_zeroshot"]
    cB = all(v["pod_equals_file"] and v["own_fill_equals_file"] and v["speakers_equal_file"] for v in b.values())
    cc = res["checks"]["C_auc"]; cC = cc["max_abs_vs_sidecar_per_repeat"] < 1e-12 and cc["abs_mean5_vs_sidecar"] < 1e-12 and cc["abs_zeroshot_vs_sidecar"] < 1e-12 and cc["abs_gap_vs_sidecar"] < 1e-12
    dD = res["checks"]["D_bootstrap"]; cD = dD["speaker_idx_equal"] and dD["usable"] == 2000 and dD["max_abs_diff_draws"] < 1e-12 and max(dD["abs_ci_diff"]) < 1e-12 and dD["speakers_in_npz_equal"]
    eE = res["checks"]["E_inputs"]; cE = eE["stage_arrays_equal_source"] and eE["sha_equal"] and eE["stage_names_equal_pod_order"]
    res["summary"] = {"A": bool(cA), "B": bool(cB), "C": bool(cC), "D": bool(cD), "E": bool(cE), "podverify_PASS": S["podverify"]["PASS"]}
    res["PASS"] = bool(cA and cB and cC and cD and cE and S["podverify"]["PASS"])
    os.makedirs(f"{C}/verify", exist_ok=True)
    json.dump(res, open(f"{C}/verify/p26_verify.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["summary"]), "PASS", res["PASS"])
    print("C", {k: v for k, v in cc.items() if k != "per_repeat_rank"})
    print("D", {k: v for k, v in dD.items()})
    print("F", json.dumps(f["per_seed"]))
