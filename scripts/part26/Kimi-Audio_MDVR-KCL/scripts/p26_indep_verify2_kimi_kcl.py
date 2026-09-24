"""PART26 Kimi-Audio MDVR-KCL, second independent check (run 2, 24 Sep). Written fresh; it shares no code with
p25a_nested5.py, p25a_podverify.py, p26_gap_kimi_kcl.py or p26_verify_kimi_kcl.py. Reads only; writes one json.
A. sha256 of every input equals the sidecar.
B. states: shape, finite, 32 distinct layers, 1500 frames per clip; clip, speaker and label order equal the zero-shot file;
   the new LM states agree with the saved kimi_new LM states (same audio went in).
C. the fold columns in the pod OOF csv equal the release fold files read here straight from release/folds.
D. per-repeat AUCs and the answer AUC as exact fractions (Mann-Whitney count with Fraction).
E. inner layer choice re-derived on the Mac for every outer fold: own greedy 4-fold fill, own StandardScaler +
   LogisticRegression(max_iter=2000, class_weight='balanced') fits, inner AUC as exact fractions; the saved layer must be
   one of the layers at the exact maximum.
F. refit at the saved layers on the Mac: out-of-fold AUC equals the pod AUC exactly (as a fraction).
G. bootstrap rebuilt with an own loop: fresh default_rng(0), rng.choice over np.unique speakers, 2000 draws; equals the
   saved draws; 2.5 and 97.5 percentiles equal the sidecar.
H. per-clip csv equals the pod OOF csv and the zero-shot p_yes, clip by clip.
usage: /usr/local/bin/python3 p26_indep_verify2_kimi_kcl.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import csv, json, hashlib, datetime, platform
from fractions import Fraction
from multiprocessing import get_context
import numpy as np, sklearn, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R = "scores/part26/Kimi-Audio_MDVR-KCL"
SC = json.load(open(f"{R}/per_clip/p26_kimi_kcl_perclip.sidecar.json"))
F = SC["files"]
FOLDDIR = "<local data dir>/release/folds"
OLD = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_states.npz"
OUT = f"{R}/verify/p26_indep_verify2_kimi_kcl.json"
chk = {}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def auc_frac(y, s):
    """exact Mann-Whitney AUC: pairs pos>neg count 1, ties count 1/2"""
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    gt = sum(int((p > neg).sum()) for p in pos); eq = sum(int((p == neg).sum()) for p in pos)
    return Fraction(2 * gt + eq, 2 * len(pos) * len(neg))

# A
chk["A_sha256_equal_sidecar"] = {k: sha(F[k]) == SC["sha256"][k] for k in ("pod_oof_csv", "pod_json", "zeroshot", "states", "perclip_csv", "draws_npz")}
chk["A_sha256_equal_sidecar"]["ok"] = all(chk["A_sha256_equal_sidecar"].values())

# B
z = np.load(F["states"], allow_pickle=True)
E = z["enc"]; names = [str(n) for n in z["name"]]; spk = np.array([str(s) for s in z["spk"]]); y = z["label"].astype(int)
zs_rows = list(csv.DictReader(open(F["zeroshot"])))
lay_same = [bool(np.array_equal(E[:, a], E[:, b])) for a in range(E.shape[1]) for b in range(a + 1, E.shape[1])]
chk["B_states"] = {"shape": list(E.shape), "dtype": str(E.dtype), "finite": bool(np.isfinite(E).all()),
                   "layers_all_distinct": not any(lay_same), "enc_frames": sorted(set(int(v) for v in z["enc_frames"])),
                   "order_equals_zeroshot_file": [r["clip"] for r in zs_rows] == names
                   and [r["speaker"] for r in zs_rows] == list(spk) and [int(r["label"]) for r in zs_rows] == list(y),
                   "n": len(names), "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum())}
o = np.load(OLD, allow_pickle=True)
same_order = [str(n) for n in o["name"]] == names
if same_order:
    Lo, Ln = o["llm"].astype(np.float64), z["llm"].astype(np.float64)
    cors = [float(np.corrcoef(Lo[:, l].ravel(), Ln[:, l].ravel())[0, 1]) for l in range(Lo.shape[1])]
    chk["B_states"]["llm_vs_saved_kimi_new"] = {"same_clip_order": True, "min_layer_corr": min(cors),
                                                "max_abs_diff": float(np.abs(Lo - Ln).max())}
else:
    chk["B_states"]["llm_vs_saved_kimi_new"] = {"same_clip_order": False}
b = chk["B_states"]
b["ok"] = (b["shape"] == [37, 32, 1280] and b["finite"] and b["layers_all_distinct"] and b["enc_frames"] == [1500]
           and b["order_equals_zeroshot_file"] and b["llm_vs_saved_kimi_new"].get("min_layer_corr", 0) > 0.999)

# C
oof = list(csv.DictReader(open(F["pod_oof_csv"])))
assert [r["clip"] for r in oof] == names
TAGS = ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]
FF = {t: (f"kcl_groupkfold5_pod_{t}_UNVERIFIED.csv" if t != "single" else "kcl_groupkfold5_pod.csv") for t in TAGS}
FOLD = {}
cc = {}
for t in TAGS:
    m = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(os.path.join(FOLDDIR, FF[t])))}
    FOLD[t] = np.array([m[n][0] for n in names])
    cc[t] = {"file": FF[t], "sha_equals_sidecar": sha(os.path.join(FOLDDIR, FF[t])) == SC["probe"]["fold_file_sha256"][t],
             "folds_equal_oof_csv": bool(np.array_equal(FOLD[t], [int(r[f"fold_{t}"]) for r in oof])),
             "speakers_equal": all(m[n][1] == s for n, s in zip(names, spk)), "fold_sizes": np.bincount(FOLD[t]).tolist()}
chk["C_folds"] = cc; chk["C_folds"]["ok"] = all(v["sha_equals_sidecar"] and v["folds_equal_oof_csv"] and v["speakers_equal"] for v in cc.values())

# D
P = {t: np.array([float(r[f"p_{t}"]) for r in oof]) for t in TAGS}
zs = np.array([float(r["p_yes"]) for r in zs_rows])
per = [auc_frac(y, P[f"seed{s}"]) for s in range(5)]; m5 = sum(per) / 5; za = auc_frac(y, zs)
J = json.load(open(F["pod_json"]))
chk["D_aucs"] = {"per_repeat_exact": [str(a) for a in per], "per_repeat": [float(a) for a in per],
                 "mean5_exact": str(m5), "mean5": float(m5), "single_exact": str(auc_frac(y, P["single"])),
                 "answer_exact": str(za), "answer": float(za), "gap_exact": str(m5 - za), "gap": float(m5 - za)}
chk["D_aucs"]["ok"] = (all(abs(float(a) - b) < 1e-12 for a, b in zip(per, SC["probe"]["per_repeat_auc"]))
                       and abs(float(m5) - SC["probe"]["mean5"]) < 1e-12 and abs(float(za) - SC["answer"]["auc"]) < 1e-12
                       and abs(float(m5 - za) - SC["gap"]["point"]) < 1e-12 and abs(float(auc_frac(y, P["single"])) - J["single_split_auc"]) < 1e-12)

# E and F
def groups_of(t):
    if t == "single": return spk.copy()
    u = np.unique(spk); perm = np.random.default_rng(int(t[4:])).permutation(u)
    ren = {s: i for i, s in enumerate(perm)}
    return np.array([ren[s] for s in spk])

def fill(g, k):
    """GroupKFold greedy fill: biggest group first (ties: later sorted id first), each to the least loaded fold (lowest index)"""
    ids = sorted(set(g.tolist())); cnt = {i: int((g == i).sum()) for i in ids}
    seq = sorted(range(len(ids)), key=lambda j: (-cnt[ids[j]], -j))
    load = [0] * k; where = {}
    for j in seq:
        f = min(range(k), key=lambda q: (load[q], q)); where[ids[j]] = f; load[f] += cnt[ids[j]]
    return np.array([where[i] for i in g.tolist()])

def fit_score(Xtr, ytr, Xte):
    s = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(Xtr), ytr).predict_proba(s.transform(Xte))[:, 1]

G = {t: groups_of(t) for t in TAGS}
X = E.astype(np.float32)

def job(a):
    t, k, l = a
    tr = np.where(FOLD[t] != k)[0]; f4 = fill(G[t][tr], 4); fr = []
    for j in range(4):
        itr, ite = tr[f4 != j], tr[f4 == j]
        fr.append(auc_frac(y[ite], fit_score(X[itr, l], y[itr], X[ite, l])) if len(set(y[ite])) > 1 else Fraction(1, 2))
    return (t, k, l, fr)

if __name__ == "__main__":
    # outer fold files must also equal the greedy fill on the renumbered ids (the probe script asserts this; check it here too)
    chk["C_folds"]["outer_equals_own_greedy_fill"] = {t: bool(np.array_equal(fill(G[t], 5), FOLD[t])) for t in TAGS}
    jobs = [(t, k, l) for t in TAGS for k in range(5) for l in range(X.shape[1])]
    with get_context("fork").Pool(8) as pool:
        res = pool.map(job, jobs, chunksize=8)
    inner = {}
    for t, k, l, fr in res: inner.setdefault((t, k), {})[l] = fr
    E_rows = []; worst_float = 0.0; F_rows = {}
    for t in TAGS:
        refit = np.zeros(len(y))
        for k in range(5):
            means = {l: sum(fr) / 4 for l, fr in inner[(t, k)].items()}
            top = max(means.values()); tied = sorted(l for l, v in means.items() if v == top)
            saved = J["tags"][t]["layers"][k]
            pod_inner = J["tags"][t]["inner"][k]
            worst_float = max(worst_float, max(abs(float(means[l]) - pod_inner[l]) for l in means))
            E_rows.append({"tag": t, "fold": k, "saved_layer": saved, "exact_max": str(top), "layers_at_exact_max": tied,
                           "saved_in_max_set": saved in tied,
                           "pod_float_argmax_first_index": int(np.argmax(pod_inner)) == saved,
                           "saved_inner_fracs": [str(v) for v in inner[(t, k)][saved]]})
            tr, te = np.where(FOLD[t] != k)[0], np.where(FOLD[t] == k)[0]
            refit[te] = fit_score(X[tr][:, saved], y[tr], X[te][:, saved])
        F_rows[t] = {"auc_mac_exact": str(auc_frac(y, refit)), "auc_pod_exact": str(auc_frac(y, P[t])),
                     "equal": auc_frac(y, refit) == auc_frac(y, P[t]), "max_abs_pred_diff": float(np.abs(refit - P[t]).max())}
    ties = [r for r in E_rows if len(r["layers_at_exact_max"]) > 1]
    chk["E_inner_choice"] = {"n_outer_folds": len(E_rows), "all_saved_in_exact_max_set": all(r["saved_in_max_set"] for r in E_rows),
                             "all_saved_equal_pod_float_argmax": all(r["pod_float_argmax_first_index"] for r in E_rows),
                             "n_folds_with_exact_ties": len(ties), "max_abs_mac_vs_pod_inner_float": worst_float,
                             "tied_folds": [{k: r[k] for k in ("tag", "fold", "saved_layer", "exact_max", "layers_at_exact_max")} for r in ties],
                             "rows": E_rows}
    chk["E_inner_choice"]["ok"] = chk["E_inner_choice"]["all_saved_in_exact_max_set"] and chk["E_inner_choice"]["all_saved_equal_pod_float_argmax"]
    chk["F_refit_saved_layers"] = F_rows; chk["F_refit_saved_layers"]["ok"] = all(v["equal"] for v in F_rows.values())

    # tie sensitivity on the five repeats: swap the saved layer for each other layer at the exact max, one fold at a time
    sens = []
    for r in ties:
        t, k = r["tag"], r["fold"]
        if t == "single": continue
        tr, te = np.where(FOLD[t] != k)[0], np.where(FOLD[t] == k)[0]
        for alt in r["layers_at_exact_max"]:
            if alt == r["saved_layer"]: continue
            p = P[t].copy(); p[te] = fit_score(X[tr][:, alt], y[tr], X[te][:, alt])
            a = auc_frac(y, p); pr = list(per); pr[int(t[4:])] = a; mm = sum(pr) / 5
            sens.append({"tag": t, "fold": k, "alt_layer": alt, "repeat_auc_alt": float(a), "mean5_alt": float(mm), "gap_alt": float(mm - za)})
    chk["E_tie_sensitivity"] = sens

    # G
    u = np.unique(spk); rows = [np.where(spk == s)[0] for s in u]; rng = np.random.default_rng(0)
    D = np.load(F["draws_npz"], allow_pickle=True)
    diff = np.empty(2000); m5b = np.empty(2000); zb = np.empty(2000); idx_same = True
    Pm = np.stack([P[f"seed{s}"] for s in range(5)])
    for bb in range(2000):
        idx = rng.choice(len(u), size=len(u), replace=True)
        idx_same &= bool(np.array_equal(idx, D["speaker_idx"][bb]))
        ii = np.concatenate([rows[i] for i in idx]); yy = y[ii]
        if yy.min() == yy.max(): diff[bb] = m5b[bb] = zb[bb] = np.nan; continue
        m5b[bb] = np.mean([float(auc_frac(yy, Pm[r][ii])) for r in range(5)]); zb[bb] = float(auc_frac(yy, zs[ii])); diff[bb] = m5b[bb] - zb[bb]
    ok = ~np.isnan(diff)
    lo, hi = float(np.percentile(diff[ok], 2.5)), float(np.percentile(diff[ok], 97.5))
    chk["G_bootstrap"] = {"usable": int(ok.sum()), "speaker_idx_equal_saved": idx_same,
                          "max_abs_diff_vs_saved": float(np.nanmax(np.abs(diff - D["diff"]))),
                          "lo": lo, "hi": hi, "sidecar_lo": SC["gap"]["ci_2p5_97p5"][0], "sidecar_hi": SC["gap"]["ci_2p5_97p5"][1],
                          "probe_mean5_ci": [float(np.percentile(m5b[ok], 2.5)), float(np.percentile(m5b[ok], 97.5))],
                          "answer_ci": [float(np.percentile(zb[ok], 2.5)), float(np.percentile(zb[ok], 97.5))],
                          "saved_draw_keys": list(D.files)}
    g = chk["G_bootstrap"]
    g["ok"] = (g["usable"] == 2000 and idx_same and g["max_abs_diff_vs_saved"] < 1e-12
               and abs(lo - g["sidecar_lo"]) < 1e-12 and abs(hi - g["sidecar_hi"]) < 1e-12)

    # H
    pc = list(csv.DictReader(open(F["perclip_csv"])))
    hh = ([r["clip"] for r in pc] == names
          and all(float(pc[i][f"p_probe_seed{s}"]) == P[f"seed{s}"][i] for i in range(len(names)) for s in range(5))
          and all(int(pc[i][f"fold_seed{s}"]) == FOLD[f"seed{s}"][i] for i in range(len(names)) for s in range(5))
          and all(float(pc[i]["p_yes_zeroshot"]) == zs[i] for i in range(len(names)))
          and all(int(pc[i]["label"]) == y[i] and pc[i]["speaker"] == spk[i] for i in range(len(names))))
    chk["H_perclip_csv"] = {"ok": bool(hh), "columns": list(pc[0].keys())}

    all_ok = all(v.get("ok") for k, v in chk.items() if isinstance(v, dict) and "ok" in v)
    rec = {"what": "PART26 Kimi-Audio MDVR-KCL second independent check (fresh code, Mac)", "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
           "checks": chk, "all_pass": bool(all_ok)}
    json.dump(rec, open(OUT, "w"), indent=1, default=str)
    print("ALL_PASS", all_ok, {k: v.get("ok") for k, v in chk.items() if isinstance(v, dict)})
    print("D", chk["D_aucs"]); print("E ties", chk["E_inner_choice"]["tied_folds"], "sens", sens)
    print("G", {k: g[k] for k in ("usable", "lo", "hi", "max_abs_diff_vs_saved", "speaker_idx_equal_saved")})
