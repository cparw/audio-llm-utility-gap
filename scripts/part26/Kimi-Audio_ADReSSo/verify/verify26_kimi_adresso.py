"""PART 26 independent check of the Kimi-Audio ADReSSo cell. Written separately from gap26_kimi_adresso.py and p25a_nested5.py.
1. Pulled files: local sha256 equals the pod's own out_sha256.txt; per-clip csv equals the pod OOF csv value by value.
2. AUCs recomputed with a rank-based Mann-Whitney formula (scipy.stats.rankdata), not sklearn.
3. Nested probe rerun on the Mac from the ORIGINAL states file (the pulled PART 26 GPU states npz, key enc, not the slim copy the pod read), with an own
   greedy GroupKFold (sorted by (-count, -group index), fill the lightest fold) for the inner folds and the saved fold
   files for the outer folds. Compares chosen layers and per-repeat AUCs with the pod run (pod and Mac numeric
   libraries differ, so small score differences are expected; the folds README gives the scale).
4. Bootstrap rebuilt from the per-clip csv and the zero-shot file with the same rng stream and the own AUC; compares
   every draw with the saved npz and recomputes the interval.
Writes verify/verify26_kimi_adresso.json.   usage: /usr/local/bin/python3 verify26_kimi_adresso.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
import csv, json, hashlib, time, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

C = "scores/part26/Kimi-Audio_ADReSSo"
POD = f"{C}/pull/p26-kimi-adresso-cpu2"
STATES = f"{C}/pull/p26-kimi-adresso-gpu/out/kimi_adresso_enc_states.npz"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv"
FOLDS = "<local data dir>/release/folds"
out = {}

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def auc(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# 1. pulled file integrity
remote = {}
for line in open(f"{POD}/logs/out_sha256.txt"):
    h, f = line.split(); remote[f] = h
out["pulled_sha_match"] = {f: sha(f"{POD}/out/{f}") == h for f, h in remote.items()}
pod_rows = {r["clip"]: r for r in csv.DictReader(open(f"{POD}/out/p26_kimi_adresso_enc_nested5_oof.csv"))}
pc_rows = list(csv.DictReader(open(f"{C}/per_clip/Kimi-Audio_ADReSSo_perclip.csv")))
out["perclip_equals_pod_oof"] = all(
    all(float(r[f"p_probe_seed{s}"]) == float(pod_rows[r["clip"]][f"p_seed{s}"]) and r[f"fold_seed{s}"] == pod_rows[r["clip"]][f"fold_seed{s}"]
        for s in range(5)) and float(r["p_probe_single"]) == float(pod_rows[r["clip"]]["p_single"]) for r in pc_rows) and len(pc_rows) == len(pod_rows) == 237

# 2. own AUCs
names = [r["clip"] for r in pc_rows]; y = np.array([int(r["label"]) for r in pc_rows]); spk = np.array([r["speaker"] for r in pc_rows])
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in pc_rows] for s in range(5)])
zmap = {r["clip"]: r for r in csv.DictReader(open(ZS))}
zs = np.array([float(zmap[c]["p_yes"]) for c in names])
out["zs_file_matches_perclip"] = all(float(zmap[c]["p_yes"]) == float(r["p_yes_zeroshot"]) and int(zmap[c]["label"]) == int(r["label"]) for c, r in zip(names, pc_rows))
per_own = [auc(y, P[s]) for s in range(5)]; zs_own = auc(y, zs)
side = json.load(open(f"{C}/per_clip/Kimi-Audio_ADReSSo_perclip.sidecar.json"))
out["own_auc"] = {"per_repeat": per_own, "mean5": float(np.mean(per_own)), "zeroshot": zs_own, "gap": float(np.mean(per_own)) - zs_own,
                  "max_abs_vs_sidecar_per_repeat": float(np.max(np.abs(np.array(per_own) - np.array(side["probe"]["per_repeat_auc"])))),
                  "abs_vs_sidecar_zeroshot": abs(zs_own - side["zeroshot"]["auc"])}

# 3. nested probe rerun on the Mac from the original states file
z = np.load(STATES, allow_pickle=True)
Xs = z["enc"].astype(np.float32); ys = z["label"].astype(int); ss = z["spk"].astype(str); ns = z["name"].astype(str)
order = np.array([names.index(n) for n in ns])  # states row -> per-clip row
assert np.array_equal(y[order], ys) and np.array_equal(spk[order], ss)

def greedy_folds(groups, k):
    ug, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    seq = sorted(range(len(ug)), key=lambda i: (-cnt[i], -i)); load = [0] * k; fg = [0] * len(ug)
    for gi in seq:
        f = min(range(k), key=lambda j: (load[j], j)); fg[gi] = f; load[f] += cnt[gi]
    return np.array(fg)[inv]

def fp(Xtr, ytr, Xte):
    m = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(m.transform(Xtr), ytr).predict_proba(m.transform(Xte))[:, 1]

def inner_score(tr, l, g):
    f4 = greedy_folds(g[tr], 4); tot = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        tot += auc(ys[b], fp(Xs[a, l], ys[a], Xs[b, l])) if len(set(ys[b])) > 1 else 0.5
    return tot / 4

pod_json = json.load(open(f"{POD}/out/p26_kimi_adresso_enc_nested5.json"))
t0 = time.time(); mac = {}
for s in range(5):
    fr = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{FOLDS}/adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv"))}
    fold = np.array([fr[n] for n in ns])
    rng = np.random.default_rng(s); perm = {u: i for i, u in enumerate(rng.permutation(np.unique(ss)))}; g = np.array([perm[u] for u in ss])
    assert np.array_equal(fold, greedy_folds(g, 5)), f"seed {s}: own greedy rule differs from the saved outer folds"
    oof = np.zeros(len(ys)); layers = []
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = Parallel(n_jobs=8)(delayed(inner_score)(tr, l, g) for l in range(Xs.shape[1]))
        b = int(np.argmax(sc)); layers.append(b); oof[te] = fp(Xs[tr][:, b], ys[tr], Xs[te][:, b])
    pod_p = P[s][order]
    mac[f"seed{s}"] = {"auc_mac": auc(ys, oof), "auc_pod": per_own[s], "layers_mac": layers,
                       "layers_pod": pod_json["tags"][f"seed{s}"]["layers"], "layers_equal": layers == pod_json["tags"][f"seed{s}"]["layers"],
                       "max_abs_p_diff_mac_vs_pod": float(np.max(np.abs(oof - pod_p))),
                       "rank_corr_mac_vs_pod": float(np.corrcoef(rankdata(oof), rankdata(pod_p))[0, 1])}
    print(f"seed{s} mac {mac[f'seed{s}']['auc_mac']:.4f} pod {per_own[s]:.4f} layers mac {layers} pod {mac[f'seed{s}']['layers_pod']} "
          f"max|dp| {mac[f'seed{s}']['max_abs_p_diff_mac_vs_pod']:.2e}  {time.time()-t0:.0f}s", flush=True)
out["mac_rerun"] = mac
out["mac_rerun_mean5"] = float(np.mean([mac[f"seed{s}"]["auc_mac"] for s in range(5)]))
out["mac_rerun_layers_all_equal"] = all(mac[f"seed{s}"]["layers_equal"] for s in range(5))

# 4. bootstrap rebuilt
D = np.load(f"{C}/draws/Kimi-Audio_ADReSSo_draws.npz", allow_pickle=True)
u = sorted(set(spk.tolist())); where = {s: [i for i in range(len(spk)) if spk[i] == s] for s in u}
rng = np.random.default_rng(0); diffs = []; idx_equal = True
for b in range(2000):
    idx = rng.choice(len(u), size=len(u), replace=True)
    idx_equal &= bool(np.array_equal(idx, D["speaker_idx"][b]))
    ii = [j for i in idx for j in where[u[i]]]; yy = y[ii]
    if yy.min() == yy.max(): diffs.append(np.nan); continue
    diffs.append(np.mean([auc(yy, P[r][ii]) for r in range(5)]) - auc(yy, zs[ii]))
diffs = np.array(diffs); ok = ~np.isnan(diffs)
out["bootstrap_rebuilt"] = {"speaker_idx_equal": idx_equal, "usable": int(ok.sum()),
                            "nan_pattern_equal": bool(np.array_equal(np.isnan(diffs), np.isnan(D["diff"]))),
                            "max_abs_draw_diff": float(np.nanmax(np.abs(diffs - D["diff"]))),
                            "ci": [float(np.percentile(diffs[ok], 2.5)), float(np.percentile(diffs[ok], 97.5))],
                            "ci_sidecar": side["gap"]["ci_2p5_97p5"], "point": float(np.mean(per_own)) - zs_own, "point_sidecar": side["gap"]["point"]}
out["ci_matches_sidecar_1e-9"] = bool(np.max(np.abs(np.array(out["bootstrap_rebuilt"]["ci"]) - np.array(side["gap"]["ci_2p5_97p5"]))) < 1e-9)
out["sources"] = {"states": STATES, "states_sha256": sha(STATES), "zeroshot": ZS, "zeroshot_sha256": sha(ZS)}
out["PASS"] = bool(all(out["pulled_sha_match"].values()) and out["perclip_equals_pod_oof"] and out["zs_file_matches_perclip"]
                   and out["own_auc"]["max_abs_vs_sidecar_per_repeat"] < 1e-12 and out["own_auc"]["abs_vs_sidecar_zeroshot"] < 1e-12
                   and out["bootstrap_rebuilt"]["speaker_idx_equal"] and out["bootstrap_rebuilt"]["nan_pattern_equal"]
                   and out["bootstrap_rebuilt"]["max_abs_draw_diff"] < 1e-12 and out["ci_matches_sidecar_1e-9"]
                   and abs(out["mac_rerun_mean5"] - side["probe"]["mean5"]) < 0.01)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(f"{C}/verify/verify26_kimi_adresso.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("pulled_sha_match", "perclip_equals_pod_oof", "zs_file_matches_perclip", "own_auc", "mac_rerun_mean5",
                                       "mac_rerun_layers_all_equal", "bootstrap_rebuilt", "ci_matches_sidecar_1e-9", "PASS")}, indent=1))
