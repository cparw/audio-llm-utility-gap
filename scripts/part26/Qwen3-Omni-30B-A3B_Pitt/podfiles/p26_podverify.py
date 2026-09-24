"""PART26 copy of PART25 A p25a_podverify.py (md5 ad83f08cca336c64a9ff38e952b29c4c). Edits: TAGS from env P26_TAGS
(default the five seeds) and the fold file stem from env P26_FOLD_STEM (default "pod"). Checks unchanged.
Original docstring follows.
PART25 A pod-side independent check of p25a_nested5.py output, written separately (no shared code):
 1. fold columns in the OOF csv equal the release fold files, read again here by clip;
 2. every outer fold is refitted at the saved layer (StandardScaler + LogisticRegression(max_iter=2000,
    class_weight='balanced'), the model spec), predictions compared with the saved OOF column;
 3. the inner layer choice is re-derived for EVERY outer fold with an own greedy group fill (stable tie order) and a
    multiprocessing pool, compared with the saved layer and the saved 32 inner scores;
 4. every AUC is recomputed with the rank (Mann-Whitney) formula.
usage: p25a_podverify.py STATES.npz DATASET FOLDDIR OUTPREFIX   (reads OUTPREFIX_enc_nested5_oof.csv and .json)"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, csv, json, time, numpy as np, warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NPZ, DS, FOLDDIR, OUTP = sys.argv[1:5]
z = np.load(NPZ, allow_pickle=True)
X = z["enc"].astype(np.float32); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
rows = list(csv.DictReader(open(OUTP + "_enc_nested5_oof.csv")))
J = json.load(open(OUTP + "_enc_nested5.json"))
assert [r["clip"] for r in rows] == list(names), "OOF csv clip order differs from the states"
TAGS = os.environ.get("P26_TAGS", "seed0 seed1 seed2 seed3 seed4").split()
STEM = os.environ.get("P26_FOLD_STEM", "pod")

def rank_auc(yy, s):
    r = rankdata(s); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def zfit(Xtr, ytr, Xte):
    # the model itself is the spec (StandardScaler then LogisticRegression), so it is not re-implemented here
    s = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(Xtr), ytr)
    return m.predict_proba(s.transform(Xte))[:, 1]

def greedy(groups, k):
    keys = sorted(set(groups)); pos = {g: i for i, g in enumerate(keys)}
    cnt = np.zeros(len(keys), int)
    for g in groups: cnt[pos[g]] += 1
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); fog = {}
    for gi in order:
        f = int(np.argmin(load)); fog[keys[gi]] = f; load[f] += cnt[gi]
    return np.array([fog[g] for g in groups])

def groups_for(tag):
    if tag == "single": return spk
    rng = np.random.default_rng(int(tag[4:])); u = np.unique(spk)
    m = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([m[s] for s in spk])

def inner_job(args):
    tag, k, l = args
    g = GROUPS[tag]; fo = FOLDS[tag]; tr = np.where(fo != k)[0]
    f4 = greedy(list(g[tr]), 4); s = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        yb = y[b]
        s += rank_auc(yb, zfit(X[a, l], y[a], X[b, l])) if len(set(yb)) > 1 else 0.5
    return (tag, k, l, s / 4)

GROUPS = {t: groups_for(t) for t in TAGS}
FOLDS = {}
out = {"dataset": DS, "tags": {}}
for t in TAGS:
    fn = [f for f in os.listdir(FOLDDIR) if f.startswith(f"{DS}_groupkfold5_{STEM}") and
          ((t == "single" and f == f"{DS}_groupkfold5_pod.csv") or (t != "single" and f"_{t}" in f))]
    assert len(fn) == 1, (t, fn)
    fmap = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(FOLDDIR, fn[0])))}
    ff = np.array([fmap[n] for n in names]); fc = np.array([int(r[f"fold_{t}"]) for r in rows])
    FOLDS[t] = ff
    out["tags"][t] = {"fold_file": fn[0], "csv_folds_equal_file": bool(np.array_equal(ff, fc)),
                      "greedy_outer_equals_file": bool(np.array_equal(greedy(list(GROUPS[t]), 5), ff))}
t0 = time.time()
jobs = [(t, k, l) for t in TAGS for k in range(5) for l in range(X.shape[1])]
with mp.get_context("fork").Pool(int(os.environ.get("P25_NJ", "24"))) as pool:
    res = pool.map(inner_job, jobs, chunksize=4)
inner = {}
for t, k, l, s in res: inner.setdefault((t, k), [0.0] * X.shape[1])[l] = s
worst_pred = 0.0; worst_inner = 0.0; layer_mismatch = []; auc_rows = {}
for t in TAGS:
    saved = np.array([float(r[f"p_{t}"]) for r in rows]); refit = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(FOLDS[t] != k)[0], np.where(FOLDS[t] == k)[0]
        L = J["tags"][t]["layers"][k]
        mine = int(np.argmax(inner[(t, k)]))
        if mine != L: layer_mismatch.append({"tag": t, "fold": k, "saved": L, "rederived": mine})
        worst_inner = max(worst_inner, float(np.max(np.abs(np.array(inner[(t, k)]) - np.array(J["tags"][t]["inner"][k])))))
        refit[te] = zfit(X[tr][:, L], y[tr], X[te][:, L])
    d = float(np.max(np.abs(refit - saved))); worst_pred = max(worst_pred, d)
    a_saved, a_refit = rank_auc(y, saved), rank_auc(y, refit)
    auc_rows[t] = {"auc_json": J["tags"][t]["auc"], "auc_rank_saved_oof": a_saved, "auc_rank_refit": a_refit,
                   "abs_json_vs_rank": abs(a_saved - J["tags"][t]["auc"]), "max_abs_pred_diff_refit_vs_saved": d}
    out["tags"][t].update(auc_rows[t])
m5 = float(np.mean([auc_rows[f"seed{s}"]["auc_rank_saved_oof"] for s in range(5)]))
out.update({"mean5_rank_from_oof": m5, "mean5_json": J["mean5"], "abs_mean5_diff": abs(m5 - J["mean5"]),
            "max_abs_pred_diff": worst_pred, "max_abs_inner_score_diff": worst_inner,
            "layer_mismatches": layer_mismatch, "seconds": round(time.time() - t0, 1)})
ok = (all(v["csv_folds_equal_file"] and v["greedy_outer_equals_file"] for v in out["tags"].values())
      and worst_pred < 1e-6 and not layer_mismatch and out["abs_mean5_diff"] < 1e-9)
out["PASS"] = bool(ok)
json.dump(out, open(OUTP + "_podverify.json", "w"), indent=1)
print(f"PODVERIFY {DS} PASS={ok} max_pred_diff={worst_pred:.3g} max_inner_diff={worst_inner:.3g} "
      f"layer_mismatch={len(layer_mismatch)} mean5 {m5:.6f} vs {J['mean5']:.6f}", flush=True)
