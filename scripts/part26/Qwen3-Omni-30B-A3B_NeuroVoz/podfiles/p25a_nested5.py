"""PART25 A: Qwen2.5-Omni ENCODER nested probe, the Table 1 estimator, rerun with the saved fold files and with the
per-repeat out-of-fold score vectors SAVED this time.

Estimator (release/scripts/nested_repeats_all.py, the script behind every omni_<ds>_nested_repeats.json):
  for seed in 0..4: speakers renumbered with numpy.random.default_rng(seed).permutation(numpy.unique(spk));
  outer GroupKFold(5) on the renumbered ids; inside each outer training set an inner GroupKFold(4) on the same ids
  scores every encoder layer by mean inner AUC (sklearn roc_auc_score); argmax layer; refit StandardScaler +
  LogisticRegression(max_iter=2000, class_weight='balanced') on the outer training rows at that layer; score the
  held-out rows. Table 1 value = mean over the five seeds of the out-of-fold AUC.
Tag "single" (no renumbering, groups = speaker strings) is the single split behind omni_<ds>_enc_nested_oof.csv
(release/scripts/nested_probe_local.py), run as a direct clip-by-clip comparison with the saved original.

Folds: the OUTER folds are read from the saved release fold files (<ds>_groupkfold5_pod[_seedS][_UNVERIFIED].csv,
matched by clip). The script also rebuilds them with the pod tie rule (numpy.argsort(counts, kind='stable') reversed,
the GroupKFold greedy fill) and with this pod's own sklearn GroupKFold, and records both agreements. It stops if the
saved file and the pod tie rule disagree. INNER folds use the pod tie rule; agreement with this pod's sklearn
GroupKFold(4) is recorded for every outer fold.
Checkpoint after every outer fold.
usage: p25a_nested5.py STATES.npz DATASET FOLDDIR OUTPREFIX"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, json, time, csv, hashlib, platform, numpy as np, warnings; warnings.filterwarnings("ignore")
import sklearn, scipy, joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

NPZ, DS, FOLDDIR, OUTP = sys.argv[1:5]
KEY = "enc"
z = np.load(NPZ, allow_pickle=True)
y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
X = z[KEY].astype(np.float32); nl = X.shape[1]
NJ = int(os.environ.get("P25_NJ", "24"))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def stable_gkf(groups, k):
    uniq, inv = np.unique(groups, return_inverse=True); counts = np.bincount(inv)
    order = np.argsort(counts, kind="stable")[::-1]; fg = np.zeros(len(uniq), int); load = np.zeros(k)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += counts[gi]
    return fg[inv]

def sk_gkf(groups, k):
    fold = np.full(len(groups), -1)
    for j, (_, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(groups)), None, groups=groups)): fold[te] = j
    return fold

def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

def inner(tr, l, g):
    s = 0.0; f4 = stable_gkf(g[tr], 4)
    for j in range(4):
        itr, ite = np.where(f4 != j)[0], np.where(f4 == j)[0]
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4

def fold_file(tag):
    cands = ([f"{DS}_groupkfold5_pod.csv"] if tag == "single" else
             [f"{DS}_groupkfold5_pod_{tag}.csv", f"{DS}_groupkfold5_pod_{tag}_UNVERIFIED.csv"])
    for c in cands:
        p = os.path.join(FOLDDIR, c)
        if os.path.exists(p): return p
    raise SystemExit(f"no fold file for {DS} {tag}")

CK = OUTP + "_ckpt.json"
ck = json.load(open(CK)) if os.path.exists(CK) else {}
t0 = time.time()
TAGS = ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]
for tag in TAGS:
    if tag == "single":
        g = spk
    else:
        seed = int(tag[4:]); rng = np.random.default_rng(seed); u = np.unique(spk)
        perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    ff = fold_file(tag)
    fr = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(ff))}
    assert len(fr) == len(names) and set(fr) == set(names), f"{tag}: fold file clips differ from states clips"
    assert all(fr[n][1] == s for n, s in zip(names, spk)), f"{tag}: fold file speaker ids differ"
    fold_saved = np.array([fr[n][0] for n in names])
    fold_rule = stable_gkf(g, 5); fold_sk = sk_gkf(g, 5)
    if not np.array_equal(fold_saved, fold_rule):
        raise SystemExit(f"{tag}: saved fold file and pod tie rule disagree on {int((fold_saved != fold_rule).sum())} clips")
    st = ck.get(tag, {"oof": [None] * len(y), "layers": [None] * 5, "inner": [None] * 5, "inner_sk_equal": [None] * 5,
                      "fold": fold_saved.tolist(), "fold_file": os.path.basename(ff), "fold_file_sha256": sha256(ff),
                      "fold_file_equals_pod_tie_rule": True,
                      "fold_file_equals_this_pod_sklearn": bool(np.array_equal(fold_saved, fold_sk)),
                      "n_clips_sklearn_differs": int((fold_saved != fold_sk).sum())})
    assert st["fold"] == fold_saved.tolist(), f"fold mismatch on resume {tag}"
    for k in range(5):
        if st["layers"][k] is not None: continue
        tr, te = np.where(fold_saved != k)[0], np.where(fold_saved == k)[0]
        sc = Parallel(n_jobs=min(nl, NJ))(delayed(inner)(tr, l, g) for l in range(nl))
        best = int(np.argmax(sc))
        p = fitpred(X[tr][:, best], y[tr], X[te][:, best])
        for i, v in zip(te, p): st["oof"][int(i)] = float(v)
        st["layers"][k] = best; st["inner"][k] = [float(v) for v in sc]
        st["inner_sk_equal"][k] = bool(np.array_equal(stable_gkf(g[tr], 4), sk_gkf(g[tr], 4)))
        ck[tag] = st
        json.dump(ck, open(CK + ".tmp", "w")); os.replace(CK + ".tmp", CK)
        srt = sorted(sc, reverse=True)
        print(f"{tag} fold {k} layer {best} inner {srt[0]:.6f} runner-up {srt[1]:.6f}  {time.time()-t0:.0f}s", flush=True)
    st["auc"] = float(roc_auc_score(y, np.array(st["oof"], float))); ck[tag] = st
    json.dump(ck, open(CK + ".tmp", "w")); os.replace(CK + ".tmp", CK)
    print(f"RESULT {DS} {tag} {KEY} AUC {st['auc']:.6f} layers {st['layers']}", flush=True)

with open(OUTP + f"_{KEY}_nested5_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label"] + [f"p_{t}" for t in TAGS] + [f"fold_{t}" for t in TAGS])
    for i in range(len(y)):
        w.writerow([names[i], spk[i], int(y[i])] + [repr(ck[t]["oof"][i]) for t in TAGS] + [ck[t]["fold"][i] for t in TAGS])
per = [ck[f"seed{s}"]["auc"] for s in range(5)]
res = {"dataset": DS, "stream": KEY, "n": int(len(y)), "n_pos": int(y.sum()), "n_speakers": int(len(np.unique(spk))),
       "n_layers": int(nl), "per_repeat_auc": per, "mean5": float(np.mean(per)),
       "per_repeat_auc_4dp": [round(a, 4) for a in per], "mean5_4dp": round(float(np.mean(per)), 4),
       "single_split_auc": ck["single"]["auc"],
       "tags": {t: {k: ck[t][k] for k in ("auc", "layers", "inner", "inner_sk_equal", "fold_file", "fold_file_sha256",
                                         "fold_file_equals_pod_tie_rule", "fold_file_equals_this_pod_sklearn",
                                         "n_clips_sklearn_differs")} for t in TAGS},
       "states": {"file": os.path.abspath(NPZ), "sha256": sha256(NPZ),
                  "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}},
       "versions": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                    "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
       "n_jobs": NJ, "seconds": round(time.time() - t0, 1), "command": " ".join(sys.argv)}
json.dump(res, open(OUTP + f"_{KEY}_nested5.json", "w"), indent=1)
print(f"NESTED5 DONE {DS} mean5 {res['mean5']:.6f} per {[round(a, 4) for a in per]} single {res['single_split_auc']:.6f}", flush=True)
