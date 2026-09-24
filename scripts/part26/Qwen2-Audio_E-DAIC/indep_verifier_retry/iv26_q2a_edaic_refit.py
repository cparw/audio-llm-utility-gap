#!/usr/local/bin/python3
"""Extra independent check: redo the nested probe on the Mac from the saved states and saved fold files.
Inner folds: my own implementation of GroupKFold(4) with the stable tie order (all speakers have one clip).
Model: StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'). Mac sklearn, not the pod build,
so small numeric differences are expected. Reports layer agreement and max abs OOF difference vs the per-clip file."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import csv, json, sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CELL = "scores/part26/Qwen2-Audio_E-DAIC"
PERCLIP = f"{CELL}/Qwen2-Audio_E-DAIC_perclip.csv"
PODJSON = f"{CELL}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5.json"
STATES = "<local data dir>/paper1_local_runs/probe2/edaic_states.npz"
FOLDDIR = "<local data dir>/release/folds"

pc = list(csv.DictReader(open(PERCLIP, newline="")))
st = np.load(STATES, allow_pickle=False); nm = [str(x) for x in st["name"]]; ix = {n: i for i, n in enumerate(nm)}
order = [ix[r["clip"]] for r in pc]
X = st["enc"][order].astype(np.float32); y = np.array([int(r["label"]) for r in pc]); spk = np.array([r["speaker"] for r in pc])
assert np.array_equal(st["label"][order], y)
L = X.shape[1]
pj = json.load(open(PODJSON))

def fitpred(Xtr, ytr, Xte):
    s = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(Xtr), ytr).predict_proba(s.transform(Xte))[:, 1]

def gkf_stable(gid, k):
    ug, cnt = np.unique(gid, return_counts=True); load = np.zeros(k); m = {}
    for gi in np.argsort(cnt, kind="stable")[::-1]:
        j = int(np.argmin(load)); m[ug[gi]] = j; load[j] += cnt[gi]
    return np.array([m[g] for g in gid])

def inner_score(tr, gid_tr, l):
    f = gkf_stable(gid_tr, 4); s = 0.0
    for j in range(4):
        a, b = tr[f != j], tr[f == j]
        s += roc_auc_score(y[b], fitpred(X[a, l], y[a], X[b, l])) if len(set(y[b])) > 1 else 0.5
    return s / 4

res = {}
for tag in [f"seed{s}" for s in range(5)] + ["single"]:
    fn = f"{FOLDDIR}/edaic_groupkfold5_pod_seed{tag[4:]}_UNVERIFIED.csv" if tag != "single" else f"{FOLDDIR}/edaic_groupkfold5_pod.csv"
    fd = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(fn, newline=""))}
    fold = np.array([fd[r["clip"]] for r in pc])
    u = np.unique(spk)
    if tag != "single":
        perm = np.random.default_rng(int(tag[4:])).permutation(u); g = {s: i for i, s in enumerate(perm)}
    else:
        g = {s: i for i, s in enumerate(u)}
    gid = np.array([g[s] for s in spk])
    oof = np.zeros(len(y)); layers = []; inner_best = []
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        sc = Parallel(n_jobs=8)(delayed(inner_score)(tr, gid[tr], l) for l in range(L))
        best = int(np.argmax(sc)); layers.append(best); srt = sorted(sc, reverse=True); inner_best.append([srt[0], srt[1]])
        oof[te] = fitpred(X[tr, best], y[tr], X[te, best])
    saved = np.array([float(r[f"p_probe_{tag}"]) for r in pc])
    res[tag] = {"layers_mac": layers, "layers_pod": pj["tags"][tag]["layers"], "layers_equal": layers == pj["tags"][tag]["layers"],
                "auc_mac": roc_auc_score(y, oof), "auc_saved": roc_auc_score(y, saved),
                "oof_maxabs_vs_perclip": float(np.max(np.abs(oof - saved))), "inner_top2": inner_best}
    print(tag, json.dumps(res[tag]), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
