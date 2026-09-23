"""MAC COPY (spawn-safe Pool initializer; same fit and AUC code). Independent verifier refit (own code, t5 verify). Per layer: StandardScaler fit on the training fold,
LogisticRegression(max_iter=2000, class_weight='balanced') (the model the saved curves name), predict_proba on the
held out fold. Folds come from my own GroupKFold reimplementation (stable tie order) unless a fold file is given.
AUC: own midrank Mann Whitney AUC, plus sklearn roc_auc_score on the pooled vector for a bit-level comparison.
usage: v5_refit.py NPZ STREAMS(enc,proj) FOLDS(mine|mine_default|<file.csv>) DTYPE(f32|f64) NPROC TAG OUTDIR
Thread counts and OPENBLAS_CORETYPE come from the caller's environment."""
import os, sys, json, time, csv, warnings; warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NPZ, STREAMS, FOLDS, DT, NP, TAG, OUT = sys.argv[1:8]; NP = int(NP)
z = np.load(NPZ); y = z["label"].astype(np.int64); spk = z["spk"].astype(str); names = [str(v) for v in z["name"]]

def my_groupkfold(groups, k=5, kind="stable"):
    uniq, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    order = np.argsort(cnt, kind=kind)[::-1]
    load = np.zeros(k); g2f = np.zeros(len(uniq), dtype=np.int64)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    return g2f[inv]

if FOLDS == "mine": fold = my_groupkfold(spk, kind="stable")
elif FOLDS == "mine_default": fold = my_groupkfold(spk, kind="quicksort")
else:
    m = {}
    with open(FOLDS) as fh:
        for r in csv.DictReader(fh): m[r["clip_id"]] = int(r["fold"])
    fold = np.array([m[n] for n in names], dtype=np.int64)

def auc_rank(t, s):
    t = np.asarray(t); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(s, kind="mergesort"); ss = s[o]; r = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(t.sum()); nneg = len(t) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return float((r[t == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))

X = None
def _init(key, dt):
    global X
    X = z[key].astype(np.float32 if dt == 'f32' else np.float64)
def fit_layer(l):
    oof = np.zeros(len(y)); fa = []
    for k in range(5):
        te = np.flatnonzero(fold == k); tr = np.flatnonzero(fold != k)
        Xtr = X[tr, l]; Xte = X[te, l]
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y[tr])
        p = m.predict_proba(sc.transform(Xte))[:, 1]; oof[te] = p
        fa.append(auc_rank(y[te], p))
    fa = np.array([a for a in fa if a == a]); mu = fa.sum() / len(fa); sd = (((fa - mu) ** 2).sum() / len(fa)) ** 0.5
    return l, mu, sd, auc_rank(y, oof), float(roc_auc_score(y, oof)), oof

if __name__ == "__main__":
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    meta = dict(npz=NPZ, tag=TAG, folds=FOLDS, dtype=DT, nproc=NP, n=len(y),
                env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE")},
                fold_sizes=np.bincount(fold, minlength=5).tolist())
    oofs = {}
    for key in STREAMS.split(","):
        X = z[key].astype(np.float32 if DT == "f32" else np.float64)
        with Pool(NP, initializer=_init, initargs=(key, DT)) as pool: rows = sorted(pool.map(fit_layer, range(X.shape[1])), key=lambda r: r[0])
        with open(f"{OUT}/{TAG}_{key}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["layer", "auc_mean", "auc_std", "auc_oof_rank", "auc_oof_sk"])
            for r in rows: w.writerow([r[0], repr(r[1]), repr(r[2]), repr(r[3]), repr(r[4])])
        oofs[key] = np.stack([r[5] for r in rows], axis=1)
        print(TAG, key, "done", round(time.time() - t0, 1), "s", flush=True)
    np.savez_compressed(f"{OUT}/{TAG}_oof.npz", name=np.array(names), label=y, fold=fold, **oofs)
    meta["seconds"] = round(time.time() - t0, 1)
    json.dump(meta, open(f"{OUT}/{TAG}_meta.json", "w"), indent=1)
    print("REFIT DONE", TAG, meta["seconds"], flush=True)
