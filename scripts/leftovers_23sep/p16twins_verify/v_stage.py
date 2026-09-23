"""Verifier (p16twins): per-stage probe refit, written from scratch for the verification.
usage: v_stage.py NPZ STREAM TAG
For every stage: 5 repeats; repeat r relabels the sorted unique speakers by numpy default_rng(r).permutation,
GroupKFold(5) on those labels, StandardScaler fit on the training fold, LogisticRegression(max_iter=2000,
class_weight='balanced'); OOF probability; AUC per repeat; stage value = mean of the 5 AUCs.
Best stage = first argmax. CI at the best stage: 2000 speaker draws, fresh default_rng(0),
idx = rng.choice(N, N, replace=True) over sorted unique speakers, value per draw = mean over repeats of the AUC.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, time, platform
import numpy as np
from joblib import Parallel, delayed

NPZ, STREAM, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
z = np.load(NPZ, allow_pickle=True)
Xall = np.asarray(z[STREAM], dtype=np.float32)
y = np.asarray(z["label"]).astype(int)
spk = np.asarray(z["spk"]).astype(str)
N, L, D = Xall.shape
print(f"{TAG} {STREAM} X {Xall.shape} n={N} pos={y.sum()} spk={len(np.unique(spk))} load {time.time()-t0:.0f}s", flush=True)


def auc_mw(lab, score):
    """Mann-Whitney AUC by explicit counting with searchsorted (ties count one half)."""
    lab = np.asarray(lab); score = np.asarray(score, dtype=np.float64)
    pos = np.sort(score[lab == 1]); neg = np.sort(score[lab == 0])
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lt = np.searchsorted(neg, pos, side="left")    # negatives strictly below each positive
    le = np.searchsorted(neg, pos, side="right")   # negatives at or below
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg)))


def relabel(r):
    u = np.unique(spk)
    order = np.random.default_rng(r).permutation(u)
    code = {s: i for i, s in enumerate(order)}
    return np.array([code[s] for s in spk])


GROUPS = [relabel(r) for r in range(5)]


def one(X, yv, g):
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    p = np.zeros(len(yv))
    for tr, te in GroupKFold(n_splits=5).split(X, yv, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), yv[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


nj = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 8)
XS = [np.ascontiguousarray(Xall[:, l, :]) for l in range(L)]
jobs = [(l, r) for l in range(L) for r in range(5)]
res = Parallel(n_jobs=nj)(delayed(one)(XS[l], y, GROUPS[r]) for l, r in jobs)
P = np.zeros((L, 5, N))
for (l, r), p in zip(jobs, res):
    P[l, r] = p
A = np.array([[auc_mw(y, P[l, r]) for r in range(5)] for l in range(L)])
M = A.mean(1)
best = int(np.argmax(M))
print(f"fits {time.time()-t0:.0f}s best {best} mean {M[best]:.6f}", flush=True)

u = np.unique(spk)
rows = {s: np.flatnonzero(spk == s) for s in u}
rng = np.random.default_rng(0)
vals = []
for _ in range(2000):
    idx = rng.choice(len(u), size=len(u), replace=True)
    ii = np.concatenate([rows[u[k]] for k in idx])
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    vals.append(np.mean([auc_mw(yy, P[best, r, ii]) for r in range(5)]))
vals = np.array(vals)
lo, hi = np.percentile(vals, [2.5, 97.5])
try:
    import threadpoolctl
    arch = [d.get("architecture") for d in threadpoolctl.threadpool_info()]
except Exception as e:
    arch = str(e)
import sklearn, scipy
cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":")[1].strip()
out = {"tag": TAG, "stream": STREAM, "n": int(N), "stages": int(L), "best_stage": best,
       "best_mean": float(M[best]), "best_ci": [float(lo), float(hi)], "boot_usable": int(len(vals)),
       "best_per_repeat": [float(a) for a in A[best]],
       "stage_means": [float(m) for m in M], "stage_per_repeat": A.tolist(),
       "top5": [[int(i), float(M[i])] for i in np.argsort(-M)[:5]],
       "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
       "python": platform.python_version(), "cpu": cpu, "openblas_arch": arch,
       "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE", "default"), "seconds": round(time.time() - t0, 1)}
os.makedirs("/workspace/out", exist_ok=True)
json.dump(out, open(f"/workspace/out/v_{TAG}.json", "w"), indent=1)
np.savez_compressed(f"/workspace/out/v_{TAG}_best_oof.npz", P=P[best], y=y, spk=spk, name=np.asarray(z["name"]).astype(str))
print(f"DONE {TAG} best {best} {M[best]:.4f} [{lo:.4f}, {hi:.4f}] per_repeat {[round(a,4) for a in A[best]]} top5 {out['top5']} arch {arch} {time.time()-t0:.0f}s", flush=True)
