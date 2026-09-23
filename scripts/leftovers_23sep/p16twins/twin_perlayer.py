"""p16twins: independent per-layer refit of the POD3 best-stage values.
Input: /workspace/in/<ds>_<stream>.npy  (float32, shape n x layers x dim), /workspace/in/<ds>_meta.npz (label, spk, name).
Estimator being twinned (POD3 perlayer_nested.py / perlayer_stream.py, per-layer arm):
  for each layer l and repeat r in 0..4: speaker ids permuted with numpy default_rng(r), GroupKFold(5) on the
  permuted ids, StandardScaler fit on the training fold, LogisticRegression(max_iter=2000, class_weight=balanced),
  out-of-fold P(1); per-repeat AUC by the tie-averaged rank formula; layer value = mean of the 5 per-repeat AUCs.
  best stage = argmax over layers of that mean. CI = speaker bootstrap of the mean-over-repeats AUC, 2000 draws,
  fresh default_rng(0), idx = rng.choice(N, N, replace=True) over the sorted unique speaker ids.
This file is written from scratch; it does not import or copy the POD3 scripts.
usage: python3 twin_perlayer.py DS STREAM"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import sys, json, time, csv
import numpy as np
from multiprocessing import Pool
import sklearn, scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DS, ST = sys.argv[1], sys.argv[2]
R = 5
T0 = time.time()
X = np.load(f"/workspace/in/{DS}_{ST}.npy", mmap_mode="r")
meta = np.load(f"/workspace/in/{DS}_meta.npz", allow_pickle=True)
y = meta["label"].astype(int); spk = meta["spk"].astype(str); name = meta["name"].astype(str)
N, NL, D = X.shape
print(f"{DS} {ST} X {X.shape} {X.dtype} n={N} pos={int(y.sum())} spk={len(np.unique(spk))} "
      f"sklearn {sklearn.__version__} numpy {np.__version__} scipy {scipy.__version__}", flush=True)

def auc_rank(yv, s):
    yv = np.asarray(yv); s = np.asarray(s, dtype=np.float64)
    n1 = int((yv == 1).sum()); n0 = len(yv) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = scipy.stats.rankdata(s)          # average ranks for ties
    return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

USPK = np.unique(spk)
def perm_groups(seed):
    order = np.random.default_rng(seed).permutation(USPK)
    pos = {s: i for i, s in enumerate(order)}
    return np.array([pos[s] for s in spk])
GROUPS = [perm_groups(r) for r in range(R)]

def job(args):
    l, r = args
    Xl = np.ascontiguousarray(X[:, l, :], dtype=np.float32)
    oof = np.zeros(N)
    for tr, te in GroupKFold(n_splits=5).split(Xl, y, groups=GROUPS[r]):
        sc = StandardScaler().fit(Xl[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xl[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(Xl[te]))[:, 1]
    return l, r, oof

jobs = [(l, r) for l in range(NL) for r in range(R)]
with Pool(min(len(jobs), os.cpu_count() or 8)) as p:
    res = p.map(job, jobs, chunksize=1)
OOF = np.zeros((NL, R, N))
for l, r, o in res: OOF[l, r] = o
print(f"fits done {time.time()-T0:.0f}s", flush=True)

per = np.array([[auc_rank(y, OOF[l, r]) for r in range(R)] for l in range(NL)])
means = per.mean(1)
best = int(np.argmax(means))

def boot_mean_auc(oofs):
    rng = np.random.default_rng(0)
    rows = {s: np.flatnonzero(spk == s) for s in USPK}
    vals = []
    for _ in range(2000):
        idx = rng.choice(len(USPK), size=len(USPK), replace=True)
        ii = np.concatenate([rows[USPK[k]] for k in idx])
        yy = y[ii]
        if yy.min() == yy.max(): continue
        vals.append(np.mean([auc_rank(yy, o[ii]) for o in oofs]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

lo, hi, nd = boot_mean_auc(OOF[best])
os.makedirs("/workspace/out", exist_ok=True)
with open(f"/workspace/out/twin_{DS}_{ST}_perlayer.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["stage", "auc_mean", "per_repeat"])
    for l in range(NL): w.writerow([l, f"{means[l]:.6f}", "|".join(f"{a:.6f}" for a in per[l])])
np.savez_compressed(f"/workspace/out/twin_{DS}_{ST}_best_oof.npz", oof=OOF[best], label=y, spk=spk, name=name, stage=best)
out = {"dataset": DS, "stream": ST, "n": int(N), "n_speakers": int(len(USPK)), "n_layers": int(NL),
       "best_stage": best, "best_stage_auc_mean": float(means[best]), "best_ci": [lo, hi], "boot_usable": nd,
       "best_per_repeat": [float(a) for a in per[best]],
       "all_means": [float(m) for m in means],
       "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
       "python": sys.version.split()[0], "runtime_s": round(time.time() - T0, 1)}
json.dump(out, open(f"/workspace/out/twin_{DS}_{ST}.json", "w"), indent=1)
print(f"BEST {DS} {ST} stage {best} mean {means[best]:.6f} [{lo:.6f},{hi:.6f}] per_repeat {np.round(per[best],4).tolist()} "
      f"{time.time()-T0:.0f}s", flush=True)
