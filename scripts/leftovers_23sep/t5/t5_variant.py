"""Variant refit of the per-layer curve with an explicit fold file: the probe lines are those of
curves_from_states.py (StandardScaler fit on the training fold, LogisticRegression(max_iter=2000,
class_weight='balanced'), roc_auc_score per fold and on the pooled out-of-fold vector).
usage: t5_variant.py JOB STATES.npz FOLDFILE TAG OUTDIR [f32|f64] [NJOBS]
Environment variables (threads, OPENBLAS_CORETYPE) are set by the caller and recorded."""
import os, sys, csv, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
job, npz, ffile, tag, outd = sys.argv[1:6]
dt = sys.argv[6] if len(sys.argv) > 6 else "f32"; nj = int(sys.argv[7]) if len(sys.argv) > 7 else max(2, (os.cpu_count() or 4) - 2)
z = np.load(npz, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); names = [str(v) for v in z["name"]]
m = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(ffile))}
fold = np.array([m[n] for n in names])
splits = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
def layer_oof(X, l):
    oof = np.zeros(len(y)); aucs = []
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr, l])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
        p = clf.predict_proba(sc.transform(X[te, l]))[:, 1]; oof[te] = p
        aucs.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
    return (l, float(np.nanmean(aucs)), float(np.nanstd(aucs)), float(roc_auc_score(y, oof)), oof)
t0 = time.time(); meta = dict(job=job, tag=tag, fold_file=ffile, dtype=dt, n_jobs=nj,
                              env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE")})
oofs = {}
for key in [k for k in ("enc", "proj", "llm", "ans") if k in z.files]:
    X = z[key].astype(np.float32 if dt == "f32" else np.float64)
    rows = sorted(Parallel(n_jobs=nj)(delayed(layer_oof)(X, l) for l in range(X.shape[1])), key=lambda r: r[0])
    with open(f"{outd}/{job}__{tag}_{key}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["layer", "auc_mean", "auc_std", "auc_oof"]); w.writerows([r[:4] for r in rows])
    oofs[key] = np.stack([r[4] for r in rows], axis=1)
np.savez_compressed(f"{outd}/{job}__{tag}_oof.npz", name=np.array(names), label=y, fold=fold, **oofs)
meta["seconds"] = round(time.time() - t0, 1)
json.dump(meta, open(f"{outd}/{job}__{tag}_meta.json", "w"), indent=1)
print("VARIANT DONE", job, tag, meta["seconds"], "s", flush=True)
