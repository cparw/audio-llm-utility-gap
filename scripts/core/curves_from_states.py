"""Per-layer probe curves from a saved states npz: out-of-fold AUC at every fixed layer, GroupKFold(5) by speaker.
sklearn LogisticRegression l2 C=1 balanced 2000 iter, StandardScaler fit inside each training fold.
usage: curves_from_states.py STATES.npz OUTPREFIX"""
import os; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np, csv, sys, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
NPZ, OUTP = sys.argv[1:3]
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str)
def layer_oof(X, l):
    oof = np.zeros(len(y)); aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=spk):
        sc = StandardScaler().fit(X[tr, l])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
        p = clf.predict_proba(sc.transform(X[te, l]))[:, 1]; oof[te] = p
        aucs.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
    return (l, float(np.nanmean(aucs)), float(np.nanstd(aucs)), float(roc_auc_score(y, oof)))
for key, fname, colname in (("enc", "encoder_perlayer.csv", "layer"), ("proj", "proj_perlayer.csv", "stage"),
                            ("llm", "llm_perlayer.csv", "stage"), ("ans", "ans_perlayer.csv", "stage")):
    if key not in z.files: continue
    X = z[key].astype(np.float32)
    rows = sorted(Parallel(n_jobs=max(2, os.cpu_count() - 2))(delayed(layer_oof)(X, l) for l in range(X.shape[1])))
    with open(f"{OUTP}_{fname}", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow([colname, "auc_mean", "auc_std", "auc_oof"]); w.writerows(rows)
    best = max(rows, key=lambda r: r[3])
    print(f"{key}: {X.shape[1]} points, best auc_oof {best[3]:.4f} at {best[0]}, last {rows[-1][3]:.4f}", flush=True)
print("CURVES DONE", flush=True)
