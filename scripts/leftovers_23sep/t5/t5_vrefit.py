"""Independent verifier refit (written separately from the main run's scripts).
Folds come from the released pod-order fold file, matched to the states by clip name (no GroupKFold call);
fold AUCs and the pooled AUC use a Mann-Whitney rank formula (scipy rankdata, ties averaged), not roc_auc_score.
The probe itself must be the paper's probe: StandardScaler on the training fold, then
LogisticRegression(max_iter=2000, class_weight='balanced'), predict_proba column 1.
usage: t5_vrefit.py JOB STATES.npz FOLDFILE OUTDIR"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, csv, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

JOB, NPZ, FOLDF, OUTD = sys.argv[1:5]
with np.load(NPZ, allow_pickle=True) as _z:
    DATA = {k: np.array(_z[k]) for k in _z.files}   # read once in the parent; forked workers share it
LAB = np.asarray(DATA["label"]).astype(int)
CLIPS = [str(c) for c in DATA["name"]]
fmap = {}
with open(FOLDF) as fh:
    for row in csv.DictReader(fh):
        fmap[row["clip_id"]] = int(row["fold"])
assert all(c in fmap for c in CLIPS), "a clip is missing from the fold file"
FOLD = np.array([fmap[c] for c in CLIPS], dtype=int)

def mw_auc(lab, score):
    lab = np.asarray(lab); score = np.asarray(score, dtype=float)
    npos = int((lab == 1).sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    rk = rankdata(score)
    return float((rk[lab == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))

def one_layer(args):
    key, li = args
    X = DATA[key][:, li, :].astype(np.float32)
    pred = np.empty(len(LAB)); per = []
    for f in sorted(set(FOLD.tolist())):
        te = FOLD == f; tr = ~te
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X[tr]), LAB[tr])
        pte = model.predict_proba(scaler.transform(X[te]))[:, 1]
        pred[te] = pte
        per.append(mw_auc(LAB[te], pte))
    per = np.array(per, dtype=float)
    return key, li, float(np.nanmean(per)), float(np.nanstd(per)), mw_auc(LAB, pred), pred

keys = [k for k in ("enc", "proj", "llm", "ans") if k in DATA]
tasks = [(k, li) for k in keys for li in range(DATA[k].shape[1])]
import multiprocessing as _mp
with ProcessPoolExecutor(max_workers=min(30, len(tasks)), mp_context=_mp.get_context("fork")) as ex:
    res = list(ex.map(one_layer, tasks))
summary = {"job": JOB, "fold_file": FOLDF, "n": int(len(LAB)), "fold_sizes": np.bincount(FOLD).tolist(), "keys": keys}
for k in keys:
    rows = sorted([r for r in res if r[0] == k], key=lambda r: r[1])
    with open(f"{OUTD}/{JOB}__vrefit_{k}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["layer", "auc_mean", "auc_std", "auc_oof"])
        for r in rows: w.writerow([r[1], repr(r[2]), repr(r[3]), repr(r[4])])
    np.save(f"{OUTD}/{JOB}__vrefit_{k}_oof.npy", np.stack([r[5] for r in rows], axis=1))
json.dump(summary, open(f"{OUTD}/{JOB}__vrefit.json", "w"), indent=1)
print("VREFIT DONE", JOB, keys, flush=True)
