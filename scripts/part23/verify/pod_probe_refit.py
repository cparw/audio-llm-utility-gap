"""PART 23 verifier, item 3: independent refit of the nested probe from saved states (runs on the pod).
Outer folds: read from the released fold files edaic_full_groupkfold5_shuffle_rs{r}.csv (fold = test fold).
Also checks those files equal sklearn GroupKFold(5, shuffle=True, random_state=r) on this pod.
Inner layer choice: GroupKFold(4, shuffle=True, random_state=r) on the outer-train rows, argmax of the
inner out-of-fold AUC (explicit Mann-Whitney). Probe: StandardScaler(train) + LogisticRegression(
max_iter=2000, class_weight='balanced', C=1.0), float32 features.
usage: python3 pod_probe_refit.py <shards_dir> <clip_list_csv(pid,label)> <folds_dir> <out_prefix> stage [stage ...]
"""
import sys, csv, json, time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

SH, CL, FD, OUT = sys.argv[1:5]
stages = sys.argv[5:]
rows = list(csv.DictReader(open(CL)))
pids = [r["pid"] for r in rows]
y = np.array([int(r["label"]) for r in rows])
spk = np.array(pids)


def auc(yy, s):
    p = s[yy == 1]; n = s[yy == 0]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def fit_pred(X, yy, tr, te):
    ss = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(ss.transform(X[tr]), yy[tr])
    return lr.predict_proba(ss.transform(X[te]))[:, 1]


folds = []
fold_check = []
for r in range(5):
    f = {row["speaker_id"]: int(row["fold"]) for row in csv.DictReader(open(f"{FD}/edaic_full_groupkfold5_shuffle_rs{r}.csv"))}
    fv = np.array([f[p] for p in pids])
    folds.append(fv)
    sk = np.full(len(pids), -1)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(np.zeros(len(y)), y, groups=spk)):
        sk[te] = k
    fold_check.append(int((sk == fv).sum()))
print("fold file == sklearn GroupKFold on this pod (clips agreeing per repeat):", fold_check, flush=True)

res = {"fold_check": fold_check}
for st in stages:
    t0 = time.time()
    X = np.stack([np.load(f"{SH}/{p}.npz")[st] for p in pids]).astype(np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    nL = X.shape[1]
    per, picks, oofs = [], [], []
    for r in range(5):
        oof = np.zeros(len(y))
        for k in range(5):
            te = np.where(folds[r] == k)[0]; tr = np.where(folds[r] != k)[0]
            if nL == 1:
                best = 0
            else:
                Xtr, ytr = X[tr], y[tr]
                isp = list(GroupKFold(n_splits=4, shuffle=True, random_state=r).split(Xtr, ytr, groups=spk[tr]))

                def score(L):
                    o = np.zeros(len(tr))
                    for a, b in isp:
                        o[b] = fit_pred(Xtr[:, L, :], ytr, a, b)
                    return auc(ytr, o)
                sc = Parallel(n_jobs=16, prefer="processes")(delayed(score)(L) for L in range(nL))
                best = int(np.argmax(sc))
            picks.append(best)
            oof[te] = fit_pred(X[:, best, :], y, tr, te)
        per.append(auc(y, oof)); oofs.append(oof)
        print(st, "repeat", r, round(per[-1], 6), picks[-5:], round(time.time() - t0), flush=True)
    res[st] = dict(per_repeat=per, mean=float(np.mean(per)), layers=picks, n_layers=nL, dim=int(X.shape[2]))
    with open(f"{OUT}_{st}_oof.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["pid", "label"] + [f"oof_r{r}" for r in range(5)])
        for i, p in enumerate(pids):
            w.writerow([p, y[i]] + [repr(float(oofs[r][i])) for r in range(5)])
    json.dump(res, open(f"{OUT}_summary.json", "w"), indent=1)
print("DONE", flush=True)
