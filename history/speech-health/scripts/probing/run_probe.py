import os, csv, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    L = int(d["n_layers"]); spk = d["speaker"]; lab = d["label"].astype(int); task = d["task"]
    grp = d["group"] if "group" in d.files else np.array([""]*len(lab))
    mask = np.ones(len(lab), bool)
    if a.tasks:
        keep = set(a.tasks.split(",")); mask &= np.array([t in keep for t in task])
    if a.groups:
        kg = set(a.groups.split(",")); mask &= np.array([g in kg for g in grp])
    idx = np.where(mask)[0]
    y = lab[idx]; groups = spk[idx]
    ngrp = len(set(groups)); nf = min(a.folds, ngrp)
    print(f"filter tasks={a.tasks or 'ALL'} groups={a.groups or 'ALL'} n={len(idx)} speakers={ngrp} PD={int(y.sum())} HC={int((y==0).sum())} folds={nf}", flush=True)
    rows = []
    for li in range(L):
        X = d[f"layer_{li:02d}"][idx]
        bas, aucs = [], []
        for tr, te in GroupKFold(n_splits=nf).split(X, y, groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
            bas.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
            try: aucs.append(roc_auc_score(y[te], p))
            except Exception: pass
        bm = float(np.mean(bas)) if bas else float("nan"); bs = float(np.std(bas)) if bas else float("nan")
        am = float(np.mean(aucs)) if aucs else float("nan"); asd = float(np.std(aucs)) if aucs else float("nan")
        rows.append((li, bm, bs, am, asd))
        print(f"  layer {li:02d}  bAcc {bm:.3f}+/-{bs:.3f}  AUC {am:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["layer","balacc_mean","balacc_std","auc_mean","auc_std"]); w.writerows(rows)
    valid=[r for r in rows if not np.isnan(r[3])]
    if valid:
        best = max(valid, key=lambda r: r[3])
        print(f"PEAK layer {best[0]}: bAcc {best[1]:.3f} AUC {best[3]:.3f} -> {a.out}", flush=True)

if __name__ == "__main__": main()
